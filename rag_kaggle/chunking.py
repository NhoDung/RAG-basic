from __future__ import annotations

import re

from .config import CHUNKER_VERSION, ChunkingConfig
from .models import Block, ChildChunk, ParentContext, ParsedDocument
from .utils import estimate_tokens, excel_column_name, rows_to_markdown, stable_id


TEXT_TYPES = {"text", "ocr_page", "caption", "heading"}
VISUAL_TYPES = {"image", "chart", "flowchart"}
BULLET_PATTERN = re.compile(r"^\s*([-•*+▪◦●]|\d+[.)]|[a-zđ][.)])\s+", re.IGNORECASE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:…])\s+(?=[A-ZÀ-Ỹ0-9\"“(])")


class ParentChildChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, document: ParsedDocument) -> tuple[list[ParentContext], list[ChildChunk]]:
        parents = self._build_parents(document)
        blocks_by_id = {block.block_id: block for block in document.blocks}
        children: list[ChildChunk] = []
        for parent in parents:
            parent_blocks = [blocks_by_id[block_id] for block_id in parent.block_ids]
            children.extend(self._chunk_parent(document, parent, parent_blocks))
        return parents, children

    # ----------------------------------------------------------------- parents

    def _build_parents(self, document: ParsedDocument) -> list[ParentContext]:
        parents = []
        current_blocks: list[Block] = []
        current_size = 0
        current_key = None

        for block in sorted(document.blocks, key=lambda item: item.reading_order):
            key = (
                tuple(block.section_path),
                block.sheet_name,
                block.page if not block.section_path and not block.sheet_name else None,
            )
            keep_with_previous = bool(block.metadata.get("derived_from")) and current_blocks
            should_flush = current_blocks and not keep_with_previous and (
                key != current_key or current_size + len(block.content) > self.config.parent_chars
            )
            if should_flush:
                parents.append(self._make_parent(document, current_blocks, len(parents)))
                current_blocks = []
                current_size = 0
            current_key = key
            current_blocks.append(block)
            current_size += len(block.content)

        if current_blocks:
            parents.append(self._make_parent(document, current_blocks, len(parents)))
        return parents

    def _make_parent(self, document, blocks, ordinal):
        section_path = next((block.section_path for block in blocks if block.section_path), [])
        labels = [
            f"Document: {document.source_file}",
            f"Section: {' > '.join(section_path)}" if section_path else "",
        ]
        content_parts = [label for label in labels if label]
        content_parts.extend(f"[{block.block_type.upper()}]\n{block.content}" for block in blocks)
        parent_id = stable_id(document.document_id, "parent", ordinal, *(block.block_id for block in blocks))
        return ParentContext(
            parent_id=parent_id,
            document_id=document.document_id,
            source_file=document.source_file,
            content="\n\n".join(content_parts),
            block_ids=[block.block_id for block in blocks],
            section_path=list(section_path),
            metadata={
                "pages": sorted({block.page for block in blocks if block.page is not None}),
                "sheets": sorted({block.sheet_name for block in blocks if block.sheet_name}),
                "reading_order": min(block.reading_order for block in blocks),
            },
        )

    # ---------------------------------------------------------------- children

    def _chunk_parent(self, document, parent, blocks) -> list[ChildChunk]:
        chunks: list[ChildChunk] = []
        text_run: list[Block] = []
        has_content = any(block.block_type != "heading" for block in blocks)

        def flush_text():
            if text_run:
                chunks.extend(self._chunk_text_run(document, parent, list(text_run), len(chunks)))
                text_run.clear()

        for block in blocks:
            if block.block_type == "heading" and has_content:
                continue  # Headings are already part of every child's section prefix.
            if block.block_type == "caption" and block.metadata.get("attached_to"):
                continue  # Embedded together with the figure/table it describes.
            if block.block_type in TEXT_TYPES:
                text_run.append(block)
                continue
            flush_text()
            if block.block_type == "table" and block.metadata.get("rows"):
                chunks.extend(self._chunk_table(document, parent, block))
            elif block.block_type in VISUAL_TYPES:
                chunks.extend(self._chunk_visual(document, parent, block))
            else:  # kpi, unstructured tables and anything new
                chunks.extend(self._chunk_single(document, parent, block, block.content))
        flush_text()
        return chunks

    def _chunk_text_run(self, document, parent, blocks: list[Block], start_index: int) -> list[ChildChunk]:
        groups: list[tuple[list[Block], str]] = []
        current: list[Block] = []
        current_text = ""
        for block in blocks:
            if len(block.content) > self.config.child_chars:
                if current:
                    groups.append((current, current_text))
                    current, current_text = [], ""
                for piece in split_text(block.content, self.config.child_chars, 0):
                    groups.append(([block], piece))
                continue
            candidate = f"{current_text}\n\n{block.content}".strip()
            if current and len(candidate) > self.config.child_chars:
                groups.append((current, current_text))
                current, candidate = [], block.content
            current.append(block)
            current_text = candidate
        if current:
            groups.append((current, current_text))

        chunks = []
        previous_text = ""
        for index, (group_blocks, text) in enumerate(groups):
            body = text
            if previous_text and self.config.child_overlap_chars > 0:
                body = f"{overlap_tail(previous_text, self.config.child_overlap_chars)}\n{text}"
            previous_text = text
            content = self._prefix(document, group_blocks[0], "text") + body
            chunks.append(
                self._make_child(document, parent, group_blocks, f"text-{start_index + index}", content, "text")
            )
        return chunks

    def _chunk_table(self, document, parent, block: Block) -> list[ChildChunk]:
        rows = block.metadata["rows"]
        inherited = block.metadata.get("inherited_header")
        header = inherited or rows[0]
        body = rows if inherited else rows[1:]
        if not body:
            body, header = rows, None
        width = max(len(row) for row in rows)
        header_cells = list(header or []) + [""] * (width - len(header or []))
        column_groups = self._column_groups(width)
        origin = block.metadata.get("origin")
        title = block.metadata.get("caption") or ""
        notes = block.metadata.get("references") or []

        chunks = []
        size = self.config.table_rows_per_chunk
        for row_start in range(0, len(body), size):
            group_rows = body[row_start : row_start + size]
            for group_index, columns in enumerate(column_groups):
                selected = [[_cell(row, column) for column in columns] for row in group_rows]
                table_rows = ([[header_cells[column] for column in columns]] if header else []) + selected
                lines = []
                if title:
                    lines.append(f"Table title: {title}")
                if inherited:
                    lines.append("(Phần tiếp theo của bảng ở trang trước)")
                if len(column_groups) > 1:
                    lines.append(f"Columns {group_index + 1}/{len(column_groups)} (cột định danh được lặp lại)")
                lines.append(rows_to_markdown(table_rows))
                if notes:
                    lines.append("Referenced by: " + " | ".join(notes))
                content = self._prefix(document, block, "table") + "\n".join(lines)
                cell_range = block.cell_range
                data_offset = 0 if inherited or not header else 1
                if origin:
                    first_row = origin[0] + data_offset + row_start
                    last_row = first_row + len(group_rows) - 1
                    first_col = origin[1] + min(columns)
                    last_col = origin[1] + max(columns)
                    cell_range = f"{excel_column_name(first_col)}{first_row}:{excel_column_name(last_col)}{last_row}"
                chunks.append(
                    self._make_child(
                        document,
                        parent,
                        [block],
                        f"rows-{row_start}-{row_start + len(group_rows) - 1}-cols-{group_index}",
                        content,
                        "table",
                        cell_range=cell_range,
                        metadata={
                            "row_start": row_start,
                            "row_end": row_start + len(group_rows) - 1,
                            "columns": list(columns),
                        },
                    )
                )
        return chunks

    def _column_groups(self, width: int) -> list[list[int]]:
        limit = max(self.config.table_max_columns_per_chunk, self.config.table_key_columns + 1)
        if width <= limit:
            return [list(range(width))]
        keys = list(range(min(self.config.table_key_columns, width)))
        others = [column for column in range(width) if column not in keys]
        step = limit - len(keys)
        return [keys + others[start : start + step] for start in range(0, len(others), step)]

    def _chunk_visual(self, document, parent, block: Block) -> list[ChildChunk]:
        lines = []
        if block.metadata.get("caption"):
            lines.append(f"Figure title/caption: {block.metadata['caption']}")
        image_type = block.metadata.get("image_type", block.block_type)
        if block.block_type == "image":
            lines.append(f"Image type: {image_type}")
            lines.append(f"OCR text:\n{block.content}")
        else:
            lines.append(block.content)
        if block.metadata.get("references"):
            lines.append("Đoạn văn tham chiếu: " + " | ".join(block.metadata["references"]))
        chunk_type = image_type if block.block_type == "image" and image_type in VISUAL_TYPES else block.block_type
        return self._chunk_single(document, parent, block, "\n".join(lines), chunk_type)

    def _chunk_single(self, document, parent, block, text, chunk_type=None) -> list[ChildChunk]:
        chunk_type = chunk_type or block.block_type
        pieces = split_text(text, self.config.child_chars, self.config.child_overlap_chars) or [text]
        return [
            self._make_child(
                document, parent, [block], f"{chunk_type}-{index}", self._prefix(document, block, chunk_type) + piece, chunk_type
            )
            for index, piece in enumerate(pieces)
            if piece.strip()
        ]

    def _prefix(self, document: ParsedDocument, block: Block, content_type: str) -> str:
        lines = [f"Document: {document.source_file}"]
        if block.section_path:
            lines.append(f"Section: {' > '.join(block.section_path)}")
        if block.page is not None:
            lines.append(f"Page: {block.page}")
        if block.sheet_name:
            lines.append(f"Sheet: {block.sheet_name}")
        if block.cell_range:
            lines.append(f"Range: {block.cell_range}")
        lines.append(f"Content type: {content_type}")
        return "\n".join(lines) + "\n\n"

    def _make_child(self, document, parent, blocks, position, content, chunk_type, cell_range=None, metadata=None):
        first = blocks[0]
        pages = [block.page for block in blocks if block.page is not None]
        return ChildChunk(
            chunk_id=stable_id(parent.parent_id, *(block.block_id for block in blocks), position),
            parent_id=parent.parent_id,
            document_id=document.document_id,
            source_file=document.source_file,
            chunk_type=chunk_type,
            content=content,
            block_ids=[block.block_id for block in blocks],
            section_path=list(first.section_path),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            sheet_name=first.sheet_name,
            cell_range=cell_range or first.cell_range,
            asset_path=first.asset_path,
            metadata=metadata or {},
            token_count=estimate_tokens(content, self.config.chars_per_token),
            parser_version=document.parser_version,
            chunker_version=CHUNKER_VERSION,
        )


def _cell(row: list[str], column: int) -> str:
    return row[column] if column < len(row) else ""


def overlap_tail(text: str, overlap_chars: int) -> str:
    tail = text[-overlap_chars:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) // 2 else tail


def split_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split by paragraph -> list/line -> sentence -> word, never cutting a bullet item."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    units: list[str] = []
    for paragraph in (part.strip() for part in text.split("\n\n")):
        if paragraph:
            units.extend(_split_unit(paragraph, target_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        separator = "\n" if BULLET_PATTERN.match(unit) or unit.startswith("|") else "\n\n"
        candidate = f"{current}{separator}{unit}" if current else unit
        if len(candidate) <= target_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = unit
    if current:
        chunks.append(current)

    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        overlapped.append(f"{overlap_tail(previous, overlap_chars)}\n{current}")
    return overlapped


def _split_unit(paragraph: str, target_chars: int) -> list[str]:
    if len(paragraph) <= target_chars:
        return [paragraph]
    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    if len(lines) > 1:
        # Keep each bullet item (with its wrapped continuation lines) intact.
        items: list[str] = []
        for line in lines:
            if items and not BULLET_PATTERN.match(line) and BULLET_PATTERN.match(items[-1]) is not None:
                items[-1] = f"{items[-1]} {line}"
            else:
                items.append(line)
        result = []
        for item in items:
            result.extend(_split_unit(item, target_chars) if len(item) > target_chars else [item])
        return result
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(paragraph) if sentence.strip()]
    if len(sentences) > 1:
        result, current = [], ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > target_chars:
                result.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            result.append(current)
        return [piece for sentence in result for piece in _hard_split(sentence, target_chars)]
    return _hard_split(paragraph, target_chars)


def _hard_split(text: str, target_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start + target_chars // 2, end)
            end = space if space > start else end
        pieces.append(text[start:end].strip())
        start = end
    return [piece for piece in pieces if piece]
