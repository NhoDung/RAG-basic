from __future__ import annotations

import re
from typing import Any

from .config import CHUNKER_VERSION, ChunkingConfig
from .knowledge import match_terms, profile_terms
from .models import Block, ChildChunk, ParentContext, ParsedDocument
from .utils import estimate_tokens, rows_to_markdown, stable_id


TEXT_TYPES = {"text", "ocr_page", "caption", "heading"}
VISUAL_TYPES = {"image", "chart", "flowchart"}
BULLET_PATTERN = re.compile(r"^\s*([-•*+▪◦●]|\d+[.)]|[a-zđ][.)])\s+", re.IGNORECASE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:…])\s+(?=[A-ZÀ-Ỹ0-9\"“(])")
MAX_CHUNK_ENTITIES = 8
MAX_CHUNK_KEYWORDS = 8
SENTENCE_END = (".", "!", "?", ";", ":", "…")
# A physical line at least this long that does not end a sentence is a wrapped line of a paragraph;
# shorter lines are list entries, names or labels and keep their own line.
WRAP_MIN_LINE_CHARS = 40


def table_header_and_body(block: Block) -> tuple[list[str] | None, list[list[str]]]:
    """Header row (own or inherited from the previous page) and the data rows of a table block."""
    rows = block.metadata.get("rows") or []
    inherited = block.metadata.get("inherited_header")
    if inherited:
        return list(inherited), rows
    if len(rows) > 1:
        return rows[0], rows[1:]
    return None, rows


def is_large_table(block: Block, config: ChunkingConfig) -> bool:
    """A table that is not stored inline: it becomes a preview chunk + a full .xlsx file."""
    header, body = table_header_and_body(block)
    if not body:
        return False
    if len(body) > config.table_inline_max_rows:
        return True
    return len(rows_to_markdown(([header] if header else []) + body)) > config.table_inline_max_chars


class StructureAwareChunker:
    """Chunk along the document structure instead of a fixed window.

    * paragraph / sentence group -> one text chunk (short neighbours of the same section
      are merged, a paragraph longer than ``max_text_chars`` is split on sentences);
    * table -> one chunk with the whole Markdown table; a table too large to inline is
      one *preview* chunk (header + first rows/columns + key column values) and the full
      table lives in an .xlsx file;
    * image / chart / flowchart -> one chunk (OCR text + caption + references).

    A *section* (stored in the ``parents`` table) is the heading-delimited unit used to
    give the answer model surrounding context; it is never embedded.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(
        self, document: ParsedDocument, profile: dict[str, Any] | None = None
    ) -> tuple[list[ParentContext], list[ChildChunk]]:
        sections = self._build_sections(document)
        terms = profile_terms(profile)
        blocks_by_id = {block.block_id: block for block in document.blocks}
        chunks: list[ChildChunk] = []
        for section in sections:
            section_blocks = [blocks_by_id[block_id] for block_id in section.block_ids]
            chunks.extend(self._chunk_section(document, section, section_blocks, terms))
        return sections, chunks

    # ---------------------------------------------------------------- sections

    def _build_sections(self, document: ParsedDocument) -> list[ParentContext]:
        sections = []
        current_blocks: list[Block] = []
        current_size = 0
        current_key = None

        for block in sorted(document.blocks, key=lambda item: item.reading_order):
            key = (
                tuple(block.section_path),
                block.sheet_name,
                block.page if not block.section_path and not block.sheet_name else None,
            )
            size = len(self._context_text(block))
            keep_with_previous = bool(block.metadata.get("derived_from")) and current_blocks
            should_flush = current_blocks and not keep_with_previous and (
                key != current_key or current_size + size > self.config.section_max_chars
            )
            if should_flush:
                sections.append(self._make_section(document, current_blocks, len(sections)))
                current_blocks = []
                current_size = 0
            current_key = key
            current_blocks.append(block)
            current_size += size

        if current_blocks:
            sections.append(self._make_section(document, current_blocks, len(sections)))
        return sections

    def _make_section(self, document, blocks, ordinal):
        section_path = next((block.section_path for block in blocks if block.section_path), [])
        labels = [
            f"Document: {document.source_file}",
            f"Section: {' > '.join(section_path)}" if section_path else "",
        ]
        content_parts = [label for label in labels if label]
        content_parts.extend(f"[{block.block_type.upper()}]\n{self._context_text(block)}" for block in blocks)
        section_id = stable_id(document.document_id, "parent", ordinal, *(block.block_id for block in blocks))
        return ParentContext(
            parent_id=section_id,
            document_id=document.document_id,
            source_file=document.source_file,
            content="\n\n".join(content_parts),
            block_ids=[block.block_id for block in blocks],
            section_path=list(section_path),
            metadata={
                "kind": "section",
                "pages": sorted({block.page for block in blocks if block.page is not None}),
                "sheets": sorted({block.sheet_name for block in blocks if block.sheet_name}),
                "reading_order": min(block.reading_order for block in blocks),
            },
        )

    def _context_text(self, block: Block) -> str:
        """Text a block contributes to its section; large tables contribute only their preview."""
        if block.block_type == "table" and block.metadata.get("rows"):
            return self._render_table(block)[0]
        return block.content

    # ------------------------------------------------------------------ chunks

    def _chunk_section(self, document, section, blocks, terms) -> list[ChildChunk]:
        chunks: list[ChildChunk] = []
        text_run: list[Block] = []
        run_size = 0
        has_content = any(block.block_type != "heading" for block in blocks)

        def flush_text():
            nonlocal run_size
            if text_run:
                chunks.extend(self._chunk_text_group(document, section, list(text_run), len(chunks), terms))
                text_run.clear()
                run_size = 0

        for block in blocks:
            if block.block_type == "heading" and has_content:
                continue  # Headings are already part of every chunk's section prefix.
            if block.block_type == "caption" and block.metadata.get("attached_to"):
                continue  # Embedded together with the figure/table it describes.
            if block.block_type in TEXT_TYPES:
                # Merge only while the buffer is still a short fragment and the result fits one chunk.
                if text_run and (
                    run_size >= self.config.min_text_chars
                    or run_size + len(block.content) + 2 > self.config.max_text_chars
                ):
                    flush_text()
                text_run.append(block)
                run_size += len(block.content) + (2 if run_size else 0)
                continue
            flush_text()
            if block.block_type == "table" and block.metadata.get("rows"):
                chunks.append(self._chunk_table(document, section, block, terms, len(chunks)))
            elif block.block_type in VISUAL_TYPES:
                chunks.extend(self._chunk_visual(document, section, block, terms, len(chunks)))
            else:  # kpi, unstructured tables and anything new
                chunks.extend(self._chunk_single(document, section, block, block.content, terms, len(chunks)))
        flush_text()
        return chunks

    def _chunk_text_group(self, document, section, blocks: list[Block], start_index: int, terms) -> list[ChildChunk]:
        text = "\n\n".join(block.content for block in blocks)
        pieces = [text] if len(text) <= self.config.max_text_chars else split_text(text, self.config.max_text_chars, 0)
        return [
            self._emit(document, section, blocks, f"text-{start_index + index}", piece, "text", terms)
            for index, piece in enumerate(pieces)
            if piece.strip()
        ]

    def _chunk_table(self, document, section, block: Block, terms, start_index: int) -> ChildChunk:
        text, table_metadata = self._render_table(block)
        return self._emit(
            document,
            section,
            [block],
            f"table-{start_index}",
            text,
            "table",
            terms,
            metadata=table_metadata,
        )

    def _render_table(self, block: Block) -> tuple[str, dict[str, Any]]:
        """Markdown for a small table, or the preview text for a table that is stored in .xlsx."""
        header, body = table_header_and_body(block)
        width = max((len(row) for row in [*([header] if header else []), *body]), default=0)
        title = block.metadata.get("caption") or ""
        notes = block.metadata.get("references") or []
        inherited = bool(block.metadata.get("inherited_header"))
        lines = []
        if title:
            lines.append(f"Table title: {title}")
        if inherited:
            lines.append("(Phần tiếp theo của bảng ở trang trước)")

        if not is_large_table(block, self.config):
            lines.append(rows_to_markdown(([header] if header else []) + body))
            metadata = {"table_mode": "inline", "n_rows": len(body), "n_cols": width}
        else:
            cfg = self.config
            header_cells = list(header or []) + [""] * (width - len(header or []))
            rows_shown = min(cfg.table_preview_rows, len(body))
            columns_shown = min(cfg.table_preview_columns, width)
            xlsx_path = block.metadata.get("xlsx_path")
            location = f"toàn bộ bảng được lưu trong file {xlsx_path}" if xlsx_path else "toàn bộ bảng nằm trong dữ liệu có cấu trúc của tài liệu"
            lines.append(
                f"Bảng lớn: {len(body)} dòng dữ liệu x {width} cột. "
                f"Chunk này chỉ chứa phần xem trước ({rows_shown} dòng đầu, {columns_shown} cột đầu); {location}."
            )
            names = [name for name in header_cells if name]
            if names:
                lines.append("Tên các cột: " + _clip(" | ".join(names), cfg.table_key_values_chars, " | "))
            preview = [[_cell(row, c) for c in range(columns_shown)] for row in body[:rows_shown]]
            lines.append(rows_to_markdown(([header_cells[:columns_shown]] if header else []) + preview))
            key_values = list(dict.fromkeys(value for value in (_cell(row, 0) for row in body) if value))
            if key_values:
                label = f" ({header_cells[0]})" if header_cells and header_cells[0] else ""
                lines.append(f"Giá trị cột đầu tiên{label}: " + _clip("; ".join(key_values), cfg.table_key_values_chars, "; "))
            metadata = {
                "table_mode": "preview",
                "n_rows": len(body),
                "n_cols": width,
                "preview_rows": rows_shown,
                "preview_columns": columns_shown,
            }
            if xlsx_path:
                metadata["xlsx_path"] = xlsx_path
        if notes:
            lines.append("Referenced by: " + " | ".join(notes))
        return "\n".join(lines), metadata

    def _chunk_visual(self, document, section, block: Block, terms, start_index: int) -> list[ChildChunk]:
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
        return self._chunk_single(document, section, block, "\n".join(lines), terms, start_index, chunk_type)

    def _chunk_single(self, document, section, block, text, terms, start_index, chunk_type=None) -> list[ChildChunk]:
        """One chunk per element. Only text beyond ``max_element_chars`` is split (embedding limit)."""
        chunk_type = chunk_type or block.block_type
        limit = self.config.max_element_chars
        pieces = [text] if len(text) <= limit else split_text(text, limit, 0)
        return [
            self._emit(document, section, [block], f"{chunk_type}-{start_index + index}", piece, chunk_type, terms)
            for index, piece in enumerate(pieces)
            if piece.strip()
        ]

    # ------------------------------------------------------------------ output

    def _emit(self, document, section, blocks, position, body, chunk_type, terms, cell_range=None, metadata=None):
        entities = match_terms(terms["entities"], body, MAX_CHUNK_ENTITIES)
        keywords = match_terms(terms["keywords"], body, MAX_CHUNK_KEYWORDS)
        content = self._prefix(document, blocks[0], chunk_type, entities) + body
        return self._make_child(
            document, section, blocks, position, content, chunk_type, cell_range, metadata, keywords, entities
        )

    def _prefix(self, document: ParsedDocument, block: Block, content_type: str, entities: list[str] | None = None) -> str:
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
        if entities:
            lines.append("Entities: " + "; ".join(entities))
        return "\n".join(lines) + "\n\n"

    def _make_child(
        self, document, section, blocks, position, content, chunk_type,
        cell_range=None, metadata=None, keywords=None, entities=None,
    ):
        first = blocks[0]
        pages = [block.page for block in blocks if block.page is not None]
        return ChildChunk(
            chunk_id=stable_id(section.parent_id, *(block.block_id for block in blocks), position),
            parent_id=section.parent_id,
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
            keywords=list(keywords or []),
            entities=list(entities or []),
        )


def _clip(text: str, limit: int, separator: str) -> str:
    """Cut at ``limit`` without leaving half a value: drop the trailing partial item."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind(separator)
    return (cut[:boundary] if boundary > 0 else cut).rstrip() + separator.rstrip() + " ..."


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
        # Keep each bullet item (with its wrapped continuation lines) intact, and join lines
        # that were wrapped in the middle of a sentence (typical for PDF text blocks) so a
        # split never lands inside a sentence.
        items: list[str] = []
        previous_line = ""
        for line in lines:
            wrapped = (
                bool(items)
                and len(previous_line) >= WRAP_MIN_LINE_CHARS
                and not items[-1].endswith(SENTENCE_END)
            )
            if items and not BULLET_PATTERN.match(line) and (BULLET_PATTERN.match(items[-1]) is not None or wrapped):
                items[-1] = f"{items[-1]} {line}"
            else:
                items.append(line)
            previous_line = line
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
