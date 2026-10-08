from __future__ import annotations

import re

from .config import CHUNKER_VERSION, ChunkingConfig
from .models import Block, Chunk, ParsedDocument
from .utils import estimate_tokens, rows_to_markdown, stable_id, table_xlsx_relpath


TEXT_TYPES = {"text", "ocr_page", "caption", "heading"}
VISUAL_TYPES = {"image", "chart", "flowchart"}
BULLET_PATTERN = re.compile(r"^\s*([-•*+▪◦●]|\d+[.)]|[a-zđ][.)])\s+", re.IGNORECASE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:…])\s+(?=[A-ZÀ-Ỹ0-9\"“(])")
MAX_LISTED_COLUMNS = 40


class StructureAwareChunker:
    """Chunk along the document structure instead of a fixed window.

    One chunk is exactly one of: a run of paragraphs from one section (split only at
    sentence boundaries when a single paragraph is too large), one whole table, or one
    image with its caption/OCR text. Tables and images are never cut. A table too large
    to embed is indexed as a preview and stored in full as an .xlsx file.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        blocks = sorted(document.blocks, key=lambda item: item.reading_order)
        sections_with_content = {
            self._section_key(block) for block in blocks if block.block_type != "heading"
        }
        chunks: list[Chunk] = []
        run: list[Block] = []
        run_key = None

        def flush() -> None:
            if run:
                chunks.extend(self._chunk_text_run(document, list(run)))
                run.clear()

        for block in blocks:
            if block.block_type == "heading" and self._section_key(block) in sections_with_content:
                continue  # The heading is already part of every chunk's section header.
            if block.block_type == "caption" and block.metadata.get("attached_to"):
                continue  # Embedded together with the figure/table it describes.
            if block.block_type in TEXT_TYPES:
                key = self._section_key(block)
                if run and key != run_key:
                    flush()
                run_key = key
                run.append(block)
                continue
            flush()
            if block.block_type == "table" and block.metadata.get("rows"):
                chunks.append(self._chunk_table(document, block))
            elif block.block_type in VISUAL_TYPES:
                chunks.extend(self._chunk_visual(document, block))
            else:  # kpi, unstructured tables and anything new
                chunks.extend(self._chunk_single(document, block, block.content))
        flush()
        for ordinal, chunk in enumerate(chunks):
            chunk.ordinal = ordinal
        return chunks

    @staticmethod
    def _section_key(block: Block) -> tuple:
        page = block.page if not block.section_path and not block.sheet_name else None
        return (tuple(block.section_path), block.sheet_name, page)

    # -------------------------------------------------------------------- text

    def _chunk_text_run(self, document: ParsedDocument, blocks: list[Block]) -> list[Chunk]:
        limit = self.config.text_chunk_chars
        groups: list[tuple[list[Block], str]] = []
        current: list[Block] = []
        current_text = ""
        for block in blocks:
            if len(block.content) > limit:
                if current:
                    groups.append((current, current_text))
                    current, current_text = [], ""
                for piece in split_text(block.content, limit, 0):
                    groups.append(([block], piece))
                continue
            candidate = f"{current_text}\n\n{block.content}".strip()
            if current and len(candidate) > limit:
                groups.append((current, current_text))
                current, candidate = [], block.content
            current.append(block)
            current_text = candidate
        if current:
            groups.append((current, current_text))

        return [
            self._make_chunk(document, group_blocks, f"text-{index}", self._header(group_blocks[0]) + text, "text")
            for index, (group_blocks, text) in enumerate(groups)
        ]

    # ------------------------------------------------------------------- tables

    def _chunk_table(self, document: ParsedDocument, block: Block) -> Chunk:
        rows = [list(row) for row in block.metadata["rows"]]
        inherited = block.metadata.get("inherited_header")
        table_rows = ([list(inherited)] + rows) if inherited else rows
        width = max(len(row) for row in table_rows)
        table_rows = [row + [""] * (width - len(row)) for row in table_rows]
        title = block.metadata.get("caption") or ""
        notes = block.metadata.get("references") or []

        def assemble(shown: list[list[str]], extra: list[str]) -> str:
            lines = []
            if title:
                lines.append(f"Table title: {title}")
            if inherited:
                lines.append("(Phần tiếp theo của bảng ở trang trước)")
            lines.extend(extra)
            lines.append(rows_to_markdown(shown))
            if notes:
                lines.append("Referenced by: " + " | ".join(notes))
            return self._header(block) + "\n".join(lines)

        content = assemble(table_rows, [])
        metadata = {"table_rows": len(table_rows) - 1, "table_columns": width}
        if len(content) > self.config.table_inline_max_chars:
            preview_columns = max(1, self.config.table_preview_columns)
            preview = [row[:preview_columns] for row in table_rows[: 1 + max(1, self.config.table_preview_rows)]]
            path = table_xlsx_relpath(document.document_id, block.block_id)
            columns = [cell for cell in table_rows[0] if cell][:MAX_LISTED_COLUMNS]
            extra = [
                f"[Bảng lớn: {len(table_rows) - 1} hàng × {width} cột; dưới đây là "
                f"{len(preview) - 1} hàng và {min(preview_columns, width)} cột đầu. "
                f"Bảng đầy đủ nằm trong file {path}]",
            ]
            if columns:
                extra.append("Các cột: " + "; ".join(columns))
            content = assemble(preview, extra)
            metadata.update({"table_truncated": True, "table_file": path})
        return self._make_chunk(
            document, [block], "table-0", content, "table", cell_range=block.cell_range, metadata=metadata
        )

    # ------------------------------------------------------------------ visuals

    def _chunk_visual(self, document: ParsedDocument, block: Block) -> list[Chunk]:
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
        return self._chunk_single(document, block, "\n".join(lines), chunk_type, self.config.visual_max_chars)

    def _chunk_single(self, document, block, text, chunk_type=None, limit=None) -> list[Chunk]:
        chunk_type = chunk_type or block.block_type
        pieces = split_text(text, limit or self.config.text_chunk_chars, 0) or [text]
        return [
            self._make_chunk(document, [block], f"{chunk_type}-{index}", self._header(block) + piece, chunk_type)
            for index, piece in enumerate(pieces)
            if piece.strip()
        ]

    # ------------------------------------------------------------------ helpers

    def _header(self, block: Block) -> str:
        """Short semantic header embedded with the chunk (file and page live in metadata)."""
        lines = []
        if block.section_path:
            lines.append(f"Section: {' > '.join(block.section_path)}")
        if block.sheet_name:
            lines.append(f"Sheet: {block.sheet_name}")
        if block.cell_range:
            lines.append(f"Range: {block.cell_range}")
        return "\n".join(lines) + "\n\n" if lines else ""

    def _make_chunk(self, document, blocks, position, content, chunk_type, cell_range=None, metadata=None) -> Chunk:
        first = blocks[0]
        pages = [block.page for block in blocks if block.page is not None]
        return Chunk(
            chunk_id=stable_id(document.document_id, "chunk", *(block.block_id for block in blocks), position),
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
