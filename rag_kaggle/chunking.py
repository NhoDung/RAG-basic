from __future__ import annotations

from .config import ChunkingConfig
from .models import Block, ChildChunk, ParentContext, ParsedDocument
from .parsers import rows_to_markdown, stable_id


class ParentChildChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, document: ParsedDocument) -> tuple[list[ParentContext], list[ChildChunk]]:
        parents = self._build_parents(document)
        blocks_by_id = {block.block_id: block for block in document.blocks}
        children = []
        for parent in parents:
            parent_blocks = [blocks_by_id[block_id] for block_id in parent.block_ids]
            for block in parent_blocks:
                children.extend(self._chunk_block(parent, block))
        return parents, children

    def _build_parents(self, document: ParsedDocument) -> list[ParentContext]:
        parents = []
        current_blocks: list[Block] = []
        current_size = 0
        current_key = None

        for block in document.blocks:
            key = (
                tuple(block.section_path),
                block.sheet_name,
                block.page if not block.section_path and not block.sheet_name else None,
            )
            should_flush = current_blocks and (
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
            },
        )

    def _chunk_block(self, parent: ParentContext, block: Block) -> list[ChildChunk]:
        if block.block_type == "table" and block.metadata.get("rows"):
            return self._chunk_table(parent, block)
        pieces = split_text(
            block.content,
            target_chars=self.config.child_chars,
            overlap_chars=self.config.child_overlap_chars,
        )
        chunks = []
        for index, piece in enumerate(pieces):
            content = self._prefix(block) + piece
            chunks.append(self._make_child(parent, block, index, content))
        return chunks

    def _chunk_table(self, parent: ParentContext, block: Block) -> list[ChildChunk]:
        rows = block.metadata["rows"]
        if not rows:
            return []
        header = rows[0]
        body = rows[1:]
        if not body:
            body = rows
        chunks = []
        for index, start in enumerate(range(0, len(body), self.config.table_rows_per_chunk)):
            group = body[start : start + self.config.table_rows_per_chunk]
            table_rows = [header] + group if body is not rows else group
            content = self._prefix(block) + rows_to_markdown(table_rows)
            chunks.append(self._make_child(parent, block, index, content))
        return chunks

    def _prefix(self, block: Block) -> str:
        lines = [f"Source: {block.source_file}"]
        if block.section_path:
            lines.append(f"Section: {' > '.join(block.section_path)}")
        if block.page is not None:
            lines.append(f"Page: {block.page}")
        if block.sheet_name:
            lines.append(f"Sheet: {block.sheet_name}")
        if block.cell_range:
            lines.append(f"Range: {block.cell_range}")
        lines.append(f"Content type: {block.block_type}")
        return "\n".join(lines) + "\n\n"

    def _make_child(self, parent, block, index, content):
        chunk_id = stable_id(parent.parent_id, block.block_id, index)
        return ChildChunk(
            chunk_id=chunk_id,
            parent_id=parent.parent_id,
            document_id=block.document_id,
            source_file=block.source_file,
            chunk_type=block.block_type,
            content=content,
            block_ids=[block.block_id],
            section_path=list(block.section_path),
            page_start=block.page,
            page_end=block.page,
            sheet_name=block.sheet_name,
            cell_range=block.cell_range,
            asset_path=block.asset_path,
        )


def split_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= target_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= target_chars:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + target_chars, len(paragraph))
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                break
            start = max(end - overlap_chars, start + 1)
        current = ""
    if current:
        chunks.append(current)

    if overlap_chars <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prefix = previous[-overlap_chars:]
        overlapped.append(f"{prefix}\n{current}")
    return overlapped
