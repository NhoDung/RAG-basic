from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    block_id: str
    document_id: str
    block_type: str
    content: str
    source_file: str
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    bbox: list[float] | None = None
    asset_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    document_id: str
    source_file: str
    file_type: str
    content_hash: str
    blocks: list[Block] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blocks"] = [block.to_dict() for block in self.blocks]
        return result


@dataclass
class ParentContext:
    parent_id: str
    document_id: str
    source_file: str
    content: str
    block_ids: list[str]
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    document_id: str
    source_file: str
    chunk_type: str
    content: str
    block_ids: list[str]
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    asset_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    chunk: ChildChunk
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

