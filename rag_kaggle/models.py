from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BLOCK_TYPES = ("heading", "text", "caption", "table", "image", "chart", "flowchart", "kpi", "ocr_page", "vlm")

RELATION_TYPES = (
    "belongs_to_section",
    "captioned_by",
    "referenced_by",
    "visualizes",
    "continues",
    "next_in_reading_order",
    "derived_from",
)


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
    reading_order: int = 0
    confidence: float | None = None
    raw_content: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Relationship:
    source_id: str
    target_id: str
    relation_type: str
    document_id: str
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
    relationships: list[Relationship] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    language: list[str] = field(default_factory=lambda: ["vi", "en"])
    parser_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blocks"] = [block.to_dict() for block in self.blocks]
        result["relationships"] = [relation.to_dict() for relation in self.relationships]
        return result

    def add_block(self, block: Block) -> Block:
        block.reading_order = len(self.blocks)
        self.blocks.append(block)
        return block


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
    token_count: int = 0
    parser_version: str = ""
    chunker_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def payload(self) -> dict[str, Any]:
        """Qdrant payload: searchable content and filter metadata only, no large tables."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "parent_id": self.parent_id,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "source_file": self.source_file,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "sheet_name": self.sheet_name,
            "cell_range": self.cell_range,
            "asset_path": self.asset_path,
            "original_file_name": self.metadata.get("original_file_name", self.source_file),
            "uploaded_at": self.metadata.get("uploaded_at"),
            "content_hash": self.metadata.get("content_hash"),
        }


@dataclass
class SearchHit:
    chunk: ChildChunk
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class StageStatus:
    stage: str
    status: str
    document_id: str | None = None
    source_file: str | None = None
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryPlan:
    original_query: str
    semantic_queries: list[str] = field(default_factory=list)
    keyword_query: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
