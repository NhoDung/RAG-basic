from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsingConfig:
    render_dpi: int = 220
    scan_text_threshold: int = 80
    enable_ocr: bool = True
    ocr_model_name: str = "PaddleOCR-VL-1.6"
    ocr_model_dir: str | None = None
    ocr_batch_size: int = 1
    ocr_images_in_digital_docs: bool = True
    min_image_width: int = 220
    min_image_height: int = 120
    include_hidden_sheets: bool = False


@dataclass
class ChunkingConfig:
    child_chars: int = 1800
    child_overlap_chars: int = 240
    parent_chars: int = 7500
    table_rows_per_chunk: int = 15


@dataclass
class RetrievalConfig:
    dense_model: str = "BAAI/bge-m3"
    dense_dimension: int | None = None
    dense_device: str = "cuda"
    dense_batch_size: int = 8
    dense_top_k: int = 30
    sparse_top_k: int = 30
    fused_top_k: int = 24
    rerank_top_k: int = 7
    rrf_k: int = 60
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_use_fp16: bool = True


@dataclass
class GenerationConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 700
    max_context_chars: int = 28000
    temperature: float = 0.1
    query_rewrite_enabled: bool = False


@dataclass
class PipelineConfig:
    work_dir: Path = field(default_factory=lambda: Path("/kaggle/working/rag_runtime"))
    collection_name: str = "rag_chunks"
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @property
    def source_dir(self) -> Path:
        return self.work_dir / "source"

    @property
    def asset_dir(self) -> Path:
        return self.work_dir / "assets"

    @property
    def parsed_dir(self) -> Path:
        return self.work_dir / "parsed"

    @property
    def table_dir(self) -> Path:
        return self.work_dir / "tables"

    @property
    def qdrant_dir(self) -> Path:
        return self.work_dir / "qdrant"

    @property
    def metadata_db(self) -> Path:
        return self.work_dir / "metadata.db"

    @property
    def bm25_path(self) -> Path:
        return self.work_dir / "bm25.pkl"

    def create_directories(self) -> None:
        for path in (
            self.work_dir,
            self.source_dir,
            self.asset_dir,
            self.parsed_dir,
            self.table_dir,
            self.qdrant_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
