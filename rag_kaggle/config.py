from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


PARSER_VERSION = "0.4.0"
CHUNKER_VERSION = "1.0.0-structure-aware"
ARTIFACT_SCHEMA_VERSION = 2


@dataclass
class IngestionConfig:
    max_file_mb: int = 200
    max_archive_mb: int = 4096
    max_archive_files: int = 5000
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".xlsx", ".xlsm", ".xltx", ".xltm", ".xls")
    skip_unchanged_documents: bool = True


@dataclass
class ParsingConfig:
    render_dpi: int = 220
    scan_text_threshold: int = 80
    enable_ocr: bool = True
    ocr_model_name: str = "PaddleOCR-VL-1.6"
    ocr_model_dir: str | None = None
    # Physical GPU exposed to the isolated PaddleOCR worker. For example, "0".
    # None leaves CUDA_VISIBLE_DEVICES unchanged.
    ocr_cuda_visible_devices: str | None = None
    # Python of a separate venv with paddlepaddle-gpu + paddleocr (recommended on
    # Kaggle). None runs PaddleOCR-VL inside the current process.
    ocr_python: str | None = None
    ocr_batch_size: int = 1
    ocr_images_in_digital_docs: bool = True
    min_image_width: int = 220
    min_image_height: int = 120
    include_hidden_sheets: bool = False
    include_hidden_rows_columns: bool = False
    heading_font_ratio: float = 1.15
    heading_max_chars: int = 160
    write_table_parquet: bool = True
    libreoffice_binary: str | None = None


@dataclass
class OCRCorrectionConfig:
    """Settings for the ordered OCR spelling-correction stage."""

    enabled: bool = False
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    revision: str | None = None
    device: str | None = None
    load_in_4bit: bool = True
    temperature: float = 0.0
    max_new_tokens: int = 1024
    max_input_tokens: int = 6000
    max_buffered_components: int = 8
    max_retries: int = 1
    fail_open: bool = True
    correct_labels: tuple[str, ...] = ("doc_title", "paragraph_title", "text", "caption")
    correct_tables: bool = False
    # Context budget: the prompt carries system + skill + (glossary) + the tail of the
    # previous corrected text + one segment of the current component, nothing else.
    previous_context_chars: int = 600
    segment_chars: int = 1200
    # Rolling per-document list of canonical names. Off by default: enable only after
    # benchmarking that the correction model follows it.
    glossary_enabled: bool = False
    glossary_max_terms: int = 20


@dataclass
class VisionConfig:
    enabled: bool = False
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    revision: str | None = None
    load_in_4bit: bool = True
    max_new_tokens: int = 900
    max_retries: int = 2
    min_ocr_chars_for_skip: int = 0
    image_types: tuple[str, ...] = ("flowchart", "chart", "diagram")
    # Tail of the one preceding page given to the VLM as continuation context.
    previous_page_chars: int = 1500
    # Set by the Kaggle device planner. None delegates placement to Transformers.
    device: str | None = None


@dataclass
class ChunkingConfig:
    """Structure-aware chunking: a chunk is one text unit, one table or one image."""

    # Consecutive paragraphs of one section are merged up to this size; a single
    # paragraph is only split (at sentence boundaries) when it alone exceeds it.
    text_chunk_chars: int = 1800
    # Tables stay whole as markdown up to this size. Larger tables are indexed as a
    # preview (caption, header, first rows/columns) and stored in full as an .xlsx file.
    table_inline_max_chars: int = 6000
    table_preview_rows: int = 5
    table_preview_columns: int = 5
    # An image chunk (OCR text + caption + references) is split only above this size.
    visual_max_chars: int = 6000
    chars_per_token: float = 3.6


@dataclass
class RetrievalConfig:
    dense_model: str | None = "BAAI/bge-m3"
    dense_revision: str | None = None
    dense_fallback_model: str = "BAAI/bge-m3"
    dense_query_instruction: str | None = None
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
    reranker_batch_size: int = 16
    # FlagEmbedding accepts an explicit device in current releases. Keeping this
    # separate from dense_device lets a dual-T4 notebook run the low-volume
    # reranking stage on CPU while GPU 1 handles OCR and dense embedding.
    reranker_device: str = "cuda"
    expand_relationships: bool = True
    max_related_blocks: int = 6
    # Adjacent text chunks of the same section added around a retrieved text chunk.
    neighbor_chunks: int = 1
    # Budget for expanding a preview-only table to its full rows in the answer context.
    max_table_chars_in_context: int = 12000
    # Documents that mention an entity found in the query get an extra RRF ranking.
    entity_boost: bool = True


@dataclass
class GenerationConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None
    load_in_4bit: bool = True
    max_new_tokens: int = 700
    max_context_chars: int = 28000
    temperature: float = 0.1
    query_rewrite_enabled: bool = False
    query_rewrite_count: int = 2
    structured_computation_enabled: bool = True
    # Set to e.g. "cuda:0" for a single-device Qwen placement. None uses auto placement.
    device: str | None = None


@dataclass
class GuardrailConfig:
    enabled: bool = True
    min_rerank_score: float = 0.05
    min_rrf_score: float = 0.0
    detect_prompt_injection: bool = True
    mask_pii_in_logs: bool = True
    validate_numbers: bool = True


@dataclass
class RuntimeConfig:
    unload_models_between_stages: bool = True
    keep_answer_model_loaded: bool = True
    log_traces: bool = True
    # Query traces live outside the frozen corpus so chat never mutates it.
    session_dir: str | None = None


@dataclass
class ArtifactConfig:
    include_source_documents: bool = False
    include_parsed_documents: bool = True


@dataclass
class PipelineConfig:
    work_dir: Path = field(default_factory=lambda: Path("/kaggle/working/rag_runtime"))
    collection_name: str = "rag_chunks"
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    ocr_correction: OCRCorrectionConfig = field(default_factory=OCRCorrectionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

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
    def manifest_dir(self) -> Path:
        return self.work_dir / "manifests"

    @property
    def log_dir(self) -> Path:
        return self.work_dir / "logs"

    @property
    def metadata_db(self) -> Path:
        return self.work_dir / "metadata.db"

    @property
    def session_dir(self) -> Path:
        return Path(self.runtime.session_dir) if self.runtime.session_dir else self.work_dir.parent / "rag_session"

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
            self.manifest_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        config = cls()
        _apply(config, data)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        import yaml

        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _to_plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _apply(target: Any, data: dict[str, Any]) -> None:
    known = {item.name: item for item in fields(target)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif isinstance(current, Path) or key == "work_dir":
            setattr(target, key, Path(value))
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(target, key, tuple(value))
        else:
            setattr(target, key, value)
