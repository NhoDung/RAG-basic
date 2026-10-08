from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


PARSER_VERSION = "0.3.0"
CHUNKER_VERSION = "0.3.0"
ARTIFACT_SCHEMA_VERSION = 1


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
    # Large tables are never split into several chunks: the full table goes to an
    # .xlsx file and the chunk keeps only a preview (see ChunkingConfig).
    write_table_excel: bool = True
    libreoffice_binary: str | None = None


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
    # Set by the Kaggle device planner. None delegates placement to Transformers.
    device: str | None = None


@dataclass
class CorrectionConfig:
    """LLM spelling correction of OCR text (runs on the GPU the OCR worker does not use).

    Every call is stateless: system instruction + document glossary + the previous
    (already corrected) paragraph + the paragraph being fixed. Nothing accumulates
    in the model context between paragraphs.
    """

    enabled: bool = False
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None
    load_in_4bit: bool = True
    # Set by the Kaggle device planner. None delegates placement to Transformers.
    device: str | None = None
    max_new_tokens: int = 1024
    # Longer OCR blocks are corrected in segments of at most this many characters.
    segment_chars: int = 1500
    # Tail of the previous corrected text shown as read-only context.
    context_chars: int = 1200
    context_blocks: int = 1
    # Never look further back than this many pages (1 = the page before the current one).
    max_pages_back: int = 1
    min_chars: int = 12
    # Terms (people, places, signers, ...) collected while parsing one document and
    # fed back so the same entity is always spelled the same way.
    use_document_memory: bool = True
    memory_max_items_per_kind: int = 30
    memory_max_chars: int = 800
    memory_similarity: float = 0.9
    # A correction is rejected (the original text is kept) when it changes too much.
    min_similarity: float = 0.6
    max_length_change: float = 0.35
    max_consecutive_failures: int = 3


@dataclass
class ChunkingConfig:
    """Structure-aware chunking: one chunk per structural element.

    paragraph / sentence group -> 1 chunk, image -> 1 chunk, table -> 1 chunk (Markdown),
    or a preview chunk + a full .xlsx file when the table is too large to inline.
    """

    strategy: str = "structure_aware"
    # A paragraph longer than this is split on sentence boundaries (never mid-sentence).
    max_text_chars: int = 1800
    # Consecutive paragraphs of one section are merged while the buffer is shorter than this.
    min_text_chars: int = 120
    # Tables above either limit become "preview chunk + xlsx" instead of Markdown.
    table_inline_max_chars: int = 6000
    table_inline_max_rows: int = 60
    table_preview_rows: int = 5
    table_preview_columns: int = 5
    table_key_values_chars: int = 600
    # Image/chart/flowchart/KPI elements stay in one chunk up to this size (embedding-input safety cap).
    max_element_chars: int = 8000
    # Safety cap for one structural section (the unit used for context expansion).
    section_max_chars: int = 7500
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
    max_parent_chars_in_context: int = 6000
    # Large tables are answered from the rows that best match the question.
    max_table_rows_in_context: int = 30
    max_neighbor_chunks: int = 1


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
    vision: VisionConfig = field(default_factory=VisionConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
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
