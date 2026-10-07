from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

from .chunking import ParentChildChunker
from .computation import compute_from_blocks
from .config import ARTIFACT_SCHEMA_VERSION, CHUNKER_VERSION, PARSER_VERSION, PipelineConfig
from .generation import LocalQwen
from .guardrails import (
    REFUSAL_TEXT,
    detect_prompt_injection,
    retrieval_confidence,
    unsupported_numbers,
)
from .ingestion import IngestionError, discover_input_files, validate_file
from .models import StageStatus
from .paddleocr_vl import PaddleOCRVLAdapter
from .parsers import DocumentParser, save_parsed_document, save_table_parquet
from .relationships import build_relationships
from .retrieval import HybridRetriever, Reranker
from .storage import DenseEncoder, HybridIndex, MetadataStore
from .tracing import Trace
from .utils import file_sha256
from .vision import VisionReasoner


LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[str], None]


class IngestionInProgressError(RuntimeError):
    """Raised instead of queueing a second expensive ingestion run."""


class PipelineModeError(RuntimeError):
    """Raised when a stage is unavailable in the selected runtime mode."""


class RAGPipeline:
    VALID_MODES = {"full", "ingestion", "retrieval"}

    def __init__(self, config: PipelineConfig | None = None, mode: str = "full"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown pipeline mode: {mode}")
        self.config = config or PipelineConfig()
        self.mode = mode
        self.config.create_directories()
        if mode in ("full", "ingestion"):
            self.ocr = PaddleOCRVLAdapter(
                model_name=self.config.parsing.ocr_model_name,
                model_dir=self.config.parsing.ocr_model_dir,
                python_executable=self.config.parsing.ocr_python,
                log_path=self.config.log_dir / "ocr_worker.log",
                cuda_visible_devices=self.config.parsing.ocr_cuda_visible_devices,
            )
            self.vision = VisionReasoner(self.config)
            self.parser = DocumentParser(self.config, self.ocr, self.vision)
            self.chunker = ParentChildChunker(self.config.chunking)
        else:
            self.ocr = self.vision = self.parser = self.chunker = None
        if mode in ("full", "retrieval"):
            self.reranker = Reranker(self.config)
            self.generator = LocalQwen(self.config)
        else:
            self.reranker = self.generator = None
        # SQLite/Qdrant are mutable shared state; ingestion must be exclusive.
        self._ingestion_lock = threading.Lock()
        self._open_stores()

    def _open_stores(self) -> None:
        self.metadata = MetadataStore(self.config.metadata_db)
        self.encoder = DenseEncoder(self.config, self.metadata)
        self.index = HybridIndex(self.config, self.metadata, self.encoder)
        self.retriever = (
            HybridRetriever(self.config, self.metadata, self.index, self.reranker)
            if self.reranker is not None
            else None
        )

    # --------------------------------------------------------------- ingestion

    def ingest(
        self,
        files: list[str | Path],
        reset: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict:
        if self.mode not in ("full", "ingestion"):
            raise PipelineModeError("Ingestion is disabled in retrieval-only mode.")
        if self.metadata.get_setting("corpus.frozen", False):
            raise PipelineModeError(
                "This corpus is frozen. Create a new ingestion work_dir to build another corpus."
            )
        if not self._ingestion_lock.acquire(blocking=False):
            raise IngestionInProgressError(
                "An ingestion run is already in progress. Wait for it to finish before uploading another batch."
            )
        try:
            return self._ingest(files, reset=reset, progress=progress)
        finally:
            self._ingestion_lock.release()

    def ingest_sources(
        self,
        sources: list[str | Path],
        reset: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """Ingest supported documents discovered under files, directories or ZIP archives."""
        files, discovery = discover_input_files(sources, self.config)
        if not files:
            skipped = ", ".join(
                f"{item['path']} ({item['reason']})" for item in discovery["skipped"][:20]
            ) or "none"
            raise ValueError(
                "No supported documents were found. "
                f"Configured sources: {discovery['sources']}. Skipped/not found: {skipped}. "
                f"Supported extensions: {', '.join(self.config.ingestion.allowed_extensions)}"
            )
        report = self.ingest(files, reset=reset, progress=progress)
        report["input_discovery"] = discovery
        return report

    def _ingest(
        self,
        files: list[str | Path],
        reset: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """Ingestion mode (§12): parse -> OCR/VLM -> chunk -> embed -> index -> manifest.

        ``reset=False`` adds to the existing index. Identical re-uploads are
        skipped; different contents with the same name are kept as versions.
        """
        paths = [Path(path) for path in files]
        if not paths:
            raise ValueError("No input files")
        run_id = uuid.uuid4().hex[:12]
        notify = progress or (lambda message: None)
        report: dict[str, Any] = {"run_id": run_id, "documents": [], "skipped": [], "errors": [], "statuses": []}
        notify(f"Starting ingestion: {len(paths)} file(s).")

        def record(stage, status, source=None, document_id=None, error_code=None, message=None, retryable=False, started=None):
            item = StageStatus(
                stage=stage,
                status=status,
                document_id=document_id,
                source_file=source,
                error_code=error_code,
                message=message,
                retryable=retryable,
                duration_ms=round((time.perf_counter() - started) * 1000, 1) if started else None,
            )
            self.metadata.record_status(item, run_id)
            report["statuses"].append(item.to_dict())
            notify(f"[{stage}] {status}: {source or ''} {message or ''}".strip())

        if reset:
            self.metadata.reset()
            notify("Đã xoá index cũ.")
        if self.config.runtime.unload_models_between_stages:
            # Free GPU for OCR/VLM/embedding; chat models reload lazily.
            if self.generator is not None:
                self.generator.unload()
            if self.reranker is not None:
                self.reranker.unload()

        new_chunks = []
        for file_index, source in enumerate(paths, start=1):
            name = source.name
            started = time.perf_counter()
            try:
                facts = validate_file(source, self.config)
                content_hash = file_sha256(source)
                existing = self.metadata.document_by_original_and_hash(name, content_hash)
                if (
                    not reset
                    and self.config.ingestion.skip_unchanged_documents
                    and existing is not None
                ):
                    report["skipped"].append({"file": name, "document_id": existing["document_id"], "reason": "unchanged"})
                    record("parse", "skipped", name, existing["document_id"], message="unchanged file already indexed")
                    continue
                # Existing versions with the same original filename are retained.
                uploaded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                stored_source = self._copy_source(source, content_hash)
                notify(f"[{file_index}/{len(paths)}] Parsing {name}...")
                document = self.parser.parse(stored_source, display_name=name, progress=notify)
                document.metadata.update(
                    {
                        "validation": facts,
                        "original_file_name": name,
                        "uploaded_at": uploaded_at,
                        "stored_source": stored_source.relative_to(self.config.work_dir).as_posix(),
                    }
                )
                warnings = document.metadata.get("warnings", [])
                record(
                    "parse",
                    "partial" if warnings else "success",
                    name,
                    document.document_id,
                    message=f"{len(document.blocks)} blocks; {len(warnings)} warnings" if warnings else f"{len(document.blocks)} blocks",
                    retryable=bool(warnings),
                    started=started,
                )
                for warning in warnings:
                    if warning.get("stage") in ("ocr", "vlm"):
                        record(warning["stage"], "partial", name, document.document_id, message=warning["message"], retryable=True)

                stage_started = time.perf_counter()
                build_relationships(document)
                parents, chunks = self.chunker.chunk(document)
                version_metadata = {
                    "original_file_name": name,
                    "uploaded_at": uploaded_at,
                    "content_hash": content_hash,
                }
                for parent in parents:
                    parent.metadata.update(version_metadata)
                for chunk in chunks:
                    chunk.metadata.update(version_metadata)
                if self.config.parsing.write_table_parquet:
                    save_table_parquet(document, self.config.table_dir)
                save_parsed_document(document, self.config.parsed_dir)
                self.metadata.upsert_document(document, parents, chunks)
                record("chunk", "success" if chunks else "failed", name, document.document_id,
                       error_code=None if chunks else "no_chunks",
                       message=f"{len(parents)} parents, {len(chunks)} chunks, {len(document.relationships)} relationships",
                       started=stage_started)
                new_chunks.extend(chunks)
                report["documents"].append(
                    {
                        "file": name,
                        "document_id": document.document_id,
                        "blocks": len(document.blocks),
                        "relationships": len(document.relationships),
                        "parents": len(parents),
                        "chunks": len(chunks),
                        "warnings": warnings,
                        "has_macros": facts.get("has_macros", False),
                    }
                )
            except IngestionError as exc:
                report["errors"].append({"file": name, "error_code": exc.error_code, "error": str(exc)})
                record("parse", "failed", name, error_code=exc.error_code, message=str(exc), retryable=exc.retryable, started=started)
            except Exception as exc:
                LOGGER.exception("Failed to ingest %s", source)
                report["errors"].append({"file": name, "error_code": "parse_error", "error": str(exc)})
                record("parse", "failed", name, error_code="parse_error", message=str(exc), retryable=True, started=started)

        self.ocr.unload()
        self.vision.unload()

        if new_chunks or reset:
            started = time.perf_counter()
            notify(f"Embedding + indexing {len(new_chunks)} chunks...")
            try:
                self.index.build(new_chunks, reset=reset, progress=notify)
                record("index", "success", message=f"{len(new_chunks)} chunks upserted", started=started)
            except Exception as exc:
                LOGGER.exception("Indexing failed")
                # Keep metadata consistent with the vector index: drop documents that were not indexed.
                for document in report["documents"]:
                    self.metadata.delete_document(document["document_id"])
                report["errors"].append({"file": None, "error_code": "index_error", "error": str(exc)})
                record("index", "failed", error_code="index_error", message=str(exc), retryable=True, started=started)
                report["documents"] = []
            if self.config.runtime.unload_models_between_stages:
                self.encoder.unload()

        report["stats"] = self.metadata.stats()
        report["ok"] = not report["errors"]
        self._write_manifest(report)
        notify("Hoàn tất ingestion.")
        return report

    def _write_manifest(self, report: dict) -> Path:
        manifest = {
            **{key: value for key, value in report.items() if key != "statuses"},
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "collection_name": self.config.collection_name,
            "dense_model": self.metadata.get_setting("index.dense_model", self.config.retrieval.dense_model),
            "dense_dimension": self.metadata.get_setting("index.dimension"),
            "ocr_model": self.config.parsing.ocr_model_name,
            "vlm_model": self.config.vision.model if self.config.vision.enabled else None,
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "chunking": self.config.to_dict()["chunking"],
            "status_counts": _count_statuses(report["statuses"]),
        }
        text = json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
        path = self.config.manifest_dir / f"manifest_{report['run_id']}.json"
        path.write_text(text, encoding="utf-8")
        (self.config.work_dir / "manifest.json").write_text(text, encoding="utf-8")
        (self.config.work_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    # -------------------------------------------------------------------- chat

    def ask(self, query: str, filters: dict[str, Any] | None = None) -> dict:
        if self.mode not in ("full", "retrieval"):
            raise PipelineModeError("Question answering is disabled in ingestion-only mode.")
        self.validate_corpus(require_frozen=self.mode == "retrieval")
        query = (query or "").strip()
        if not query:
            return {"answer": "Vui lòng nhập câu hỏi.", "citations": [], "trace": [], "warnings": []}

        guard = self.config.guardrails
        trace = Trace(
            "query",
            self.config.session_dir if self.config.runtime.log_traces else None,
            guard.mask_pii_in_logs,
        )
        trace.set("query", query)
        warnings: list[str] = []
        if guard.enabled and guard.detect_prompt_injection and detect_prompt_injection(query):
            warnings.append("Câu hỏi chứa mẫu giống prompt injection; hệ thống chỉ trả lời dựa trên tài liệu.")

        with trace.stage("query_rewrite"):
            plan = self.generator.plan_query(
                query,
                self.metadata.list_values("source_file"),
                self.metadata.list_values("sheet_name"),
            )
        applied_filters = {**plan.filters, **{k: v for k, v in (filters or {}).items() if v}}
        trace.set("query_plan", plan.to_dict())
        trace.set("filters", applied_filters)

        with trace.stage("retrieve"):
            hits, stage_trace = self.retriever.retrieve_with_trace(
                query, plan.semantic_queries, applied_filters, plan.keyword_query
            )
        trace.set("retrieval", stage_trace)
        hit_rows = [_hit_row(hit) for hit in hits]
        trace.set("hits", hit_rows)

        confident, reason = retrieval_confidence(hits, guard, self.config.retrieval.reranker_enabled)
        if not confident:
            trace.set("refusal_reason", reason)
            data = trace.finish()
            return {
                "answer": REFUSAL_TEXT,
                "citations": [],
                "source_ids": [],
                "refused": True,
                "refusal_reason": reason,
                "warnings": warnings,
                "query_variants": plan.semantic_queries,
                "query_plan": plan.to_dict(),
                "trace": hit_rows,
                "contexts": [],
                "trace_id": trace.trace_id,
                "latency_ms": data["latency_ms"],
            }

        with trace.stage("expand_context"):
            contexts = self.retriever.expand_context(hits)
        for context in contexts:
            if guard.enabled and guard.detect_prompt_injection and detect_prompt_injection(context["content"]):
                warnings.append(f"{context['citation']} chứa đoạn văn giống chỉ dẫn hệ thống; đã được xử lý như dữ liệu.")

        computed = []
        if self.config.generation.structured_computation_enabled:
            with trace.stage("structured_computation"):
                computed = self._compute(query, hits, len(contexts))
        trace.set("computation", [item["result"] for item in computed])

        with trace.stage("generate"):
            result = self.generator.answer(query, contexts, computed)
        trace.set("usage", self.generator.last_usage)

        if not result["refused"] and not result["citation_valid"]:
            warnings.append("Câu trả lời không có SOURCE_ID hợp lệ; hãy kiểm tra lại với các nguồn đã truy xuất.")
        if guard.enabled and guard.validate_numbers and not result["refused"]:
            missing = unsupported_numbers(result["answer"], result.get("prompt_context", ""))
            if missing:
                warnings.append("Các số không tìm thấy trong ngữ cảnh: " + ", ".join(missing))
                result["unsupported_numbers"] = missing

        result.pop("prompt_context", None)
        result.update(
            {
                "warnings": warnings,
                "query_variants": plan.semantic_queries,
                "query_plan": plan.to_dict(),
                "filters": applied_filters,
                "trace": hit_rows,
                "contexts": [
                    {
                        "source_id": item["source_id"],
                        "citation": item["citation"],
                        "block_ids": item["block_ids"],
                        "related_block_ids": item["related_block_ids"],
                        "asset_paths": item["asset_paths"],
                        "preview": item["content"][:800],
                    }
                    for item in [*computed, *contexts]
                ],
                "computation": [item["result"] for item in computed],
                "trace_id": trace.trace_id,
            }
        )
        trace.set("answer", result["answer"])
        trace.set("citations", result["citations"])
        trace.set("warnings", warnings)
        trace.set("context_ids", [item["source_id"] for item in result["contexts"]])
        result["latency_ms"] = trace.finish()["latency_ms"]
        return result

    def _compute(self, query: str, hits, offset: int) -> list[dict]:
        block_ids = []
        for hit in hits:
            if hit.chunk.chunk_type in ("table", "kpi"):
                block_ids.extend(hit.chunk.block_ids)
        blocks = [block for block in self.metadata.get_blocks(list(dict.fromkeys(block_ids))) if block.metadata.get("rows")]
        result = compute_from_blocks(query, blocks)
        if result is None:
            return []
        if result.sheet_name:
            citation = f'[{result.source_file}, sheet "{result.sheet_name}", vùng {result.cell_range}]'
        elif result.page is not None:
            citation = f"[{result.source_file}, trang {result.page}]"
        else:
            citation = f"[{result.source_file}]"
        return [
            {
                "source_id": f"SOURCE_{offset + 1}",
                "parent": None,
                "content": result.render(),
                "citation": citation,
                "block_ids": [result.block_id],
                "related_block_ids": [],
                "asset_paths": [],
                "result": result.to_dict() | {"rows_used": result.rows_used[:20]},
            }
        ]

    # --------------------------------------------------------------- utilities

    def evaluate(self, dataset_path: str | Path, run_answers: bool = True, ks: tuple[int, ...] = (5, 10)) -> dict:
        if self.mode not in ("full", "retrieval"):
            raise PipelineModeError("Evaluation is disabled in ingestion-only mode.")
        from .evaluation import run_evaluation

        return run_evaluation(self, dataset_path, run_answers=run_answers, ks=ks)

    def freeze_corpus(self) -> dict[str, Any]:
        """Seal the current index and write the compatibility contract used by chat sessions."""
        if self.mode not in ("full", "ingestion"):
            raise PipelineModeError("Only an ingestion runtime can freeze a corpus.")
        stats = self.metadata.stats()
        if not stats.get("chunks"):
            raise RuntimeError("Cannot freeze an empty corpus")
        corpus_id = self.metadata.get_setting("corpus.id") or uuid.uuid4().hex
        self.metadata.set_setting("corpus.id", corpus_id)
        self.metadata.set_setting("corpus.frozen", True)
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "corpus_id": corpus_id,
            "frozen": True,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "collection_name": self.config.collection_name,
            "stats": stats,
            "embedding": {
                "model": self.metadata.get_setting("index.dense_model"),
                "revision": self.config.retrieval.dense_revision,
                "dimension": self.metadata.get_setting("index.dimension"),
                "normalize": True,
                "query_instruction": self.config.retrieval.dense_query_instruction,
            },
            "models": {
                "ocr": self.config.parsing.ocr_model_name,
                "vision": self.config.vision.model if self.config.vision.enabled else None,
            },
            "versions": {"parser": PARSER_VERSION, "chunker": CHUNKER_VERSION},
            "chunking": self.config.to_dict()["chunking"],
        }
        path = self.config.work_dir / "corpus_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.metadata.connection.commit()
        return manifest

    def validate_corpus(self, require_frozen: bool = True) -> dict[str, Any]:
        manifest_path = self.config.work_dir / "corpus_manifest.json"
        if not manifest_path.exists():
            if require_frozen:
                raise RuntimeError("No frozen corpus is loaded. Restore a corpus bundle first.")
            return {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported corpus schema {manifest.get('artifact_schema_version')}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}."
            )
        if require_frozen and not manifest.get("frozen"):
            raise RuntimeError("The loaded corpus is not frozen.")
        if require_frozen and not self.metadata.get_setting("corpus.frozen", False):
            raise RuntimeError("Metadata database does not mark this corpus as frozen.")
        embedding = manifest.get("embedding") or {}
        indexed_model = embedding.get("model")
        if indexed_model != self.metadata.get_setting("index.dense_model"):
            raise RuntimeError("Corpus manifest and metadata database disagree on the embedding model.")
        if embedding.get("dimension") != self.metadata.get_setting("index.dimension"):
            raise RuntimeError("Corpus manifest and metadata database disagree on vector dimension.")
        configured_model = self.config.retrieval.dense_model
        if configured_model is None:
            self.config.retrieval.dense_model = indexed_model
            self.config.retrieval.dense_revision = embedding.get("revision")
            self.config.retrieval.dense_query_instruction = embedding.get("query_instruction")
        elif indexed_model and configured_model != indexed_model:
            raise RuntimeError(
                f"Corpus uses embedding model {indexed_model}, but retrieval is configured with "
                f"{configured_model}. Set dense_model=None to inherit it from the corpus."
            )
        else:
            _adopt_or_validate_embedding_options(self.config, embedding)
        if self.config.collection_name != manifest.get("collection_name"):
            raise RuntimeError(
                f"Corpus collection is {manifest.get('collection_name')}, but config uses "
                f"{self.config.collection_name}."
            )
        if not (self.config.metadata_db.exists() and self.config.qdrant_dir.exists()):
            raise RuntimeError("Corpus is missing metadata.db or Qdrant storage")
        expected_stats = manifest.get("stats") or {}
        current_stats = self.metadata.stats()
        if any(current_stats.get(key) != value for key, value in expected_stats.items()):
            raise RuntimeError("Corpus manifest and metadata database have different record counts.")
        return manifest

    def export_corpus_bundle(
        self,
        destination: str | Path | None = None,
        include_source_documents: bool | None = None,
    ) -> Path:
        destination = Path(destination or self.config.work_dir.parent / "corpus_bundle.zip")
        if destination.resolve().is_relative_to(self.config.work_dir.resolve()):
            raise ValueError("Corpus bundle must be written outside work_dir")
        include_source = (
            self.config.artifacts.include_source_documents
            if include_source_documents is None
            else include_source_documents
        )
        manifest = self.freeze_corpus()
        manifest["bundle"] = {
            "includes_source_documents": include_source,
            "includes_parsed_documents": self.config.artifacts.include_parsed_documents,
        }
        (self.config.work_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.index.close()  # Release the Qdrant local lock while zipping.
        self.metadata.connection.commit()
        included_roots = {"qdrant", "metadata.db", "bm25.pkl", "manifests", "config.json", "manifest.json", "corpus_manifest.json", "tables", "assets"}
        if self.config.artifacts.include_parsed_documents:
            included_roots.add("parsed")
        if include_source:
            included_roots.add("source")
        files = [
            path for path in self.config.work_dir.rglob("*")
            if path.is_file() and path.relative_to(self.config.work_dir).parts[0] in included_roots
        ]
        checksums = {
            path.relative_to(self.config.work_dir).as_posix(): _sha256(path)
            for path in sorted(files)
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(files):
                bundle.write(path, path.relative_to(self.config.work_dir).as_posix())
            bundle.writestr("checksums.json", json.dumps(checksums, ensure_ascii=False, indent=2))
        return destination

    def export_artifacts(self, destination: str | Path | None = None) -> Path:
        """Backward-compatible alias for the frozen corpus bundle exporter."""
        return self.export_corpus_bundle(destination)

    def restore_corpus_bundle(self, archive: str | Path) -> dict:
        """Validate and atomically activate a frozen corpus bundle."""
        archive = Path(archive)
        work_dir = self.config.work_dir.resolve()
        work_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix="rag_restore_", dir=work_dir.parent))
        extracted = temporary_root / "corpus"
        extracted.mkdir()
        try:
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.namelist():
                    target = (extracted / member).resolve()
                    if not target.is_relative_to(extracted.resolve()):
                        raise ValueError(f"Unsafe path in archive: {member}")
                bundle.extractall(extracted)
            _validate_extracted_bundle(extracted)
            manifest = json.loads((extracted / "corpus_manifest.json").read_text(encoding="utf-8"))
            embedding = manifest.get("embedding") or {}
            configured_model = self.config.retrieval.dense_model
            if configured_model is not None and configured_model != embedding.get("model"):
                raise RuntimeError(
                    f"Corpus uses embedding model {embedding.get('model')}, but retrieval is configured "
                    f"with {configured_model}. Set dense_model=None to inherit it."
                )
            if configured_model is None:
                self.config.retrieval.dense_model = embedding.get("model")
                self.config.retrieval.dense_revision = embedding.get("revision")
                self.config.retrieval.dense_query_instruction = embedding.get("query_instruction")
            else:
                _adopt_or_validate_embedding_options(self.config, embedding)
            self.config.collection_name = manifest.get("collection_name", self.config.collection_name)

            self.close()
            backup = work_dir.parent / f".{work_dir.name}.backup-{uuid.uuid4().hex[:8]}"
            try:
                if work_dir.exists():
                    os.replace(work_dir, backup)
                os.replace(extracted, work_dir)
                self.config.create_directories()
                self._open_stores()
                self.validate_corpus(require_frozen=True)
            except Exception:
                try:
                    self.index.close()
                    self.metadata.close()
                except Exception:
                    pass
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                if backup.exists():
                    os.replace(backup, work_dir)
                self.config.create_directories()
                self._open_stores()
                raise
            else:
                if backup.exists():
                    shutil.rmtree(backup)
            return {"manifest": manifest, "stats": self.metadata.stats()}
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def restore_artifacts(self, archive: str | Path) -> dict:
        """Backward-compatible alias for restoring a frozen corpus bundle."""
        result = self.restore_corpus_bundle(archive)
        return result["stats"]

    @classmethod
    def from_artifacts(cls, archive: str | Path, config: PipelineConfig | None = None) -> "RAGPipeline":
        pipeline = cls(config)
        pipeline.restore_corpus_bundle(archive)
        return pipeline

    def _copy_source(self, source: Path, content_hash: str | None = None) -> Path:
        """Keep every content version without overwriting a same-named upload."""
        digest = content_hash or file_sha256(source)
        target = self.config.source_dir / f"{source.stem}__{digest[:12]}{source.suffix.lower()}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def close(self):
        self.index.close()
        self.metadata.close()


class IngestionPipeline(RAGPipeline):
    """Write-capable runtime used only to build and freeze a corpus."""

    def __init__(self, config: PipelineConfig | None = None):
        super().__init__(config, mode="ingestion")


class RetrievalAnswerPipeline(RAGPipeline):
    """Read-only application runtime; document ingestion is intentionally unavailable."""

    def __init__(self, config: PipelineConfig | None = None):
        super().__init__(config, mode="retrieval")


def _hit_row(hit) -> dict:
    return {
        "chunk_id": hit.chunk.chunk_id,
        "source_file": hit.chunk.source_file,
        "chunk_type": hit.chunk.chunk_type,
        "page": hit.chunk.page_start,
        "sheet_name": hit.chunk.sheet_name,
        "cell_range": hit.chunk.cell_range,
        "dense_rank": hit.dense_rank,
        "sparse_rank": hit.sparse_rank,
        "rrf_score": hit.rrf_score,
        "rerank_score": hit.rerank_score,
        "asset_path": hit.chunk.asset_path,
        "preview": hit.chunk.content[:500],
    }


def _count_statuses(statuses: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status in statuses:
        key = f"{status['stage']}:{status['status']}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_extracted_bundle(root: Path) -> None:
    required = ("corpus_manifest.json", "checksums.json", "metadata.db", "bm25.pkl", "qdrant")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError("Corpus bundle is missing: " + ", ".join(missing))
    manifest = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION or not manifest.get("frozen"):
        raise RuntimeError("Corpus bundle has an unsupported schema or is not frozen")
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    if actual_files != set(checksums):
        missing = sorted(set(checksums) - actual_files)
        unchecked = sorted(actual_files - set(checksums))
        raise RuntimeError(f"Corpus checksum inventory mismatch; missing={missing}, unchecked={unchecked}")
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RuntimeError(f"Corpus artifact is missing or unsafe: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for corpus artifact: {relative}")


def _adopt_or_validate_embedding_options(config: PipelineConfig, embedding: dict[str, Any]) -> None:
    revision = embedding.get("revision")
    if config.retrieval.dense_revision is not None and config.retrieval.dense_revision != revision:
        raise RuntimeError(
            f"Corpus embedding revision is {revision!r}, but config uses "
            f"{config.retrieval.dense_revision!r}."
        )
    instruction = embedding.get("query_instruction")
    if (
        config.retrieval.dense_query_instruction is not None
        and config.retrieval.dense_query_instruction != instruction
    ):
        raise RuntimeError("Corpus and retrieval use different dense query instructions.")
    config.retrieval.dense_revision = revision
    config.retrieval.dense_query_instruction = instruction
