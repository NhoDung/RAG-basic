from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .chunking import ParentChildChunker
from .config import PipelineConfig
from .generation import LocalQwen
from .paddleocr_vl import PaddleOCRVLAdapter
from .parsers import DocumentParser, save_parsed_document
from .retrieval import HybridRetriever, Reranker
from .storage import DenseEncoder, HybridIndex, MetadataStore


LOGGER = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.config.create_directories()
        self.ocr = PaddleOCRVLAdapter(
            model_name=self.config.parsing.ocr_model_name,
            model_dir=self.config.parsing.ocr_model_dir,
        )
        self.parser = DocumentParser(self.config, self.ocr)
        self.chunker = ParentChildChunker(self.config.chunking)
        self.metadata = MetadataStore(self.config.metadata_db)
        self.encoder = DenseEncoder(self.config)
        self.index = HybridIndex(self.config, self.metadata, self.encoder)
        self.reranker = Reranker(self.config)
        self.retriever = HybridRetriever(
            self.config,
            self.metadata,
            self.index,
            self.reranker,
        )
        self.generator = LocalQwen(self.config)

    def ingest(self, files: list[str | Path], reset: bool = True) -> dict:
        paths = [Path(path) for path in files]
        if not paths:
            raise ValueError("No input files")
        if reset:
            self.metadata.reset()

        all_chunks = []
        report = {"documents": [], "errors": []}
        for source in paths:
            try:
                stored_source = self._copy_source(source)
                document = self.parser.parse(stored_source)
                parents, chunks = self.chunker.chunk(document)
                save_parsed_document(document, self.config.parsed_dir)
                self.metadata.upsert_document(document, parents, chunks)
                all_chunks.extend(chunks)
                report["documents"].append(
                    {
                        "file": source.name,
                        "document_id": document.document_id,
                        "blocks": len(document.blocks),
                        "parents": len(parents),
                        "chunks": len(chunks),
                    }
                )
            except Exception as exc:
                LOGGER.exception("Failed to ingest %s", source)
                report["errors"].append({"file": source.name, "error": str(exc)})

        self.ocr.unload()
        if not all_chunks:
            raise RuntimeError(f"No chunks were created. Errors: {report['errors']}")
        self.index.build(all_chunks, reset=reset)
        report["stats"] = self.metadata.stats()
        manifest = self.config.work_dir / "manifest.json"
        manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def ask(self, query: str) -> dict:
        query = query.strip()
        if not query:
            return {"answer": "Vui lòng nhập câu hỏi.", "citations": [], "trace": []}
        variants = self.generator.rewrite_query(query)
        hits = self.retriever.retrieve(query, variants)
        contexts = self.retriever.expand_context(hits)
        result = self.generator.answer(query, contexts)
        result["query_variants"] = variants
        result["trace"] = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "source_file": hit.chunk.source_file,
                "chunk_type": hit.chunk.chunk_type,
                "dense_rank": hit.dense_rank,
                "sparse_rank": hit.sparse_rank,
                "rrf_score": hit.rrf_score,
                "rerank_score": hit.rerank_score,
                "preview": hit.chunk.content[:500],
            }
            for hit in hits
        ]
        return result

    def export_artifacts(self, destination: str | Path | None = None) -> Path:
        destination = Path(destination or "/kaggle/working/rag_artifacts.zip")
        base_name = destination.with_suffix("")
        self.index.close()
        archive = shutil.make_archive(str(base_name), "zip", self.config.work_dir)
        return Path(archive)

    def _copy_source(self, source: Path) -> Path:
        target = self.config.source_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def close(self):
        self.index.close()
        self.metadata.close()
