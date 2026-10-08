from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
import re
import sqlite3
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Callable

from .chunking import table_header_and_body
from .config import PipelineConfig
from .models import Block, ChildChunk, ParentContext, ParsedDocument, Relationship, StageStatus
from .utils import normalize_for_match, text_sha1


LOGGER = logging.getLogger(__name__)
DENSE_VECTOR_NAME = "dense"
FILTER_FIELDS = ("document_id", "chunk_type", "source_file", "sheet_name", "entities")


class MetadataStore:
    """SQLite structured storage: documents, blocks, parents, chunks, graph, status."""

    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                file_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blocks (
                block_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                block_type TEXT NOT NULL,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parents (
                parent_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relationships (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                document_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, relation_type)
            );
            CREATE TABLE IF NOT EXISTS ingestion_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                document_id TEXT,
                source_file TEXT,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                error_code TEXT,
                message TEXT,
                retryable INTEGER NOT NULL DEFAULT 0,
                duration_ms REAL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_blocks_document ON blocks(document_id);
            CREATE INDEX IF NOT EXISTS idx_parents_document ON parents(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_file);
            """
        )
        self.connection.commit()

    # ---------------------------------------------------------------- writes

    def upsert_document(
        self,
        document: ParsedDocument,
        parents: list[ParentContext],
        chunks: list[ChildChunk],
    ) -> None:
        """Replace everything stored for ``document_id`` atomically."""
        with self.connection:
            self._delete_document_rows(document.document_id)
            payload = {key: value for key, value in document.to_dict().items() if key not in ("blocks", "relationships")}
            payload["metadata"] = {k: v for k, v in document.metadata.items() if k != "ocr_raw_pages"}
            self.connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (
                    document.document_id,
                    document.source_file,
                    document.file_type,
                    document.content_hash,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
            self.connection.executemany(
                "INSERT OR REPLACE INTO blocks VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        block.block_id,
                        block.document_id,
                        block.block_type,
                        block.content,
                        json.dumps(block.to_dict(), ensure_ascii=False, default=str),
                    )
                    for block in document.blocks
                ],
            )
            self.connection.executemany(
                "INSERT OR REPLACE INTO parents VALUES (?, ?, ?, ?)",
                [
                    (parent.parent_id, parent.document_id, parent.content, json.dumps(parent.to_dict(), ensure_ascii=False))
                    for parent in parents
                ],
            )
            self.connection.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.parent_id,
                        chunk.document_id,
                        chunk.content,
                        json.dumps(chunk.to_dict(), ensure_ascii=False, default=str),
                    )
                    for chunk in chunks
                ],
            )
            self.connection.executemany(
                "INSERT OR REPLACE INTO relationships VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        relation.source_id,
                        relation.target_id,
                        relation.relation_type,
                        relation.document_id,
                        json.dumps(relation.metadata, ensure_ascii=False, default=str),
                    )
                    for relation in document.relationships
                ],
            )

    def delete_document(self, document_id: str) -> None:
        with self.connection:
            self._delete_document_rows(document_id)

    def _delete_document_rows(self, document_id: str) -> None:
        for table in ("chunks", "parents", "blocks", "relationships", "documents"):
            self.connection.execute(f"DELETE FROM {table} WHERE document_id = ?", (document_id,))

    def reset(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                DELETE FROM chunks;
                DELETE FROM parents;
                DELETE FROM blocks;
                DELETE FROM relationships;
                DELETE FROM documents;
                DELETE FROM settings WHERE key LIKE 'index.%';
                """
            )

    def record_status(self, status: StageStatus, run_id: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO ingestion_status
                (run_id, document_id, source_file, stage, status, error_code, message, retryable, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    status.document_id,
                    status.source_file,
                    status.stage,
                    status.status,
                    status.error_code,
                    status.message,
                    int(status.retryable),
                    status.duration_ms,
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def set_setting(self, key: str, value: Any) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False))
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ----------------------------------------------------------------- reads

    def get_document(self, document_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT payload_json FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def documents_by_source(self, source_file: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT document_id, content_hash FROM documents WHERE source_file = ?", (source_file,)
        ).fetchall()
        return [dict(row) for row in rows]

    def document_by_original_and_hash(self, original_file_name: str, content_hash: str) -> dict | None:
        """Find an already indexed upload independently of its stored filename."""
        row = self.connection.execute(
            """SELECT document_id FROM documents
               WHERE content_hash = ?
                 AND json_extract(payload_json, '$.metadata.original_file_name') = ?
               LIMIT 1""",
            (content_hash, original_file_name),
        ).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        rows = self.connection.execute(
            "SELECT document_id, source_file, file_type, content_hash FROM documents ORDER BY source_file"
        ).fetchall()
        return [dict(row) for row in rows]

    def list_values(self, field: str) -> list[str]:
        if field == "source_file":
            rows = self.connection.execute("SELECT DISTINCT source_file FROM documents ORDER BY 1").fetchall()
            return [row[0] for row in rows]
        if field == "sheet_name":
            rows = self.connection.execute(
                "SELECT DISTINCT json_extract(payload_json, '$.sheet_name') FROM chunks "
                "WHERE json_extract(payload_json, '$.sheet_name') IS NOT NULL ORDER BY 1"
            ).fetchall()
            return [row[0] for row in rows]
        if field == "chunk_type":
            rows = self.connection.execute(
                "SELECT DISTINCT json_extract(payload_json, '$.chunk_type') FROM chunks ORDER BY 1"
            ).fetchall()
            return [row[0] for row in rows if row[0]]
        if field == "entity":
            rows = self.connection.execute(
                "SELECT DISTINCT value FROM chunks, json_each(chunks.payload_json, '$.entities') ORDER BY 1"
            ).fetchall()
            return [row[0] for row in rows if row[0]]
        raise ValueError(field)

    def get_parent(self, parent_id: str) -> ParentContext | None:
        row = self.connection.execute(
            "SELECT payload_json FROM parents WHERE parent_id = ?", (parent_id,)
        ).fetchone()
        return ParentContext(**json.loads(row[0])) if row else None

    def chunks_in_parent(self, parent_id: str) -> list[ChildChunk]:
        """Chunks of one section in reading order (insertion order)."""
        rows = self.connection.execute(
            "SELECT payload_json FROM chunks WHERE parent_id = ? ORDER BY rowid", (parent_id,)
        ).fetchall()
        return [ChildChunk(**json.loads(row[0])) for row in rows]

    def get_document_profile(self, document_id: str) -> dict:
        document = self.get_document(document_id) or {}
        return (document.get("metadata") or {}).get("profile") or {}

    def get_chunk(self, chunk_id: str) -> ChildChunk | None:
        row = self.connection.execute(
            "SELECT payload_json FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return ChildChunk(**json.loads(row[0])) if row else None

    def get_block(self, block_id: str) -> Block | None:
        row = self.connection.execute(
            "SELECT payload_json FROM blocks WHERE block_id = ?", (block_id,)
        ).fetchone()
        return Block(**json.loads(row[0])) if row else None

    def get_blocks(self, block_ids: list[str]) -> list[Block]:
        if not block_ids:
            return []
        placeholders = ",".join("?" for _ in block_ids)
        rows = self.connection.execute(
            f"SELECT payload_json FROM blocks WHERE block_id IN ({placeholders})", list(block_ids)
        ).fetchall()
        by_id = {block.block_id: block for block in (Block(**json.loads(row[0])) for row in rows)}
        return [by_id[block_id] for block_id in block_ids if block_id in by_id]

    def related(self, block_ids: list[str], relation_types: tuple[str, ...]) -> list[Relationship]:
        """Relationships touching ``block_ids`` in either direction."""
        if not block_ids:
            return []
        placeholders = ",".join("?" for _ in block_ids)
        type_placeholders = ",".join("?" for _ in relation_types)
        rows = self.connection.execute(
            f"""SELECT * FROM relationships
            WHERE relation_type IN ({type_placeholders})
              AND (source_id IN ({placeholders}) OR target_id IN ({placeholders}))""",
            [*relation_types, *block_ids, *block_ids],
        ).fetchall()
        return [
            Relationship(row["source_id"], row["target_id"], row["relation_type"], row["document_id"], json.loads(row["metadata_json"]))
            for row in rows
        ]

    def list_chunks(self) -> list[ChildChunk]:
        rows = self.connection.execute("SELECT payload_json FROM chunks ORDER BY rowid").fetchall()
        return [ChildChunk(**json.loads(row[0])) for row in rows]

    def chunk_ids_for_document(self, document_id: str) -> list[str]:
        rows = self.connection.execute("SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,)).fetchall()
        return [row[0] for row in rows]

    def statuses(self, run_id: str | None = None, limit: int = 200) -> list[dict]:
        if run_id:
            rows = self.connection.execute(
                "SELECT * FROM ingestion_status WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM ingestion_status ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        result = {}
        for table in ("documents", "blocks", "parents", "chunks", "relationships"):
            result[table] = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return result

    # ------------------------------------------------------- embedding cache

    def cached_vectors(self, keys: list[str]) -> dict[str, bytes]:
        result = {}
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT cache_key, vector FROM embedding_cache WHERE cache_key IN ({placeholders})", batch
            ).fetchall()
            result.update({row[0]: row[1] for row in rows})
        return result

    def store_vectors(self, model: str, items: list[tuple[str, bytes]]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?)",
                [(key, model, blob) for key, blob in items],
            )

    def close(self):
        self.connection.close()


class DenseEncoder:
    def __init__(self, config: PipelineConfig, metadata: MetadataStore | None = None):
        self.config = config
        self.metadata = metadata
        self.model = None
        self.model_name: str | None = None

    def load(self):
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        name = self.config.retrieval.dense_model
        if not name:
            raise RuntimeError("dense_model is not configured; restore a corpus manifest or set a model name")
        try:
            self.model = SentenceTransformer(
                name,
                revision=self.config.retrieval.dense_revision,
                trust_remote_code=True,
                device=self.config.retrieval.dense_device,
            )
        except Exception as exc:
            fallback = self.config.retrieval.dense_fallback_model
            if not fallback or fallback == name:
                raise
            LOGGER.warning("Dense model %s failed to load (%s); using fallback %s", name, exc, fallback)
            release_cuda()
            name = fallback
            self.model = SentenceTransformer(
                name,
                revision=self.config.retrieval.dense_revision,
                trust_remote_code=True,
                device=self.config.retrieval.dense_device,
            )
        self.model_name = name

    def encode(self, texts: list[str], show_progress: bool = False, is_query: bool = False):
        import numpy as np

        self.load()
        instruction = self.config.retrieval.dense_query_instruction if is_query else None
        if instruction:
            texts = [f"{instruction}{text}" for text in texts]

        revision = self.config.retrieval.dense_revision or "default"
        cache_keys = [text_sha1(f"{self.model_name}@{revision}|{text}") for text in texts]
        cached: dict[str, bytes] = {}
        if self.metadata is not None and not is_query:
            cached = self.metadata.cached_vectors(cache_keys)
        missing = [index for index, key in enumerate(cache_keys) if key not in cached]

        vectors: dict[int, Any] = {
            index: np.frombuffer(cached[key], dtype=np.float32) for index, key in enumerate(cache_keys) if key in cached
        }
        if missing:
            encoded = self._encode_with_retry([texts[index] for index in missing], show_progress)
            for index, vector in zip(missing, encoded):
                vectors[index] = np.asarray(vector, dtype=np.float32)
            if self.metadata is not None and not is_query:
                self.metadata.store_vectors(
                    self.model_name, [(cache_keys[index], vectors[index].tobytes()) for index in missing]
                )
        return np.vstack([vectors[index] for index in range(len(texts))])

    def _encode_with_retry(self, texts: list[str], show_progress: bool):
        """Halve the batch size on CUDA OOM and retry a bounded number of times (§15)."""
        batch_size = self.config.retrieval.dense_batch_size
        for _ in range(4):
            try:
                return self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size == 1:
                    raise
                release_cuda()
                batch_size = max(1, batch_size // 2)
                LOGGER.warning("Embedding OOM; retrying with batch_size=%s", batch_size)
        raise RuntimeError("Embedding failed after reducing batch size")

    def unload(self):
        self.model = None
        release_cuda()


class HybridIndex:
    def __init__(self, config: PipelineConfig, metadata: MetadataStore, encoder: DenseEncoder):
        self.config = config
        self.metadata = metadata
        self.encoder = encoder
        self.client = None
        self.bm25 = None
        self.bm25_chunk_ids: list[str] = []
        self.bm25_fields: list[dict[str, Any]] = []

    def open(self):
        if self.client is None:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(path=str(self.config.qdrant_dir))
        self._load_bm25()

    def build(
        self,
        chunks: list[ChildChunk],
        reset: bool = True,
        removed_document_ids: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        """Upsert ``chunks`` (idempotent IDs) and rebuild BM25 over the full corpus."""
        notify = progress or (lambda _message: None)
        self.open()
        from qdrant_client.models import Distance, PointStruct, VectorParams

        indexed_model = self.metadata.get_setting("index.dense_model")
        if reset and self.client.collection_exists(self.config.collection_name):
            self.client.delete_collection(self.config.collection_name)
            indexed_model = None

        for document_id in removed_document_ids or []:
            self.delete_document(document_id)

        if chunks:
            notify(f"Dense embedding {len(chunks)} chunk(s) on {self.config.retrieval.dense_device}.")
            vectors = self.encoder.encode([chunk.content for chunk in chunks], show_progress=True)
            notify("Dense embedding complete; writing vectors to Qdrant.")
            model_name = self.encoder.model_name
            if indexed_model and indexed_model != model_name:
                raise RuntimeError(
                    f"Index was built with {indexed_model} but current dense model is {model_name}. "
                    "Use the same model or re-ingest with reset=True."
                )
            dimension = int(vectors.shape[1])
            if not self.client.collection_exists(self.config.collection_name):
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config={DENSE_VECTOR_NAME: VectorParams(size=dimension, distance=Distance.COSINE)},
                )
                self._create_payload_indexes()
            total_batches = (len(chunks) + 63) // 64
            for batch_number, start in enumerate(range(0, len(chunks), 64), start=1):
                points = [
                    PointStruct(
                        id=qdrant_point_id(chunk.chunk_id),
                        vector={DENSE_VECTOR_NAME: vector.tolist()},
                        payload=chunk.payload(),
                    )
                    for chunk, vector in zip(chunks[start : start + 64], vectors[start : start + 64])
                ]
                self.client.upsert(self.config.collection_name, points=points, wait=True)
                notify(f"Qdrant upsert batch {batch_number}/{total_batches}.")
            self.metadata.set_setting("index.dense_model", model_name)
            self.metadata.set_setting("index.dimension", dimension)
            # Persist the model that actually produced the corpus, including fallback use.
            self.config.retrieval.dense_model = model_name

        notify("Rebuilding BM25 index over the full corpus.")
        self._build_bm25(self.metadata.list_chunks())

    def delete_document(self, document_id: str) -> None:
        self.open()
        if not self.client.collection_exists(self.config.collection_name):
            return
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            ),
            wait=True,
        )

    def dense_search(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[str]:
        self.open()
        if not self.client.collection_exists(self.config.collection_name):
            return []
        indexed_model = self.metadata.get_setting("index.dense_model")
        self.encoder.load()
        if indexed_model and self.encoder.model_name != indexed_model:
            raise RuntimeError(
                f"Query encoder {self.encoder.model_name} differs from index model {indexed_model}; "
                "document and query must use the same dense model."
            )
        vector = self.encoder.encode([query], is_query=True)[0].tolist()
        query_filter = build_qdrant_filter(filters)
        try:
            result = self.client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=["chunk_id"],
            ).points
        except AttributeError:
            result = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=(DENSE_VECTOR_NAME, vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=["chunk_id"],
            )
        return [point.payload["chunk_id"] for point in result]

    def sparse_search(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[str]:
        self.open()
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize_vi(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results = []
        for index in ranked:
            if scores[index] <= 0:
                break
            if filters and not _matches_filters(self.bm25_fields[index], filters):
                continue
            results.append(self.bm25_chunk_ids[index])
            if len(results) >= limit:
                break
        return results

    def entity_search(self, query: str, limit: int, filters: dict[str, Any] | None = None) -> list[str]:
        """Chunks tagged with an entity the question mentions, best BM25 match first.

        A soft signal fused with dense/sparse by RRF: a question naming "Nguyễn Văn A"
        pulls up the chunks about that person without excluding anything else.
        """
        self.open()
        if self.bm25 is None:
            return []
        normalized_query = normalize_for_match(query)
        mentioned = {
            entity
            for fields in self.bm25_fields
            for entity in (fields.get("entities") or [])
            if len(entity) >= 3 and re.search(r"(?<!\w)" + re.escape(normalize_for_match(entity)) + r"(?!\w)", normalized_query)
        }
        if not mentioned:
            return []
        scores = self.bm25.get_scores(tokenize_vi(query))
        candidates = [
            index
            for index, fields in enumerate(self.bm25_fields)
            if mentioned.intersection(fields.get("entities") or [])
            and (not filters or _matches_filters(fields, filters))
        ]
        candidates.sort(key=lambda index: (-scores[index], index))
        return [self.bm25_chunk_ids[index] for index in candidates[:limit]]

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def _create_payload_indexes(self):
        import warnings

        from qdrant_client.models import PayloadSchemaType

        for field in (*FILTER_FIELDS, "section_path"):
            try:
                with warnings.catch_warnings():
                    # Embedded/local Qdrant ignores payload indexes; server Qdrant uses them.
                    warnings.simplefilter("ignore", UserWarning)
                    self.client.create_payload_index(
                        self.config.collection_name, field_name=field, field_schema=PayloadSchemaType.KEYWORD
                    )
            except Exception:
                pass

    def _build_bm25(self, chunks: list[ChildChunk]):
        if not chunks:
            self.bm25, self.bm25_chunk_ids, self.bm25_fields = None, [], []
            if self.config.bm25_path.exists():
                self.config.bm25_path.unlink()
            return
        from rank_bm25 import BM25Okapi

        self.bm25_chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.bm25_fields = [{field: getattr(chunk, field) for field in FILTER_FIELDS} for chunk in chunks]
        self.bm25 = BM25Okapi([tokenize_vi(self._sparse_text(chunk)) for chunk in chunks])
        with self.config.bm25_path.open("wb") as stream:
            pickle.dump({"ids": self.bm25_chunk_ids, "fields": self.bm25_fields, "index": self.bm25}, stream)

    def _sparse_text(self, chunk: ChildChunk) -> str:
        """Text indexed by BM25. A large table is embedded as a short preview, but a keyword that
        names any of its rows (the first column: code, name, branch, ...) must still find it."""
        if chunk.metadata.get("table_mode") != "preview" or not chunk.block_ids:
            return chunk.content
        block = self.metadata.get_block(chunk.block_ids[0])
        if block is None:
            return chunk.content
        _, body = table_header_and_body(block)
        keys = dict.fromkeys(row[0] for row in body if row and row[0])
        return chunk.content + "\n" + "\n".join(keys)

    def _load_bm25(self):
        if self.bm25 is not None or not self.config.bm25_path.exists():
            return
        with self.config.bm25_path.open("rb") as stream:
            payload = pickle.load(stream)
        self.bm25_chunk_ids = payload["ids"]
        self.bm25 = payload["index"]
        self.bm25_fields = payload.get("fields") or [{} for _ in self.bm25_chunk_ids]


def _matches_filters(fields: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected in (None, "", []):
            continue
        values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        actual = fields.get(key)
        if isinstance(actual, (list, tuple, set)):  # e.g. entities: any overlap matches
            if not set(actual).intersection(values):
                return False
        elif actual not in values:
            return False
    return True


def build_qdrant_filter(filters: dict[str, Any] | None):
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    conditions = []
    for key, value in filters.items():
        if value in (None, "", []) or key not in FILTER_FIELDS:
            continue
        if isinstance(value, (list, tuple, set)):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions) if conditions else None


def tokenize_vi(text: str) -> list[str]:
    """Lowercase + NFC syllable tokenizer; keeps numbers, codes and stopwords (§8.2)."""
    text = unicodedata.normalize("NFC", text.lower())
    tokens = re.findall(r"[\wÀ-ỹ]+(?:[-/.][\wÀ-ỹ]+)*", text, flags=re.UNICODE)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if re.search(r"[-/.]", token):
            # "499.000" -> also "499000"; "the-visa" -> parts, so codes match both ways.
            expanded.append(re.sub(r"[-/.]", "", token))
            expanded.extend(part for part in re.split(r"[-/.]", token) if part)
    return expanded


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def release_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
