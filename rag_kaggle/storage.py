from __future__ import annotations

import json
import pickle
import re
import sqlite3
import uuid
from pathlib import Path

from .config import PipelineConfig
from .models import ChildChunk, ParentContext, ParsedDocument


class MetadataStore:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path)
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
            CREATE INDEX IF NOT EXISTS idx_blocks_document ON blocks(document_id);
            CREATE INDEX IF NOT EXISTS idx_parents_document ON parents(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
            """
        )
        self.connection.commit()

    def upsert_document(
        self,
        document: ParsedDocument,
        parents: list[ParentContext],
        chunks: list[ChildChunk],
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)",
            (
                document.document_id,
                document.source_file,
                document.file_type,
                document.content_hash,
                json.dumps(document.metadata, ensure_ascii=False),
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
                    json.dumps(block.to_dict(), ensure_ascii=False),
                )
                for block in document.blocks
            ],
        )
        self.connection.executemany(
            "INSERT OR REPLACE INTO parents VALUES (?, ?, ?, ?)",
            [
                (
                    parent.parent_id,
                    parent.document_id,
                    parent.content,
                    json.dumps(parent.to_dict(), ensure_ascii=False),
                )
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
                    json.dumps(chunk.to_dict(), ensure_ascii=False),
                )
                for chunk in chunks
            ],
        )
        self.connection.commit()

    def reset(self) -> None:
        self.connection.executescript(
            """
            DELETE FROM chunks;
            DELETE FROM parents;
            DELETE FROM blocks;
            DELETE FROM documents;
            """
        )
        self.connection.commit()

    def get_parent(self, parent_id: str) -> ParentContext | None:
        row = self.connection.execute(
            "SELECT payload_json FROM parents WHERE parent_id = ?", (parent_id,)
        ).fetchone()
        return ParentContext(**json.loads(row[0])) if row else None

    def get_chunk(self, chunk_id: str) -> ChildChunk | None:
        row = self.connection.execute(
            "SELECT payload_json FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return ChildChunk(**json.loads(row[0])) if row else None

    def list_chunks(self) -> list[ChildChunk]:
        rows = self.connection.execute("SELECT payload_json FROM chunks ORDER BY rowid").fetchall()
        return [ChildChunk(**json.loads(row[0])) for row in rows]

    def stats(self) -> dict[str, int]:
        result = {}
        for table in ("documents", "blocks", "parents", "chunks"):
            result[table] = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return result

    def close(self):
        self.connection.close()


class DenseEncoder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.config.retrieval.dense_model,
            trust_remote_code=True,
            device=self.config.retrieval.dense_device,
        )

    def encode(self, texts: list[str], show_progress: bool = False):
        self.load()
        return self.model.encode(
            texts,
            batch_size=self.config.retrieval.dense_batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    def move_to_cpu(self):
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass


class HybridIndex:
    def __init__(self, config: PipelineConfig, metadata: MetadataStore, encoder: DenseEncoder):
        self.config = config
        self.metadata = metadata
        self.encoder = encoder
        self.client = None
        self.bm25 = None
        self.bm25_chunk_ids: list[str] = []

    def open(self):
        if self.client is None:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(path=str(self.config.qdrant_dir))
        self._load_bm25()

    def build(self, chunks: list[ChildChunk], reset: bool = True):
        if not chunks:
            raise ValueError("No chunks to index")
        self.open()
        vectors = self.encoder.encode([chunk.content for chunk in chunks], show_progress=True)
        dimension = int(vectors.shape[1])

        from qdrant_client.models import Distance, PointStruct, VectorParams

        if reset and self.client.collection_exists(self.config.collection_name):
            self.client.delete_collection(self.config.collection_name)
        if not self.client.collection_exists(self.config.collection_name):
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            points = []
            for chunk, vector in zip(chunks[start : start + batch_size], vectors[start : start + batch_size]):
                payload = chunk.to_dict()
                points.append(
                    PointStruct(
                        id=qdrant_point_id(chunk.chunk_id),
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )
            self.client.upsert(self.config.collection_name, points=points, wait=True)

        self._build_bm25(chunks)

    def dense_search(self, query: str, limit: int) -> list[str]:
        self.open()
        vector = self.encoder.encode([query])[0].tolist()
        try:
            result = self.client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
            ).points
        except AttributeError:
            result = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        return [point.payload["chunk_id"] for point in result]

    def sparse_search(self, query: str, limit: int) -> list[str]:
        self.open()
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize_vi(query))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [self.bm25_chunk_ids[index] for index in ranked[:limit] if scores[index] > 0]

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

    def _build_bm25(self, chunks: list[ChildChunk]):
        from rank_bm25 import BM25Okapi

        self.bm25_chunk_ids = [chunk.chunk_id for chunk in chunks]
        corpus = [tokenize_vi(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        with self.config.bm25_path.open("wb") as stream:
            pickle.dump({"ids": self.bm25_chunk_ids, "index": self.bm25}, stream)

    def _load_bm25(self):
        if self.bm25 is not None or not self.config.bm25_path.exists():
            return
        with self.config.bm25_path.open("rb") as stream:
            payload = pickle.load(stream)
        self.bm25_chunk_ids = payload["ids"]
        self.bm25 = payload["index"]


def tokenize_vi(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
