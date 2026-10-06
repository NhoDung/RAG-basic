from __future__ import annotations

from collections import defaultdict

from .config import PipelineConfig
from .models import SearchHit
from .storage import HybridIndex, MetadataStore


class Reranker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None

    def load(self):
        if self.model is not None or not self.config.retrieval.reranker_enabled:
            return
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker(
            self.config.retrieval.reranker_model,
            use_fp16=self.config.retrieval.reranker_use_fp16,
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not self.config.retrieval.reranker_enabled:
            return [0.0] * len(texts)
        self.load()
        pairs = [[query, text] for text in texts]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(score) for score in scores]


class HybridRetriever:
    def __init__(
        self,
        config: PipelineConfig,
        metadata: MetadataStore,
        index: HybridIndex,
        reranker: Reranker,
    ):
        self.config = config
        self.metadata = metadata
        self.index = index
        self.reranker = reranker

    def retrieve(self, query: str, query_variants: list[str] | None = None) -> list[SearchHit]:
        queries = [query]
        for variant in query_variants or []:
            if variant and variant not in queries:
                queries.append(variant)

        rankings: list[tuple[str, list[str]]] = []
        for current_query in queries:
            rankings.append(
                ("dense", self.index.dense_search(current_query, self.config.retrieval.dense_top_k))
            )
            rankings.append(
                ("sparse", self.index.sparse_search(current_query, self.config.retrieval.sparse_top_k))
            )

        scores = defaultdict(float)
        dense_rank = {}
        sparse_rank = {}
        for kind, ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] += 1.0 / (self.config.retrieval.rrf_k + rank)
                target = dense_rank if kind == "dense" else sparse_rank
                target[chunk_id] = min(target.get(chunk_id, rank), rank)

        fused_ids = sorted(scores, key=scores.get, reverse=True)[: self.config.retrieval.fused_top_k]
        hits = []
        for chunk_id in fused_ids:
            chunk = self.metadata.get_chunk(chunk_id)
            if chunk is None:
                continue
            hits.append(
                SearchHit(
                    chunk=chunk,
                    dense_rank=dense_rank.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                    rrf_score=scores[chunk_id],
                )
            )

        if self.config.retrieval.reranker_enabled and hits:
            rerank_scores = self.reranker.score(query, [hit.chunk.content for hit in hits])
            for hit, score in zip(hits, rerank_scores):
                hit.rerank_score = score
            hits.sort(key=lambda hit: hit.rerank_score or float("-inf"), reverse=True)

        return hits[: self.config.retrieval.rerank_top_k]

    def expand_context(self, hits: list[SearchHit]) -> list[dict]:
        contexts = []
        seen_parents = set()
        for hit in hits:
            parent = self.metadata.get_parent(hit.chunk.parent_id)
            if parent is None or parent.parent_id in seen_parents:
                continue
            seen_parents.add(parent.parent_id)
            contexts.append(
                {
                    "source_id": f"SOURCE_{len(contexts) + 1}",
                    "parent": parent,
                    "hit": hit,
                    "citation": build_citation(hit),
                }
            )
        return contexts


def build_citation(hit: SearchHit) -> str:
    chunk = hit.chunk
    if chunk.sheet_name:
        location = f'sheet "{chunk.sheet_name}"'
        if chunk.cell_range:
            location += f", vùng {chunk.cell_range}"
        return f"[{chunk.source_file}, {location}]"
    if chunk.page_start is not None:
        return f"[{chunk.source_file}, trang {chunk.page_start}]"
    if chunk.section_path:
        return f"[{chunk.source_file}, mục \"{' > '.join(chunk.section_path)}\"]"
    return f"[{chunk.source_file}]"
