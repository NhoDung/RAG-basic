from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .chunking import table_header_and_body
from .config import PipelineConfig
from .guardrails import sanitize_context
from .knowledge import format_profile
from .models import Block, SearchHit
from .storage import HybridIndex, MetadataStore, release_cuda, tokenize_vi
from .utils import rows_to_markdown, text_sha1

MAX_TABLE_ROWS_SCANNED = 50000


EXPANSION_RELATIONS = ("captioned_by", "referenced_by", "visualizes", "continues", "derived_from")
RELATION_LABELS = {
    "captioned_by": "caption",
    "referenced_by": "đoạn văn tham chiếu",
    "visualizes": "dữ liệu nguồn của biểu đồ",
    "continues": "phần nối tiếp của bảng",
    "derived_from": "mô tả/nguồn hình",
}


class Reranker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None

    def load(self):
        if self.model is not None or not self.config.retrieval.reranker_enabled:
            return
        from FlagEmbedding import FlagReranker

        kwargs = {
            "use_fp16": self.config.retrieval.reranker_use_fp16,
            "devices": self.config.retrieval.reranker_device,
        }
        try:
            self.model = FlagReranker(self.config.retrieval.reranker_model, **kwargs)
        except TypeError:
            # Old FlagEmbedding releases did not expose ``devices``. The Kaggle
            # requirements pin a new enough version, but this keeps local use usable.
            kwargs.pop("devices")
            self.model = FlagReranker(self.config.retrieval.reranker_model, **kwargs)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not self.config.retrieval.reranker_enabled or not texts:
            return [0.0] * len(texts)
        self.load()
        pairs = [[query, text] for text in texts]
        scores = self.model.compute_score(
            pairs, batch_size=self.config.retrieval.reranker_batch_size, normalize=True
        )
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(score) for score in scores]

    def unload(self):
        self.model = None
        release_cuda()


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

    def retrieve(
        self,
        query: str,
        query_variants: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        keyword_query: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        hits, _ = self.retrieve_with_trace(query, query_variants, filters, keyword_query, top_k)
        return hits

    def retrieve_with_trace(
        self,
        query: str,
        query_variants: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        keyword_query: str | None = None,
        top_k: int | None = None,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        retrieval = self.config.retrieval
        queries = [query]  # The original question is always searched (§9.1).
        for variant in query_variants or []:
            if variant and variant not in queries:
                queries.append(variant)

        rankings: list[tuple[str, list[str]]] = []
        for current_query in queries:
            rankings.append(("dense", self.index.dense_search(current_query, retrieval.dense_top_k, filters)))
            rankings.append(("sparse", self.index.sparse_search(current_query, retrieval.sparse_top_k, filters)))
        if keyword_query and keyword_query not in queries:
            rankings.append(("sparse", self.index.sparse_search(keyword_query, retrieval.sparse_top_k, filters)))
        # Soft signal: chunks tagged with a person/place/signer the question names.
        entity_ids = self.index.entity_search(query, retrieval.sparse_top_k, filters)
        if entity_ids:
            rankings.append(("entity", entity_ids))

        scores: dict[str, float] = defaultdict(float)
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}
        for kind, ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] += 1.0 / (retrieval.rrf_k + rank)
                if kind == "entity":
                    continue
                target = dense_rank if kind == "dense" else sparse_rank
                target[chunk_id] = min(target.get(chunk_id, rank), rank)

        fused_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[: retrieval.fused_top_k]
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

        if retrieval.reranker_enabled and hits:
            rerank_scores = self.reranker.score(query, [hit.chunk.content for hit in hits])
            for hit, score in zip(hits, rerank_scores):
                hit.rerank_score = score
            # A score of 0.0 is a valid score; only missing scores sort last.
            hits.sort(
                key=lambda hit: (hit.rerank_score if hit.rerank_score is not None else float("-inf"), hit.rrf_score),
                reverse=True,
            )

        stage_trace = {
            "queries": queries,
            "keyword_query": keyword_query,
            "filters": filters or {},
            "rankings": [{"retriever": kind, "ids": ranking[:10]} for kind, ranking in rankings],
            "fused_ids": fused_ids,
        }
        return hits[: top_k or retrieval.rerank_top_k], stage_trace

    def expand_context(
        self, hits: list[SearchHit], max_chars: int | None = None, query: str | None = None
    ) -> list[dict]:
        """Section + relationship expansion with dedup and a character budget (§9.4)."""
        retrieval = self.config.retrieval
        budget = max_chars or self.config.generation.max_context_chars
        contexts: list[dict] = []
        by_parent: dict[str, dict] = {}
        seen_blocks: set[str] = set()
        seen_hashes: set[str] = set()
        profiles: dict[str, str] = {}
        used = 0

        for hit in hits:
            existing = by_parent.get(hit.chunk.parent_id)
            if existing is not None:
                existing["hits"].append(hit)
                rows = sanitize_context(self._table_rows_context(hit, query))
                if rows and used + len(rows) + 2 <= budget:
                    existing["content"] += "\n\n" + rows
                    used += len(rows) + 2
                continue
            parent = self.metadata.get_parent(hit.chunk.parent_id)
            if parent is None:
                continue

            if len(parent.content) <= retrieval.max_parent_chars_in_context:
                body, covered = parent.content, list(parent.block_ids)
            else:
                # Large section: the matched chunk plus its neighbours in reading order,
                # instead of the whole section.
                body, covered = self._neighbor_window(hit)
            body_hash = text_sha1(body)
            if body_hash in seen_hashes:
                continue

            extra_parts = []
            table_rows = self._table_rows_context(hit, query)
            if table_rows:
                extra_parts.append(table_rows)
            related_parts, related_ids = [], []
            if retrieval.expand_relationships:
                related_parts, related_ids = self._related_blocks(hit, set(covered) | seen_blocks)

            content = "\n\n".join([body, *extra_parts, *related_parts])
            if used + len(content) > budget:
                content = "\n\n".join([body, *extra_parts])
                related_ids = []
                if used + len(content) > budget:
                    content = body
                    if used + len(body) > budget:
                        continue
            used += len(content)
            seen_hashes.add(body_hash)
            seen_blocks.update(covered)
            seen_blocks.update(related_ids)
            if parent.document_id not in profiles:
                profiles[parent.document_id] = format_profile(self.metadata.get_document_profile(parent.document_id))
            context = {
                "parent": parent,
                "hit": hit,
                "hits": [hit],
                "content": sanitize_context(content),
                "block_ids": covered,
                "related_block_ids": related_ids,
                "citation": build_citation(hit),
                "document_info": profiles[parent.document_id],
                "asset_paths": [self._resolve_artifact_path(hit.chunk.asset_path)] if hit.chunk.asset_path else [],
                "order": (parent.source_file, parent.metadata.get("reading_order", 0)),
            }
            by_parent[parent.parent_id] = context
            contexts.append(context)

        # Sort by document and reading order, then assign backend source IDs.
        contexts.sort(key=lambda item: item["order"])
        for index, context in enumerate(contexts, start=1):
            context["source_id"] = f"SOURCE_{index}"
        return contexts

    def _neighbor_window(self, hit: SearchHit) -> tuple[str, list[str]]:
        """Matched chunk plus up to ``max_neighbor_chunks`` chunks on each side of the same section."""
        retrieval = self.config.retrieval
        siblings = self.metadata.chunks_in_parent(hit.chunk.parent_id)
        position = next((index for index, chunk in enumerate(siblings) if chunk.chunk_id == hit.chunk.chunk_id), None)
        covered = list(hit.chunk.block_ids)
        if position is None or retrieval.max_neighbor_chunks <= 0:
            return hit.chunk.content, covered
        reach = retrieval.max_neighbor_chunks
        before = siblings[max(0, position - reach) : position]
        after = siblings[position + 1 : position + 1 + reach]
        while True:
            parts = [f"[ĐOẠN LIỀN TRƯỚC]\n{strip_prefix(chunk.content)}" for chunk in before]
            parts.append(hit.chunk.content)
            parts.extend(f"[ĐOẠN LIỀN SAU]\n{strip_prefix(chunk.content)}" for chunk in after)
            text = "\n\n".join(parts)
            if len(text) <= retrieval.max_parent_chars_in_context or not (before or after):
                break
            # Drop the farthest neighbour first, alternating sides; the hit itself always stays.
            if after and len(after) >= len(before):
                after = after[:-1]
            else:
                before = before[1:]
        for chunk in [*before, *after]:
            covered.extend(block_id for block_id in chunk.block_ids if block_id not in covered)
        return text, covered

    def _table_rows_context(self, hit: SearchHit, query: str | None) -> str:
        """For a large table (preview chunk), the rows that best match the question."""
        if hit.chunk.metadata.get("table_mode") != "preview" or not hit.chunk.block_ids:
            return ""
        block = self.metadata.get_block(hit.chunk.block_ids[0])
        if block is None or not block.metadata.get("rows"):
            return ""
        header, body = table_header_and_body(block)
        selected = select_relevant_rows(body, query or "", self.config.retrieval.max_table_rows_in_context)
        if not selected:
            return ""
        table = rows_to_markdown(([header] if header else []) + selected)
        return f"[TABLE ROWS – {len(selected)}/{len(body)} dòng của bảng lớn khớp câu hỏi nhất]\n{table}"

    def _resolve_artifact_path(self, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else self.config.work_dir / path)

    def _related_blocks(self, hit: SearchHit, exclude: set[str]) -> tuple[list[str], list[str]]:
        relations = self.metadata.related(hit.chunk.block_ids, EXPANSION_RELATIONS)
        candidates: list[tuple[str, str]] = []
        for relation in relations:
            other = relation.target_id if relation.source_id in hit.chunk.block_ids else relation.source_id
            if other not in exclude and other not in {item for item, _ in candidates}:
                candidates.append((other, relation.relation_type))
        candidates = candidates[: self.config.retrieval.max_related_blocks]
        blocks = {block.block_id: block for block in self.metadata.get_blocks([item for item, _ in candidates])}
        parts, ids = [], []
        for block_id, relation_type in candidates:
            block = blocks.get(block_id)
            if block is None:
                continue
            parts.append(render_related_block(block, relation_type))
            ids.append(block_id)
        return parts, ids


def strip_prefix(content: str) -> str:
    """Drop the "Document/Section/Page/Content type" header that every chunk starts with."""
    head, separator, rest = content.partition("\n\n")
    return rest if separator and head.startswith("Document:") else content


def select_relevant_rows(rows: list[list[str]], query: str, limit: int) -> list[list[str]]:
    """Rows of a large table that share the most (rare) terms with the question, in table order."""
    query_terms = {term for term in tokenize_vi(query) if len(term) >= 2}
    scanned = rows[:MAX_TABLE_ROWS_SCANNED]
    if not query_terms or not scanned or limit <= 0:
        return []
    row_terms = [set(tokenize_vi(" ".join(row))) for row in scanned]
    document_frequency = {term: sum(1 for terms in row_terms if term in terms) for term in query_terms}
    weights = {
        term: math.log(1 + len(scanned) / (1 + count)) for term, count in document_frequency.items() if count
    }
    scored = []
    for index, terms in enumerate(row_terms):
        score = sum(weight for term, weight in weights.items() if term in terms)
        if score > 0:
            scored.append((score, index))
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    return [scanned[index] for index in sorted(index for _, index in best)]


def render_related_block(block: Block, relation_type: str, limit: int = 1500) -> str:
    location = []
    if block.page is not None:
        location.append(f"trang {block.page}")
    if block.sheet_name:
        location.append(f'sheet "{block.sheet_name}"')
    if block.cell_range:
        location.append(f"vùng {block.cell_range}")
    content = block.content
    if len(content) > limit:
        content = content[:limit].rsplit("\n", 1)[0] + "\n..."
    header = f"[RELATED {block.block_type.upper()} – {RELATION_LABELS.get(relation_type, relation_type)}"
    header += f", {', '.join(location)}]" if location else "]"
    return f"{header}\n{content}"


def build_citation(hit: SearchHit) -> str:
    chunk = hit.chunk
    if chunk.sheet_name:
        location = f'sheet "{chunk.sheet_name}"'
        if chunk.cell_range:
            location += f", vùng {chunk.cell_range}"
        return f"[{chunk.source_file}, {location}]"
    if chunk.page_start is not None:
        pages = (
            f"trang {chunk.page_start}"
            if chunk.page_end in (None, chunk.page_start)
            else f"trang {chunk.page_start}-{chunk.page_end}"
        )
        return f"[{chunk.source_file}, {pages}]"
    if chunk.section_path:
        return f"[{chunk.source_file}, mục \"{' > '.join(chunk.section_path)}\"]"
    return f"[{chunk.source_file}]"
