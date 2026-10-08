from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .guardrails import sanitize_context
from .models import Block, Chunk, SearchHit
from .storage import HybridIndex, MetadataStore, release_cuda
from .utils import rows_to_markdown, text_sha1


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
        boost_document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        hits, _ = self.retrieve_with_trace(query, query_variants, filters, keyword_query, top_k, boost_document_ids)
        return hits

    def retrieve_with_trace(
        self,
        query: str,
        query_variants: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        keyword_query: str | None = None,
        top_k: int | None = None,
        boost_document_ids: list[str] | None = None,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        retrieval = self.config.retrieval
        queries = [query]  # The original question is always searched (§9.1).
        for variant in query_variants or []:
            if variant and variant not in queries:
                queries.append(variant)

        rankings: list[tuple[str, list[str], str]] = []
        for current_query in queries:
            rankings.append(("dense", self.index.dense_search(current_query, retrieval.dense_top_k, filters), "all"))
            rankings.append(("sparse", self.index.sparse_search(current_query, retrieval.sparse_top_k, filters), "all"))
        if keyword_query and keyword_query not in queries:
            rankings.append(("sparse", self.index.sparse_search(keyword_query, retrieval.sparse_top_k, filters), "all"))
        if boost_document_ids and retrieval.entity_boost and "document_id" not in (filters or {}):
            # Soft boost: documents that mention an entity from the question get one more
            # dense and sparse ranking restricted to them, fused through RRF, never a hard filter.
            scoped = {**(filters or {}), "document_id": list(dict.fromkeys(boost_document_ids))}
            rankings.append(("dense", self.index.dense_search(query, retrieval.dense_top_k, scoped), "entity"))
            rankings.append(("sparse", self.index.sparse_search(query, retrieval.sparse_top_k, scoped), "entity"))

        scores: dict[str, float] = defaultdict(float)
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}
        for kind, ranking, _scope in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] += 1.0 / (retrieval.rrf_k + rank)
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
            "rankings": [{"retriever": kind, "scope": scope, "ids": ranking[:10]} for kind, ranking, scope in rankings],
            "boost_document_ids": boost_document_ids or [],
            "fused_ids": fused_ids,
        }
        return hits[: top_k or retrieval.rerank_top_k], stage_trace

    def expand_context(self, hits: list[SearchHit], max_chars: int | None = None) -> list[dict]:
        """Chunk + neighbour + relationship expansion with dedup and a character budget (§9.4).

        A chunk is already a whole unit (text run, table or image), so there is no parent
        to fetch. Text chunks are widened with adjacent chunks of the same section; large
        tables, indexed as a preview, are expanded to their full rows within a budget.
        """
        retrieval = self.config.retrieval
        budget = max_chars or self.config.generation.max_context_chars
        contexts: list[dict] = []
        seen_chunks: set[str] = set()
        seen_blocks: set[str] = set()
        seen_hashes: set[str] = set()
        used = 0

        for hit in hits:
            chunk = hit.chunk
            if chunk.chunk_id in seen_chunks:
                continue
            members = [chunk]
            if chunk.chunk_type == "text":
                members = sorted(
                    [chunk, *(n for n in self.metadata.neighbor_chunks(chunk, retrieval.neighbor_chunks) if n.chunk_id not in seen_chunks)],
                    key=lambda item: item.ordinal,
                )
            body = self._compose_body(chunk, members)
            covered = list(dict.fromkeys(block_id for member in members for block_id in member.block_ids))
            body_hash = text_sha1(body)
            if body_hash in seen_hashes:
                continue

            related_parts, related_ids = [], []
            if retrieval.expand_relationships:
                related_parts, related_ids = self._related_blocks(hit, set(covered) | seen_blocks)

            content = "\n\n".join([body, *related_parts])
            if used + len(content) > budget:
                if used + len(body) > budget:
                    continue
                content = body
                related_ids = []
            used += len(content)
            seen_hashes.add(body_hash)
            seen_chunks.update(member.chunk_id for member in members)
            seen_blocks.update(covered)
            seen_blocks.update(related_ids)
            profile = self.metadata.document_profile(chunk.document_id)
            pages = sorted({p for member in members for p in (member.page_start, member.page_end) if p is not None})
            contexts.append(
                {
                    "chunk": chunk,
                    "hit": hit,
                    "hits": [hit],
                    "content": sanitize_context(content),
                    "block_ids": covered,
                    "related_block_ids": related_ids,
                    "citation": build_citation(hit, pages),
                    "asset_paths": [self._resolve_artifact_path(chunk.asset_path)] if chunk.asset_path else [],
                    "table_file": self._resolve_artifact_path(chunk.metadata["table_file"])
                    if chunk.metadata.get("table_file")
                    else None,
                    "info": {
                        "source_file": chunk.source_file,
                        "pages": pages,
                        "section_path": chunk.section_path,
                        # Profile text comes from the document itself: treat it as data.
                        "title": sanitize_context(profile["title"]) if profile.get("title") else None,
                        "doc_number": profile.get("doc_number"),
                        "issue_date": profile.get("issue_date"),
                        "signers": [sanitize_context(name) for name in profile.get("signers") or []],
                    },
                    "order": (chunk.source_file, chunk.document_id, chunk.ordinal),
                }
            )

        # Sort by document and reading order, then assign backend source IDs.
        contexts.sort(key=lambda item: item["order"])
        for index, context in enumerate(contexts, start=1):
            context["source_id"] = f"SOURCE_{index}"
        return contexts

    def _compose_body(self, chunk: Chunk, members: list[Chunk]) -> str:
        if chunk.metadata.get("table_truncated"):
            full = self._full_table_text(chunk)
            if full:
                return full
        if len(members) == 1:
            return chunk.content
        parts = []
        for member in members:
            parts.append(member.content if member.chunk_id == chunk.chunk_id and not parts else _strip_header(member.content))
        return "\n\n".join(parts)

    def _full_table_text(self, chunk: Chunk) -> str | None:
        """Render a preview-only table in full, up to ``max_table_chars_in_context``."""
        block = self.metadata.get_block(chunk.block_ids[0])
        rows = (block.metadata.get("rows") if block else None) or []
        if not rows:
            return None
        inherited = block.metadata.get("inherited_header")
        table_rows = ([list(inherited)] + rows) if inherited else rows
        limit = self.config.retrieval.max_table_chars_in_context
        shown = 1
        while shown < len(table_rows) and len(rows_to_markdown(table_rows[: shown + 1])) <= limit:
            shown += 1
        lines = []
        header = chunk.content.split("\n\n", 1)[0] if chunk.content.startswith(("Section:", "Sheet:", "Range:")) else ""
        if header:
            lines.append(header)
        if block.metadata.get("caption"):
            lines.append(f"Table title: {block.metadata['caption']}")
        lines.append(rows_to_markdown(table_rows[:shown]))
        if shown < len(table_rows):
            lines.append(
                f"[... {len(table_rows) - shown} hàng còn lại không hiển thị; "
                f"bảng đầy đủ: {chunk.metadata.get('table_file')}]"
            )
        return "\n".join(lines)

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


def _strip_header(content: str) -> str:
    """Drop the Section/Sheet/Range header of a neighbouring chunk (same section as the hit)."""
    if content.startswith(("Section:", "Sheet:", "Range:")) and "\n\n" in content:
        return content.split("\n\n", 1)[1]
    return content


def build_citation(hit: SearchHit, pages: list[int] | None = None) -> str:
    chunk = hit.chunk
    if chunk.sheet_name:
        location = f'sheet "{chunk.sheet_name}"'
        if chunk.cell_range:
            location += f", vùng {chunk.cell_range}"
        return f"[{chunk.source_file}, {location}]"
    if chunk.page_start is not None:
        first = min(pages) if pages else chunk.page_start
        last = max(pages) if pages else (chunk.page_end or chunk.page_start)
        label = f"trang {first}" if first == last else f"trang {first}-{last}"
        return f"[{chunk.source_file}, {label}]"
    if chunk.section_path:
        return f"[{chunk.source_file}, mục \"{' > '.join(chunk.section_path)}\"]"
    return f"[{chunk.source_file}]"
