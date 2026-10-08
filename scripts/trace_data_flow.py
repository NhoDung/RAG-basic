"""Trace documents through the real pipeline and print what each stage produced.

    python scripts/trace_data_flow.py file.pdf file.docx file.xlsx --query "Phí thẻ Visa Gold?"

Stages shown: parsed blocks -> chunks (the exact text that is embedded) -> Qdrant payload
fields -> BM25 tokens -> retrieved hits -> context sent to the answer model.

Runs offline without a GPU: embeddings use a hashed bag-of-words stand-in (so the dense
ranking is only a rough proxy; chunk contents, payloads and BM25 are the real ones) and OCR
is disabled, so scanned pages produce no text. Use it to inspect parsing and chunking.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_kaggle import PipelineConfig, RAGPipeline  # noqa: E402
from rag_kaggle.storage import tokenize_vi  # noqa: E402


class HashedBagOfWords:
    def encode(self, texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np

        vectors = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize_vi(text):
                vectors[row, hash(token) % 256] += 1.0
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--query", help="question to run through retrieval and context expansion")
    parser.add_argument("--show-chars", type=int, default=400, help="characters of each chunk to print")
    args = parser.parse_args()

    config = PipelineConfig(work_dir=Path(tempfile.mkdtemp(prefix="trace_")) / "runtime")
    config.parsing.enable_ocr = False
    config.retrieval.reranker_enabled = False
    config.runtime.unload_models_between_stages = False
    pipeline = RAGPipeline(config)
    pipeline.encoder.model = HashedBagOfWords()
    pipeline.encoder.model_name = "hashed-bow"
    report = pipeline.ingest(args.files)
    for error in report["errors"]:
        print("ERROR:", error)

    print("\n== 1. BLOCKS (parser output) ==")
    for row in pipeline.metadata.connection.execute("SELECT payload_json FROM blocks ORDER BY rowid"):
        block = json.loads(row[0])
        print(
            f"{block['source_file'][:24]:24} {block['block_type']:8} page={block['page']} sheet={block['sheet_name']} "
            f"chars={len(block['content']):6} lines={block['content'].count(chr(10)) + 1:4} | {block['content'][:50]!r}"
        )

    chunks = pipeline.metadata.list_chunks()
    print(f"\n== 2. CHUNKS ({len(chunks)}) - exactly the text that is embedded and BM25-indexed ==")
    for index, chunk in enumerate(chunks):
        extra = {key: value for key, value in chunk.metadata.items() if key in ("table_mode", "n_rows", "n_cols", "xlsx_path")}
        print(
            f"\n[{index}] {chunk.source_file} | {chunk.chunk_type} | chars={len(chunk.content)} tokens~{chunk.token_count} "
            f"| entities={chunk.entities} keywords={chunk.keywords} {extra or ''}"
        )
        print(chunk.content[: args.show_chars] + ("..." if len(chunk.content) > args.show_chars else ""))

    if chunks:
        point = pipeline.index.client.scroll(config.collection_name, limit=1, with_payload=True)[0][0]
        print("\n== 3. QDRANT PAYLOAD fields ==", sorted(point.payload))
        print("== 4. BM25 tokens of chunk [0] (first 30) ==", tokenize_vi(chunks[0].content)[:30])

    if args.query:
        hits = pipeline.retriever.retrieve(args.query)
        print(f"\n== 5. RETRIEVAL for {args.query!r} ==")
        for hit in hits:
            print(
                f"dense_rank={hit.dense_rank} sparse_rank={hit.sparse_rank} rrf={hit.rrf_score:.4f} "
                f"{hit.chunk.source_file} {hit.chunk.chunk_type} p={hit.chunk.page_start}"
            )
        print("\n== 6. CONTEXT sent to the answer model ==")
        for context in pipeline.retriever.expand_context(hits, query=args.query):
            print(f"\n[{context['source_id']}] {context['citation']} chars={len(context['content'])}")
            print(context["content"][: args.show_chars * 2])
    pipeline.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
