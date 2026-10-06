# Multimodal RAG baseline

End-to-end Vietnamese RAG pipeline designed for a free Kaggle GPU session:

```text
PDF / DOCX / Excel
→ native parsing + PaddleOCR-VL-1.6
→ Parent–Child chunking
→ BGE dense embedding + BM25
→ Qdrant + RRF
→ bge-reranker-v2-m3
→ Qwen2.5-7B-Instruct
→ Gradio answer with citations
```

## Chạy trên Kaggle

1. Upload [`notebooks/kaggle_full_pipeline.ipynb`](notebooks/kaggle_full_pipeline.ipynb) lên Kaggle.
2. Chọn GPU T4/P100 và bật Internet.
3. Chạy lần lượt các cell.
4. Trong Gradio, upload tài liệu ở tab `Ingestion`, sau đó hỏi ở tab `Chat`.
5. Tải `rag_artifacts.zip` trước khi Kaggle session kết thúc.

Notebook tự clone repository này. Vì vậy cần push phiên bản source code mới nhất lên
GitHub trước khi chạy notebook đã upload độc lập.

## Chạy trực tiếp trong notebook đã clone repo

```python
from pathlib import Path

from rag_kaggle import PipelineConfig, RAGPipeline

config = PipelineConfig(work_dir=Path("/kaggle/working/rag_runtime"))
pipeline = RAGPipeline(config)

report = pipeline.ingest(["/kaggle/input/my-documents/report.pdf"])
result = pipeline.ask("Quy trình phê duyệt gồm những bước nào?")
print(result)
```

## Cấu trúc chính

```text
rag_kaggle/
├── parsers.py          # PDF, DOCX, Excel
├── paddleocr_vl.py     # PaddleOCR-VL compatibility adapter
├── chunking.py         # Parent–Child chunking
├── storage.py          # SQLite, Qdrant, BM25, dense embedding
├── retrieval.py        # RRF, reranker, parent expansion
├── generation.py       # Query rewrite và Qwen answer
├── pipeline.py         # End-to-end orchestration
└── ui.py               # Gradio demo
```

Thiết kế chi tiết nằm trong [`Architecture.md`](Architecture.md).
