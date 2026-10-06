# Multimodal RAG baseline

End-to-end Vietnamese RAG pipeline designed for a free Kaggle GPU session:

```text
PDF / DOCX / Excel
→ validation + native parsing + PaddleOCR-VL-1.6 (+ Qwen2.5-VL tuỳ chọn)
→ document graph (caption, tham chiếu, chart → bảng nguồn, bảng nối trang)
→ Parent–Child chunking theo modality
→ BGE dense embedding + BM25
→ Qdrant + RRF (+ query rewrite và filter tuỳ chọn)
→ bge-reranker-v2-m3
→ parent/relationship expansion + structured computation
→ Qwen2.5-7B-Instruct (JSON answer + source IDs)
→ guardrails + Gradio answer with citations, trace và evaluation
```

## Chạy trên Kaggle

1. Upload [`notebooks/kaggle_full_pipeline.ipynb`](notebooks/kaggle_full_pipeline.ipynb) lên Kaggle.
2. Chọn GPU **T4 x2** và bật Internet. Notebook pin PaddleOCR-VL/Qwen vào GPU 0, BGE/reranker
   vào GPU 1; nếu Kaggle chỉ cấp một GPU thì tự fallback về GPU 0. Không dùng P100:
   PaddleOCR-VL cần compute capability ≥ 7.0.
3. Chạy lần lượt các cell. PaddleOCR-VL được cài vào venv riêng (`/tmp/paddle_env`) với
   `paddlepaddle-gpu` 3.x từ index chính thức của Paddle và được gọi qua subprocess, nên không
   xung đột CUDA/cuDNN với PyTorch.
4. Cell **Smoke test** chạy toàn bộ pipeline trên tài liệu mẫu và in PASS/WARN/FAIL cho từng stage.
   Nếu có FAIL, xem `detail` của bước đó (log OCR worker ở `rag_smoke/logs/ocr_worker.log` khi
   chạy `run_smoke_test(config, keep=True)`).
5. Trong Gradio, upload tài liệu ở tab `Ingestion`, sau đó hỏi ở tab `Chat`.
6. Tải `rag_artifacts.zip` trước khi Kaggle session kết thúc. Lần sau có thể khôi phục bằng
   `pipeline.restore_artifacts(...)` hoặc tab `System`.

Notebook tự clone repository này. Vì vậy cần push phiên bản source code mới nhất lên
GitHub trước khi chạy notebook đã upload độc lập.

## Chạy trực tiếp trong notebook đã clone repo

```python
from pathlib import Path

from rag_kaggle import PipelineConfig, RAGPipeline

config = PipelineConfig.from_yaml("configs/baseline.yaml")
config.work_dir = Path("/kaggle/working/rag_runtime")
pipeline = RAGPipeline(config)

report = pipeline.ingest(["/kaggle/input/my-documents/report.pdf"])  # reset=True để làm lại index
result = pipeline.ask("Quy trình phê duyệt gồm những bước nào?")
print(result["answer"], result["citations"], result["warnings"])

metrics = pipeline.evaluate("evaluation/dataset.sample.jsonl")
```

## Cấu trúc chính

```text
rag_kaggle/
├── config.py           # Cấu hình dataclass, nạp từ YAML (configs/baseline.yaml)
├── models.py           # Canonical Document Model: Document, Block, Relationship, Chunk
├── ingestion.py        # Validate file (định dạng, kích thước, mật khẩu, macro), LibreOffice
├── parsers.py          # PDF (heading theo font, bảng, scan), DOCX, Excel (KPI, chart, merged cells)
├── paddleocr_vl.py     # PaddleOCR-VL adapter (in-process hoặc worker ở venv riêng)
├── vision.py           # Qwen2.5-VL cho flowchart/chart (JSON, retry, needs_review)
├── relationships.py    # Document graph (§5.3)
├── chunking.py         # Parent–Child chunking theo modality
├── storage.py          # SQLite, Qdrant (named vector, filter), BM25, embedding cache
├── retrieval.py        # RRF, reranker, parent + relationship expansion, citation
├── computation.py      # Tổng/trung bình/min/max/đếm trên bảng gốc bằng Python
├── generation.py       # Query rewrite có cấu trúc và Qwen answer dạng JSON
├── guardrails.py       # Prompt injection, PII masking, confidence, kiểm tra số liệu
├── tracing.py          # trace_id + JSONL log theo stage
├── evaluation.py       # Recall@k, MRR, correctness, citation/refusal accuracy
├── pipeline.py         # End-to-end orchestration, manifest, export/restore artifacts
├── smoke.py            # Smoke test toàn pipeline trên GPU Kaggle với tài liệu mẫu
└── ui.py               # Gradio: Ingestion, Chat, Evaluation, System
```

Thiết kế chi tiết nằm trong [`Architecture.md`](Architecture.md).

## Khác biệt có chủ đích so với Architecture.md

- **Docling chưa được dùng**: parser hiện dùng PyMuPDF/python-docx/openpyxl trực tiếp, cộng
  PaddleOCR-VL cho trang scan. Docling kéo theo nhiều dependency nặng dễ xung đột với
  PaddlePaddle trên Kaggle; có thể thêm sau như một parser bổ sung.
- **Dense model mặc định là `BAAI/bge-m3`** (fallback trong kiến trúc) vì `bge-multilingual-gemma2`
  không vừa GPU T4 khi chạy cùng Qwen 7B.
- **Cấu trúc package phẳng** `rag_kaggle/` thay vì chia nhiều package như §17, để notebook Kaggle
  import đơn giản.
- **BM25 là index local** (`bm25.pkl`), được §8.2 cho phép thay cho sparse vector trong Qdrant.
- **Recalculation công thức Excel** chưa tự động; số cell công thức thiếu cached value được ghi
  cảnh báo trong ingestion report.

## Test

```bash
pip install pymupdf python-docx openpyxl pandas pyarrow pillow rank-bm25 qdrant-client numpy pyyaml
python -m unittest discover -s tests
```

Test end-to-end dùng encoder giả lập và LLM giả lập nên không cần GPU.

## T4 x2

Notebook không tensor-parallel Qwen 7B qua hai T4 vì model 4-bit đã vừa một T4 và
cross-GPU transfer có thể làm chat chậm hơn. Thay vào đó, `configure_kaggle_devices`
phân vai cố định:

```text
GPU 0: PaddleOCR-VL worker -> Qwen answer -> Qwen-VL optional
GPU 1: BGE-M3 dense embedding + bge-reranker-v2-m3
```

Nhờ vậy Qwen có thể giữ warm trên GPU 0 trong khi query embedding và rerank dùng
GPU 1. Cell cấu hình trong notebook tự kiểm tra số GPU và áp dụng layout này.
