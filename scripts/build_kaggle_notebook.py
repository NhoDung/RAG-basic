import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_full_pipeline.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Multimodal RAG full pipeline trên Kaggle

Pipeline: **PDF/DOCX/Excel → PaddleOCR-VL-1.6 → Parent–Child → BGE → Qdrant + BM25 → RRF → Reranker → Qwen → Gradio**.

Trước khi chạy:

1. Chọn Accelerator là **GPU T4/P100**.
2. Bật **Internet** để tải dependency và model lần đầu.
3. Chạy lần lượt các cell. Cell cuối mở giao diện upload và chat.

Preset mặc định dùng `BAAI/bge-m3` vì `bge-multilingual-gemma2` quá nặng khi chạy cùng Qwen 7B trên T4.\nLibreOffice (tuỳ chọn) giúp đọc `.xls` đầy đủ hơn: `!apt-get install -y libreoffice-calc`.
"""
    ),
    markdown("## 1. Clone hoặc sử dụng source code hiện có"),
    code(
        """import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/NhoDung/RAG-basic.git"
REPO_DIR = Path("/kaggle/working/RAG-basic")

if not (Path.cwd() / "rag_kaggle").exists():
    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

PROJECT_DIR = Path.cwd()
sys.path.insert(0, str(PROJECT_DIR))
print("Project:", PROJECT_DIR)
"""
    ),
    markdown("## 2. Cài dependencies"),
    code(
        """import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"
])

# PaddleOCR-VL requires PaddlePaddle plus the document-parser extras.
# Override these specs in the cell if the official Paddle release notes for the
# current Kaggle CUDA image require a different wheel.
PADDLE_SPEC = "paddlepaddle-gpu"
PADDLEOCR_SPEC = "paddleocr[doc-parser]"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", PADDLE_SPEC])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", PADDLEOCR_SPEC])

print("Dependencies installed. If Paddle asks for a restart, use Run > Restart session once.")
"""
    ),
    markdown("## 3. Kiểm tra GPU và import pipeline"),
    code(
        """import torch
import paddle

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Torch GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Paddle:", paddle.__version__, "CUDA build:", paddle.device.is_compiled_with_cuda())

from rag_kaggle import PipelineConfig, RAGPipeline
"""
    ),
    markdown("## 4. Cấu hình baseline chạy được trên T4"),
    code(
        """from pathlib import Path

# Preset §16 Architecture.md; chỉnh trực tiếp trên object nếu cần.
config = PipelineConfig.from_yaml(PROJECT_DIR / "configs" / "baseline.yaml")
config.work_dir = Path("/kaggle/working/rag_runtime")

# OCR/document vision
config.parsing.ocr_model_dir = None  # Đặt path Kaggle Dataset nếu chạy offline.

# Dense retrieval: bge-m3 cho T4. Nếu đổi sang BAAI/bge-multilingual-gemma2 thì phải
# re-ingest với reset=True (không trộn 2 dense model trong một index) và đặt
# config.retrieval.dense_query_instruction theo model card.
config.retrieval.dense_model = "BAAI/bge-m3"

# VLM Qwen2.5-VL-3B cho flowchart/chart phức tạp: tắt mặc định để tiết kiệm GPU.
config.vision.enabled = False

# Có thể tắt reranker khi muốn test nhanh hoặc thiếu VRAM.
config.retrieval.reranker_enabled = True
config.generation.query_rewrite_enabled = False

pipeline = RAGPipeline(config)
print("Runtime:", config.work_dir)
print("Initial DB stats:", pipeline.metadata.stats())
"""
    ),
    markdown(
        """## 5. (Tuỳ chọn) Khôi phục index đã export

Nếu đã upload `rag_artifacts.zip` thành Kaggle Dataset, khôi phục để chat ngay mà không cần ingest lại.
"""
    ),
    code(
        """ARTIFACTS = None  # ví dụ: "/kaggle/input/my-rag-artifacts/rag_artifacts.zip"
if ARTIFACTS:
    print(pipeline.restore_artifacts(ARTIFACTS))
"""
    ),
    markdown(
        """## 6. Chạy Gradio

- **Ingestion**: upload PDF/DOCX/XLSX rồi bấm **Parse và lập chỉ mục**. Mặc định index được cộng dồn;
  tick "Xoá toàn bộ index cũ" để làm lại từ đầu. File không đổi sẽ được bỏ qua.
- **Chat**: hỏi đáp, xem citation, retrieved chunks, điểm dense/BM25/RRF/rerank, preview ảnh nguồn và trace.
- **Evaluation**: upload dataset JSONL (xem `evaluation/dataset.sample.jsonl`).
- **System**: thống kê, trạng thái từng stage, khôi phục artifacts.

Lần chạy đầu sẽ tải model nên mất thời gian. Không đóng session trong lúc model đang tải.
"""
    ),
    code(
        """from rag_kaggle.ui import launch_demo

launch_demo(pipeline, share=True, debug=False)
"""
    ),
    markdown("## 7. API Python trực tiếp, evaluation và export artifacts"),
    code(
        """# report = pipeline.ingest(["/kaggle/input/my-data/file.pdf"])          # cộng dồn
# report = pipeline.ingest([...], reset=True)                               # làm lại từ đầu
# result = pipeline.ask("Phí thường niên của thẻ là bao nhiêu?")
# print(result["answer"], result["citations"], result["warnings"])

# Evaluation (§14):
# metrics = pipeline.evaluate("/kaggle/input/my-eval/dataset.jsonl", run_answers=True)
# print(metrics["retrieval"], metrics["answer"])

# Export Qdrant, SQLite, parsed JSON/Parquet, assets, manifest trước khi session kết thúc:
# archive = pipeline.export_artifacts("/kaggle/working/rag_artifacts.zip")
# print(archive)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")
