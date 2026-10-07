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

1. Chọn Accelerator là **GPU T4 x2**. Notebook tự pin PaddleOCR-VL/Qwen vào GPU 0 và
   embedding/reranker vào GPU 1. **Không dùng P100**: PaddleOCR-VL cần GPU
   compute capability ≥ 7.0, P100 chỉ có 6.0.
2. Bật **Internet** để tải dependency và model lần đầu.
3. Chạy lần lượt các cell. Cell **Smoke test** kiểm tra toàn bộ pipeline trên tài liệu mẫu trước khi
   mở giao diện; nếu có bước FAIL, gửi lại output của cell đó để sửa.

Preset mặc định dùng `BAAI/bge-m3` vì `bge-multilingual-gemma2` quá nặng khi chạy cùng Qwen 7B trên T4.
LibreOffice (tuỳ chọn) giúp đọc `.xls` đầy đủ hơn: `!apt-get install -y libreoffice-calc`.
"""
    ),
    markdown("## 1. Clone hoặc cập nhật source code"),
    code(
        """import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/NhoDung/RAG-basic.git"
REPO_DIR = Path("/kaggle/working/RAG-basic")

if not (Path.cwd() / "rag_kaggle").exists():
    if REPO_DIR.exists():
        subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    else:
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

PROJECT_DIR = Path.cwd()
sys.path.insert(0, str(PROJECT_DIR))
print("Project:", PROJECT_DIR)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
"""
    ),
    markdown("## 2. Cài dependencies cho pipeline chính (PyTorch)"),
    code(
        """import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
print("Main dependencies installed.")
"""
    ),
    markdown(
        """## 3. Cài PaddleOCR-VL trong môi trường riêng

PaddlePaddle và PyTorch dùng các bản CUDA/cuDNN khác nhau; tài liệu PaddleOCR-VL khuyên tách môi trường.
Cell này tạo venv riêng ở `/tmp/paddle_env` (không tính vào dung lượng output), cài
`paddlepaddle-gpu` 3.x từ index chính thức của Paddle (bản trên PyPI chỉ đến 2.6, không chạy được
PaddleOCR-VL) và `paddleocr[doc-parser]`. Pipeline gọi OCR qua subprocess, nên PyTorch không bị ảnh hưởng
và VRAM của OCR được giải phóng hoàn toàn sau khi parse.

Nếu cài thất bại, pipeline vẫn chạy với OCR tắt (PDF có text, DOCX, Excel vẫn dùng được).
"""
    ),
    code(
        """import subprocess
import sys
from pathlib import Path

INSTALL_OCR = True
PADDLE_VERSION = "3.2.1"   # theo tài liệu PaddleOCR-VL
PADDLE_CUDA = "cu126"      # đổi sang "cu118" nếu driver GPU của session quá cũ
OCR_ENV = Path("/tmp/paddle_env")
OCR_PYTHON = None


def run(command):
    print("$", " ".join(map(str, command)))
    completed = subprocess.run(list(map(str, command)), capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout[-2000:], completed.stderr[-3000:])
        raise RuntimeError(f"Command failed: {command[0]} ... (exit {completed.returncode})")
    return completed.stdout


if INSTALL_OCR:
    try:
        python = OCR_ENV / "bin" / "python"
        if not python.exists():
            try:
                run([sys.executable, "-m", "venv", OCR_ENV])
            except RuntimeError:
                run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])
                run([sys.executable, "-m", "virtualenv", OCR_ENV])
        pip = [python, "-m", "pip", "install", "-q"]
        run(pip + ["--upgrade", "pip"])
        run(pip + [
            f"paddlepaddle-gpu=={PADDLE_VERSION}",
            "-i", f"https://www.paddlepaddle.org.cn/packages/stable/{PADDLE_CUDA}/",
            "--extra-index-url", "https://pypi.org/simple",
        ])
        run(pip + ["paddleocr[doc-parser]"])
        check = run([python, "-c",
                     "import paddle, paddleocr; print('paddle', paddle.__version__, 'cuda', "
                     "paddle.device.is_compiled_with_cuda(), 'gpus', paddle.device.cuda.device_count()); "
                     "print('paddleocr', paddleocr.__version__)"])
        print(check)
        OCR_PYTHON = str(python)
    except Exception as exc:
        print("PaddleOCR-VL setup failed; OCR will be disabled:", exc)

print("OCR_PYTHON =", OCR_PYTHON)
"""
    ),
    markdown("## 4. Kiểm tra GPU và import pipeline"),
    code(
        """import torch

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}, compute capability {props.major}.{props.minor}, "
              f"{props.total_memory / 1024**3:.1f} GB")
    if torch.cuda.get_device_properties(0).major < 7:
        print("⚠️ GPU này không hỗ trợ PaddleOCR-VL (cần compute capability ≥ 7.0). Hãy chọn T4.")
else:
    print("⚠️ Không thấy GPU. Vào Settings > Accelerator và chọn GPU T4.")

from rag_kaggle import PipelineConfig, RAGPipeline, configure_kaggle_devices
"""
    ),
    markdown("## 5. Cấu hình baseline chạy được trên T4"),
    code(
        """from pathlib import Path

# Preset §16 Architecture.md; chỉnh trực tiếp trên object nếu cần.
config = PipelineConfig.from_yaml(PROJECT_DIR / "configs" / "baseline.yaml")
config.work_dir = Path("/kaggle/working/rag_runtime")

# OCR/document vision chạy trong venv riêng (cell 3).
config.parsing.ocr_python = OCR_PYTHON
config.parsing.enable_ocr = OCR_PYTHON is not None
config.parsing.ocr_model_dir = None  # Đặt path Kaggle Dataset nếu chạy offline.

# Kaggle hiện thường cấp T4 x2. Phân vai GPU thay vì shard Qwen 7B qua 2 GPU:
# GPU 0: PaddleOCR-VL worker, Qwen answer và Qwen-VL optional.
# GPU 1: BGE dense embedding và bge reranker.
# Nếu chỉ có một GPU, helper tự fallback toàn bộ về cuda:0.
device_layout = configure_kaggle_devices(config)
print("Device layout:", device_layout)

# Dense retrieval: bge-m3 cho T4. Nếu đổi sang BAAI/bge-multilingual-gemma2 thì phải
# re-ingest với reset=True (không trộn 2 dense model trong một index) và đặt
# config.retrieval.dense_query_instruction theo model card.
config.retrieval.dense_model = "BAAI/bge-m3"

# VLM Qwen2.5-VL-3B cho flowchart/chart phức tạp: tắt mặc định để tiết kiệm GPU.
config.vision.enabled = False

# Có thể tắt reranker khi muốn test nhanh hoặc thiếu VRAM.
config.retrieval.reranker_enabled = True
config.generation.query_rewrite_enabled = False

print("OCR enabled:", config.parsing.enable_ocr)
print("Dense device:", config.retrieval.dense_device)
print("Reranker device:", config.retrieval.reranker_device)
print("Qwen device:", config.generation.device or "auto")

try:
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout)
except Exception as exc:
    print("Không đọc được nvidia-smi:", exc)
"""
    ),
    markdown(
        """## 6. Smoke test toàn pipeline

Chạy mọi stage trên tài liệu mẫu nhỏ trong thư mục riêng (`/kaggle/working/rag_smoke`, tự xoá sau khi
xong): môi trường → OCR → ingest → index → retrieve/rerank → Qwen → structured computation.
Lần đầu sẽ tải toàn bộ model (Qwen 7B khoảng 15 GB) nên có thể mất 10–20 phút.

- `PASS`: stage chạy đúng. `WARN`: chạy được nhưng kết quả cần xem lại. `FAIL`: lỗi, xem `detail`.
- Các model được unload sau smoke test, nên không chiếm GPU của pipeline chính.
"""
    ),
    code(
        """from rag_kaggle.smoke import run_smoke_test

smoke = run_smoke_test(config)
for step in smoke["steps"]:
    if step["status"] != "PASS":
        print(f"\\n--- {step['step']} ({step['status']}) ---\\n{step['detail']}")
"""
    ),
    markdown("## 7. Khởi tạo pipeline chính"),
    code(
        """pipeline = RAGPipeline(config)
print("Runtime:", config.work_dir)
print("DB stats:", pipeline.metadata.stats())
"""
    ),
    markdown(
        """## 8. (Tuỳ chọn) Khôi phục index đã export

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
        """## 9. Chạy Gradio

- **Ingestion**: chọn PDF/DOCX/XLSX là hệ thống tự parse và lập chỉ mục đúng một lần. Nhấn lại với cùng file sẽ bị từ chối; các file khác nội dung nhưng trùng tên được giữ thành version riêng.
- **Tiến độ**: xem log theo trang PDF, block DOCX, sheet Excel, rồi embedding/Qdrant/BM25. Chỉ một ingestion được phép chạy tại một thời điểm.
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
    markdown("## 10. API Python trực tiếp, evaluation và export artifacts"),
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
