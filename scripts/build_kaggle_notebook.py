import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source, label=None):
    label = label or "code cell"
    indented_source = textwrap.indent(source.rstrip(), "    ")
    timed_source = f'''from datetime import datetime as _CellDateTime
import time as _cell_time

_cell_label = {label!r}
_cell_started_at = _CellDateTime.now().astimezone()
_cell_started_perf = _cell_time.perf_counter()
print(f"[CELL START] {{_cell_label}} | {{_cell_started_at.isoformat(timespec='seconds')}}")
try:
    pass
{indented_source}
finally:
    _cell_finished_at = _CellDateTime.now().astimezone()
    _cell_elapsed_seconds = _cell_time.perf_counter() - _cell_started_perf
    print(f"[CELL END] {{_cell_label}} | {{_cell_finished_at.isoformat(timespec='seconds')}}")
    print(f"[CELL ELAPSED_SECONDS] {{_cell_label}} | {{_cell_elapsed_seconds:.3f}}")
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": timed_source.splitlines(keepends=True),
    }


BOOTSTRAP = """import os
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
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
print("Project:", PROJECT_DIR)
"""


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


ingestion_cells = [
    markdown(
        """# 01 - Build and freeze a RAG corpus

Notebook này chỉ làm **Ingestion**. Nó nhận file, folder Kaggle Dataset hoặc ZIP; sau khi hoàn tất sẽ
đóng băng kết quả thành `corpus_bundle.zip`. Notebook không load answer model hay mở chat.
"""
    ),
    markdown("## 1. Cài đặt và import source"),
    code(BOOTSTRAP, "01_ingestion / 1. Cài đặt và import source"),
    markdown("## 2. Chọn model trực tiếp và khai báo input"),
    code(
        """# Mỗi phần nhận trực tiếp Hugging Face model ID hoặc tên PaddleOCR.
OCR_MODEL = "PaddleOCR-VL-1.6"
VISION_MODEL = None  # Ví dụ: "Qwen/Qwen2.5-VL-3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = None
EMBEDDING_QUERY_INSTRUCTION = None

# Liệt kê tên các dataset theo đường dẫn thực tế được Kaggle mount.
INPUT_SOURCES = [
    "/kaggle/input/datasets/huythinhuet/raw-data-small",
]
# Chuẩn hóa ngay tại đây để các cell phía sau có thể gọi .exists(), .name, v.v.
INPUT_SOURCES = [Path(source) for source in INPUT_SOURCES]

# Có thể thêm đường dẫn file/folder cụ thể nếu cần:
# INPUT_SOURCES.append(Path("/kaggle/input/datasets/owner/another-dataset/report.pdf"))

WORK_DIR = Path("/kaggle/working/rag_ingestion")
OUTPUT_BUNDLE = Path("/kaggle/working/corpus_bundle.zip")
INCLUDE_SOURCE_DOCUMENTS = False
""",
        "01_ingestion / 2. Chọn model và khai báo input",
    ),
    markdown("## 3. Cài PaddleOCR-VL trong môi trường riêng"),
    code(
        """import shutil

INSTALL_OCR = OCR_MODEL is not None
PADDLE_VERSION = "3.2.1"
PADDLE_CUDA = "cu126"
OCR_ENV = Path("/tmp/paddle_env")
OCR_PYTHON = None


def run(command):
    print("$", " ".join(map(str, command)))
    completed = subprocess.run(list(map(str, command)), capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        print(completed.stderr[-4000:])
        raise RuntimeError(f"Command failed ({completed.returncode}): {command[0]}")
    return completed.stdout


def python_works(path):
    return path.exists() and subprocess.run(
        [str(path), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
    ).returncode == 0


if INSTALL_OCR:
    try:
        python = OCR_ENV / "bin" / "python"
        if not python_works(python):
            shutil.rmtree(OCR_ENV, ignore_errors=True)
            try:
                # Một số Kaggle image không có ensurepip đầy đủ.
                run([sys.executable, "-m", "venv", str(OCR_ENV)])
            except Exception:
                shutil.rmtree(OCR_ENV, ignore_errors=True)
                print("python -m venv failed; retrying with virtualenv...")
                run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])
                run([sys.executable, "-m", "virtualenv", str(OCR_ENV)])
        if not python_works(python):
            raise RuntimeError(f"OCR environment was created but Python is unusable: {python}")
        pip = [str(python), "-m", "pip", "install", "-q"]
        run(pip + ["--upgrade", "pip"])
        run(pip + [
            f"paddlepaddle-gpu=={PADDLE_VERSION}",
            "-i", f"https://www.paddlepaddle.org.cn/packages/stable/{PADDLE_CUDA}/",
            "--extra-index-url", "https://pypi.org/simple",
        ])
        run(pip + ["paddleocr[doc-parser]"])
        print(run([
            python,
            "-c",
            "import paddle, paddleocr; "
            "print('paddle', paddle.__version__, 'cuda', paddle.device.is_compiled_with_cuda()); "
            "print('paddleocr', paddleocr.__version__)",
        ]))
        OCR_PYTHON = str(python)
    except Exception as exc:
        print("PaddleOCR-VL setup failed; OCR will be disabled:", exc)

print("OCR_PYTHON =", OCR_PYTHON)
""",
        "01_ingestion / 3. Cài PaddleOCR-VL",
    ),
    markdown("## 4. Tạo ingestion runtime và kiểm tra tài nguyên"),
    code(
        """from rag_kaggle import (
    IngestionPipeline, PipelineConfig, configure_ingestion_devices,
    inspect_resources, suggest_model_upgrades,
)

config = PipelineConfig.from_yaml(PROJECT_DIR / "configs" / "baseline.yaml")
config.work_dir = WORK_DIR
config.parsing.ocr_model_name = OCR_MODEL or "PaddleOCR-VL-1.6"
config.parsing.ocr_python = OCR_PYTHON
config.parsing.enable_ocr = OCR_MODEL is not None and OCR_PYTHON is not None
config.vision.enabled = VISION_MODEL is not None
if VISION_MODEL:
    config.vision.model = VISION_MODEL
config.retrieval.dense_model = EMBEDDING_MODEL
config.retrieval.dense_fallback_model = EMBEDDING_MODEL  # Không âm thầm đổi sang model khác.
config.retrieval.dense_revision = EMBEDDING_REVISION
config.retrieval.dense_query_instruction = EMBEDDING_QUERY_INSTRUCTION
config.artifacts.include_source_documents = INCLUDE_SOURCE_DOCUMENTS

resources = inspect_resources()
print("Resources:", resources)
print("Allocation:", configure_ingestion_devices(config))
print("Suggestions:", suggest_model_upgrades("ingestion", resources, config))
pipeline = IngestionPipeline(config)
""",
        "01_ingestion / 4. Tạo ingestion runtime",
    ),
    markdown("## 5. Ingest, freeze và export"),
    code(
        """from rag_kaggle.ingestion import discover_input_files

print("Configured input sources:")
for source in INPUT_SOURCES:
    source_path = Path(source)
    print(" -", source_path, "exists=" + str(source_path.exists()))

files, discovery = discover_input_files(INPUT_SOURCES, config)
print(f"Found {len(files)} supported document(s):")
for path in files:
    print(" -", path)
if discovery["skipped"]:
    print("Skipped/not found:")
    for item in discovery["skipped"][:30]:
        print(" -", item)
if not files:
    kaggle_input_root = Path("/kaggle/input/datasets")
    mounted = sorted(str(path) for path in kaggle_input_root.glob("*/*")) if kaggle_input_root.exists() else []
    raise ValueError(
        f"Không tìm thấy tài liệu hỗ trợ. Dataset đang được mount: {mounted}. "
        f"Hãy sửa INPUT_SOURCES theo đúng đường dẫn hiển thị trên Kaggle."
    )

report = pipeline.ingest(files, reset=True, progress=print)
report["input_discovery"] = discovery
print("Ingestion stats:", report["stats"])
if not report["ok"]:
    raise RuntimeError(report["errors"])

bundle = pipeline.export_corpus_bundle(OUTPUT_BUNDLE)
print("Frozen corpus:", pipeline.validate_corpus())
print("Bundle:", bundle, f"({bundle.stat().st_size / 1024**2:.1f} MB)")

from IPython.display import FileLink, display
display(FileLink(str(bundle)))
""",
        "01_ingestion / 5. Ingest, freeze và export",
    ),
]


retrieval_cells = [
    markdown(
        """# 02 - Retrieve and answer from a frozen corpus

Notebook này chỉ chạy **Retrieve & Answer**. Nó không có parser/OCR/VLM và không thể ingest thêm tài liệu.
Embedding mặc định được đọc từ manifest của corpus để bảo đảm document/query dùng cùng model.
"""
    ),
    markdown("## 1. Cài đặt và import source"),
    code(BOOTSTRAP, "02_retrieve_answer / 1. Cài đặt và import source"),
    markdown("## 2. Chọn model trực tiếp"),
    code(
        """# None = bắt buộc kế thừa embedding model/revision/query instruction từ corpus manifest.
EMBEDDING_MODEL = None
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # None để tắt reranker
GENERATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GENERATION_REVISION = None

WORK_DIR = Path("/kaggle/working/rag_corpus")
SESSION_DIR = Path("/kaggle/working/rag_session")

# Đường dẫn từ Kaggle Dataset. Để None nếu muốn upload bundle trong Gradio.
CORPUS_BUNDLE = None  # Ví dụ: "/kaggle/input/my-rag-corpus/corpus_bundle.zip"
""",
        "02_retrieve_answer / 2. Chọn model",
    ),
    markdown("## 3. Tạo retrieval runtime và khôi phục corpus"),
    code(
        """from rag_kaggle import (
    PipelineConfig, RetrievalAnswerPipeline, configure_retrieval_devices,
    inspect_resources, suggest_model_upgrades,
)

config = PipelineConfig.from_yaml(PROJECT_DIR / "configs" / "baseline.yaml")
config.work_dir = WORK_DIR
config.runtime.session_dir = str(SESSION_DIR)
config.retrieval.dense_model = EMBEDDING_MODEL
config.retrieval.reranker_enabled = RERANKER_MODEL is not None
if RERANKER_MODEL:
    config.retrieval.reranker_model = RERANKER_MODEL
config.generation.model = GENERATION_MODEL
config.generation.revision = GENERATION_REVISION

resources = inspect_resources()
print("Resources:", resources)
print("Allocation:", configure_retrieval_devices(config))
print("Suggestions:", suggest_model_upgrades("retrieval", resources, config))
pipeline = RetrievalAnswerPipeline(config)

if CORPUS_BUNDLE:
    restored = pipeline.restore_corpus_bundle(CORPUS_BUNDLE)
    print("Corpus:", restored["manifest"])
""",
        "02_retrieve_answer / 3. Tạo retrieval runtime và khôi phục corpus",
    ),
    markdown("## 4. Mở Chat UI"),
    code(
        """from rag_kaggle.ui import launch_chat_demo

# Tab System cho phép upload corpus_bundle.zip từ máy cá nhân nếu CORPUS_BUNDLE=None.
launch_chat_demo(pipeline, share=True, debug=False)
""",
        "02_retrieve_answer / 4. Mở Chat UI",
    ),
    markdown("## 5. API Python trực tiếp"),
    code(
        """# result = pipeline.ask("Phí thường niên của thẻ Visa Gold là bao nhiêu?")
# print(result["answer"], result["citations"])
# metrics = pipeline.evaluate("/kaggle/input/my-evaluation/dataset.jsonl")
""",
        "02_retrieve_answer / 5. API Python trực tiếp",
    ),
]


NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
outputs = {
    NOTEBOOK_DIR / "01_ingestion.ipynb": notebook(ingestion_cells),
    NOTEBOOK_DIR / "02_retrieve_answer.ipynb": notebook(retrieval_cells),
}
for path, payload in outputs.items():
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path}")
