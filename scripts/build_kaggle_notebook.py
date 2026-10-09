import hashlib
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def cell_id(kind, source):
    """Deterministic nbformat 4.5 cell id, so regenerating does not churn the notebooks."""
    return hashlib.sha1(f"{kind}\n{source}".encode("utf-8")).hexdigest()[:8]


def markdown(source):
    return {
        "cell_type": "markdown",
        "id": cell_id("markdown", source),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


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
        "id": cell_id("code", timed_source),
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

Luồng xử lý (chi tiết: `docs/data-flow.md`):

1. Parse + PaddleOCR-VL (GPU 1). Text OCR được LLM sửa chính tả song song trên GPU 0, mỗi lần chỉ nhìn
   system + skill + đoạn liền trước đã sửa + đoạn hiện tại.
2. VLM (tuỳ chọn, chart/flowchart) chạy sau khi sửa OCR, nhận OCR đã sửa + đúng 1 trang liền trước.
3. Hồ sơ tài liệu: tiêu đề, số hiệu, ngày, người ký, keyword, thực thể.
4. Chunking theo cấu trúc: 1 chunk = nhóm đoạn / 1 bảng / 1 ảnh; bảng lớn = preview + file `.xlsx`.
5. Embedding + Qdrant + BM25, rồi đóng băng corpus.
"""
    ),
    markdown("## 1. Cài đặt và import source"),
    code(BOOTSTRAP, "01_ingestion / 1. Cài đặt và import source"),
    markdown("## 2. Chọn model trực tiếp và khai báo input"),
    code(
        """# Mỗi phần nhận trực tiếp Hugging Face model ID hoặc tên PaddleOCR.
OCR_MODEL = "PaddleOCR-VL-1.6"
OCR_CORRECTION_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # None disables OCR spelling correction.
# Glossary tên riêng xuyên suốt document cho bước sửa OCR; bật sau khi đã benchmark với model sửa lỗi.
OCR_CORRECTION_GLOSSARY = False
OCR_CORRECTION_PREVIOUS_CHARS = 600  # đuôi đoạn liền trước (đã sửa) đưa vào prompt
OCR_CORRECTION_SEGMENT_CHARS = 1200  # đoạn dài được sửa theo từng khúc <= giá trị này
OCR_CORRECTION_TABLES = False  # True: sửa bảng theo từng ô, giữ nguyên số hàng/ô
# VLM chạy sau khi sửa OCR của cả document (nhận OCR đã sửa + 1 trang liền trước), nên có thể bật cùng
# OCR_CORRECTION_MODEL. Chỉ áp dụng cho chart/flowchart.
VISION_MODEL = None  # Ví dụ: "Qwen/Qwen2.5-VL-3B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = None
EMBEDDING_QUERY_INSTRUCTION = None

# Chunking theo cấu trúc: bảng nhỏ giữ nguyên dạng markdown; bảng lớn chỉ index preview
# và lưu đầy đủ trong file .xlsx (đi kèm corpus_bundle.zip).
TEXT_CHUNK_CHARS = 1800
TABLE_INLINE_MAX_CHARS = 6000
TABLE_PREVIEW_ROWS = 5
TABLE_PREVIEW_COLUMNS = 5

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
config.ocr_correction.enabled = OCR_CORRECTION_MODEL is not None
if OCR_CORRECTION_MODEL:
    config.ocr_correction.model = OCR_CORRECTION_MODEL
config.ocr_correction.glossary_enabled = OCR_CORRECTION_GLOSSARY
config.ocr_correction.previous_context_chars = OCR_CORRECTION_PREVIOUS_CHARS
config.ocr_correction.segment_chars = OCR_CORRECTION_SEGMENT_CHARS
config.ocr_correction.correct_tables = OCR_CORRECTION_TABLES
config.vision.enabled = VISION_MODEL is not None
if VISION_MODEL:
    config.vision.model = VISION_MODEL
config.chunking.text_chunk_chars = TEXT_CHUNK_CHARS
config.chunking.table_inline_max_chars = TABLE_INLINE_MAX_CHARS
config.chunking.table_preview_rows = TABLE_PREVIEW_ROWS
config.chunking.table_preview_columns = TABLE_PREVIEW_COLUMNS
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
for item in report["documents"]:
    profile = item.get("profile") or {}
    print(
        f" - {item['file']}: {item['chunks']} chunks, {item['entities']} entities | "
        f"title={profile.get('title')!r} | số hiệu={profile.get('doc_number')} | "
        f"ngày={profile.get('issue_date')} | người ký={profile.get('signers')}"
    )
    print("   keywords:", (profile.get("keywords") or [])[:8])
    correction = item.get("ocr_correction")
    if correction:
        print(
            f"   OCR correction: {correction.get('success_count', 0)} ok, "
            f"{correction.get('fallback_count', 0)} fallback, {correction.get('skipped_count', 0)} skipped, "
            f"{correction.get('segment_count', 0)} segments, buffer peak {correction.get('buffer_peak_size', 0)}"
        )
        if correction.get("name_variants"):
            print("   name variants seen:", correction["name_variants"])

from collections import Counter

chunks = pipeline.metadata.list_chunks()
print("Chunks by type:", dict(Counter(chunk.chunk_type for chunk in chunks)))
truncated = [chunk for chunk in chunks if chunk.metadata.get("table_truncated")]
tables = sorted((config.work_dir / "tables").rglob("*.xlsx"))
print(f"Large tables indexed as preview: {len(truncated)} | .xlsx files written: {len(tables)}")
for chunk in truncated[:5]:
    print(" -", chunk.source_file, chunk.metadata["table_rows"], "rows ->", chunk.metadata["table_file"])
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

Truy xuất: dense + BM25 → RRF (thực thể trong câu hỏi khớp hồ sơ tài liệu sẽ được boost mềm) → reranker →
mở rộng chunk lân cận cùng mục; bảng lớn được nạp đủ hàng và có file `.xlsx` để tải trong Chat UI.
"""
    ),
    markdown("## 1. Cài đặt và import source"),
    code(BOOTSTRAP, "02_retrieve_answer / 1. Cài đặt và import source"),
    markdown("## 2. Chọn model trực tiếp"),
    code(
        """import shutil

# None = bắt buộc kế thừa embedding model/revision/query instruction từ corpus manifest.
EMBEDDING_MODEL = None
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # None để tắt reranker
GENERATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GENERATION_REVISION = None

# Mở rộng context sau khi tìm thấy chunk.
NEIGHBOR_CHUNKS = 1  # số chunk text kề nhau cùng mục được thêm vào hai phía
MAX_TABLE_CHARS_IN_CONTEXT = 12000  # bảng lớn (chỉ index preview) được nạp lại tới ngưỡng này
ENTITY_BOOST = True  # boost tài liệu có nhắc thực thể xuất hiện trong câu hỏi

WORK_DIR = Path("/kaggle/working/rag_corpus")
SESSION_DIR = Path("/kaggle/working/rag_session")

# Thư mục corpus trong Kaggle Dataset được nén thành bundle để restore.
CORPUS_SRC = "/kaggle/input/datasets/nhodng/data-test"
CORPUS_BUNDLE = shutil.make_archive("/kaggle/working/corpus_bundle", "zip", root_dir=CORPUS_SRC)
print("Bundle:", CORPUS_BUNDLE)
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
config.retrieval.neighbor_chunks = NEIGHBOR_CHUNKS
config.retrieval.max_table_chars_in_context = MAX_TABLE_CHARS_IN_CONTEXT
config.retrieval.entity_boost = ENTITY_BOOST

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

# Tab System cho phép upload corpus_bundle.zip từ máy cá nhân nếu cần.
launch_chat_demo(pipeline, share=True, debug=False)
""",
        "02_retrieve_answer / 4. Mở Chat UI",
    ),
    markdown("## 5. API Python trực tiếp"),
    code(
        """# result = pipeline.ask("Phí thường niên của thẻ Visa Gold là bao nhiêu?")
# print(result["answer"], result["citations"])
# print(result["query_entities"])  # thực thể trong câu hỏi khớp với hồ sơ tài liệu
# print([c["table_file"] for c in result["contexts"] if c.get("table_file")])  # bảng đầy đủ (.xlsx)
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
