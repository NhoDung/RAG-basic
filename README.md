# Multimodal RAG trên Kaggle

Pipeline được tách thành hai session độc lập:

```text
01_ingestion.ipynb
PDF / DOCX / Excel / ZIP / folder
-> parse + OCR/VLM -> parent-child chunking -> embedding
-> Qdrant + BM25 + metadata/assets
-> frozen corpus_bundle.zip

02_retrieve_answer.ipynb
corpus_bundle.zip -> validate manifest/checksum
-> Qdrant + BM25 -> RRF -> reranker -> Qwen -> answer + citation
```

Retrieve & Answer không chứa parser/OCR/VLM, không có API ingest và không thêm tài liệu vào corpus.

## 1. Chạy Ingestion

Upload [01_ingestion.ipynb](notebooks/01_ingestion.ipynb) lên Kaggle, chọn GPU T4 x2 và sửa cell cấu hình đầu:

```python
from pathlib import Path

OCR_MODEL = "PaddleOCR-VL-1.6"
OCR_CORRECTION_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # None to disable correction
VISION_MODEL = None
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = None

INPUT_DATASET_SLUGS = ["raw-data"]  # tên trong panel Input
INPUT_SOURCES = [Path("/kaggle/input") / slug for slug in INPUT_DATASET_SLUGS]
```

Input có thể là folder Kaggle Dataset, ZIP hoặc file đơn. Các định dạng được nhận gồm `.pdf`, `.docx`,
`.xlsx`, `.xls`, `.xlsm`, `.xltx`, `.xltm`. Dùng folder Kaggle Dataset nhanh hơn ZIP. File `.xls` cần
LibreOffice để chuyển đổi.

Notebook tạo `/kaggle/working/corpus_bundle.zip`; tải file này về máy hoặc lưu thành Kaggle Dataset.
Nếu muốn upload trực tiếp từ máy thay vì dùng Kaggle Dataset, có thể mở
`launch_ingestion_demo(pipeline)` và tải file/ZIP trong Gradio.

## 2. Chạy Retrieve & Answer

Upload [02_retrieve_answer.ipynb](notebooks/02_retrieve_answer.ipynb) lên một session khác:

```python
# None: đọc embedding model bắt buộc từ corpus manifest.
EMBEDDING_MODEL = None
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GENERATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"

CORPUS_BUNDLE = "/kaggle/input/my-rag-corpus/corpus_bundle.zip"
```

Có thể để `CORPUS_BUNDLE = None` rồi upload bundle từ máy trong tab `System` của Gradio.

Embedding model, revision, dimension, normalization và query instruction là một phần của corpus. Nếu
khai báo `EMBEDDING_MODEL` khác model đã ingest, restore sẽ dừng ngay. Reranker và generation model có
thể đổi mà không cần ingest lại. Model ID vẫn phải tương thích với adapter tương ứng
(`SentenceTransformer`, `FlagReranker`, causal LM hoặc vision-language model).

## 3. Corpus bundle

Bundle chứa dữ liệu cần để phục hồi đầy đủ retrieval và citation:

```text
corpus_bundle.zip
├── corpus_manifest.json
├── checksums.json
├── config.json
├── qdrant/
├── metadata.db
├── bm25.pkl
├── tables/
├── assets/
└── parsed/
```

Qdrant là vector store chính nhưng không thay thế metadata, bảng gốc và assets dùng để mở rộng parent,
tính toán và render citation. Source PDF/DOCX/Excel không được đưa vào bundle mặc định; bật
`config.artifacts.include_source_documents` nếu cần giữ bản gốc.

Restore luôn giải nén vào staging directory, kiểm tra path, schema và checksum trước khi thay corpus đang
hoạt động. Query trace được ghi ra session directory riêng, không ghi vào frozen corpus.

## 4. Python API

### Ingestion

```python
from pathlib import Path
from rag_kaggle import IngestionPipeline, PipelineConfig, configure_ingestion_devices

config = PipelineConfig.from_yaml("configs/baseline.yaml")
config.work_dir = Path("/kaggle/working/rag_ingestion")
config.retrieval.dense_model = "BAAI/bge-m3"
config.retrieval.dense_fallback_model = config.retrieval.dense_model
configure_ingestion_devices(config)

pipeline = IngestionPipeline(config)
report = pipeline.ingest_sources(["/kaggle/input/my-documents"], reset=True)
bundle = pipeline.export_corpus_bundle("/kaggle/working/corpus_bundle.zip")
```

### Retrieve & Answer

```python
from pathlib import Path
from rag_kaggle import PipelineConfig, RetrievalAnswerPipeline, configure_retrieval_devices

config = PipelineConfig.from_yaml("configs/baseline.yaml")
config.work_dir = Path("/kaggle/working/rag_corpus")
config.retrieval.dense_model = None
configure_retrieval_devices(config)

pipeline = RetrievalAnswerPipeline(config)
pipeline.restore_corpus_bundle("/kaggle/input/my-rag-corpus/corpus_bundle.zip")
result = pipeline.ask("Quy trình phê duyệt gồm những bước nào?")
print(result["answer"], result["citations"])
```

`RAGPipeline` vẫn tồn tại để tương thích với code cũ và test local, nhưng không được notebook mới sử dụng.

## 5. Phân bổ tài nguyên

T4 x2 trong Ingestion:

```text
GPU 0: OCR spelling-correction LLM during parsing, then dense embedding
GPU 1: PaddleOCR-VL worker
CPU: parser, chunker, Qdrant, BM25, metadata
```

OCR and correction overlap through a bounded ordered buffer. OCR may run ahead, while correction
always processes the next reading-order component and receives only its immediately preceding
corrected component as context. Both models unload before dense embedding starts.

T4 x2 trong Retrieve & Answer:

```text
GPU 0: generation/query rewrite
GPU 1: query embedding
CPU: reranker, Qdrant, BM25, context expansion
```

`inspect_resources()` báo GPU/VRAM, RAM và disk để đánh giá khả năng dùng model lớn hơn. Pipeline không
tự đổi model âm thầm.

## 6. Test

```bash
pip install pymupdf python-docx openpyxl pandas pyarrow pillow rank-bm25 qdrant-client numpy pyyaml
python -m unittest discover -s tests
```

Thiết kế chi tiết nằm trong [Architecture.md](Architecture.md).
