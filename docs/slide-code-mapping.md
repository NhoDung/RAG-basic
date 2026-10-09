# Mapping thông tin trên slide ↔ code trong repo

Deck: [RAG Kaggle Pipeline](https://claude.ai/artifact/Tc9LHJ5hJkDAyoyhy3FJ9o) (23 slide). Đối chiếu ngày 09/10/2026 với working tree hiện tại.

Quy ước cột **Loại nguồn**:

- **Code**: logic trong `rag_kaggle/` hoặc notebook.
- **Config**: giá trị trong [configs/baseline.yaml](../configs/baseline.yaml) hoặc default trong [config.py](../rag_kaggle/config.py). Notebook ghi đè một số giá trị (ghi rõ ở từng dòng).
- **Doc**: chỉ có trong tài liệu của repo (`Architecture.md`, `docs/*.md`), không có trong code.
- **Chạy lại**: kết quả lệnh chạy lại khi đối chiếu.
- **Ngoài repo**: nhận định chung, không kiểm chứng được bằng code. Trên slide các bảng này có nhãn "· nhận định chung". Xem [mục cuối](#phần-không-kiểm-chứng-bằng-code).

---

## 01 · Cover

| Thông tin | Nguồn | Loại |
|---|---|---|
| Tác giả "Huy Thinh & Nho Dung" | Do người dùng cung cấp | — |
| 2 notebook | [notebooks/01_ingestion.ipynb](../notebooks/01_ingestion.ipynb), [notebooks/02_retrieve_answer.ipynb](../notebooks/02_retrieve_answer.ipynb) | Code |
| T4 × 2 | Mode `dual_t4` / `ingestion_dual_gpu` / `retrieval_dual_gpu` trong [hardware.py:12-110](../rag_kaggle/hardware.py#L12) | Code |

## 02 · Bài toán và các ràng buộc

| Thông tin | Nguồn | Loại |
|---|---|---|
| Không gọi API; model tự host | Model đều load local: [generation.py:50](../rag_kaggle/generation.py#L50) (Transformers), [storage.py:428](../rag_kaggle/storage.py#L428) (SentenceTransformer), [retrieval.py:32](../rag_kaggle/retrieval.py#L32) (FlagEmbedding), [paddleocr_vl.py:25](../rag_kaggle/paddleocr_vl.py#L25). [requirements-kaggle.txt](../requirements-kaggle.txt) không có client API LLM nào | Code |
| Preset dual T4, tự chuyển 1 GPU / CPU | [hardware.py:65-114](../rag_kaggle/hardware.py#L65) | Code |
| Định dạng hỗ trợ (.pdf .docx .xlsx .xlsm .xls…) | `allowed_extensions` [config.py:18](../rag_kaggle/config.py#L18) | Config |
| Sửa dấu OCR nhưng cấm đổi số, mã, email, URL | `_protected_tokens`, `validate_correction` [ocr_correction.py:151-169](../rag_kaggle/ocr_correction.py#L151) | Code |
| Citation do backend cấp | [generation.py:159-166](../rag_kaggle/generation.py#L159) | Code |
| Trace từng bước | [tracing.py:14-51](../rag_kaggle/tracing.py#L14) | Code |

## 03 · Toàn cảnh pipeline

| Thông tin | Nguồn | Loại |
|---|---|---|
| Ingestion chỉ có parser/OCR/VLM/chunker; Retrieval từ chối ingest | [pipeline.py:62-77](../rag_kaggle/pipeline.py#L62), guard mode [pipeline.py:105](../rag_kaggle/pipeline.py#L105), [pipeline.py:371](../rag_kaggle/pipeline.py#L371); lớp [pipeline.py:743-754](../rag_kaggle/pipeline.py#L743) | Code |
| Thành phần bundle: Qdrant, bm25.pkl, metadata.db, manifest… | File bắt buộc khi restore [pipeline.py:792](../rag_kaggle/pipeline.py#L792) | Code |
| Thiết bị của từng bước (màu ô) | Xem slide 04 | Code |

## 04 · Model nào nằm trên GPU nào

| Thông tin | Nguồn | Loại |
|---|---|---|
| Ingestion: GPU 0 = sửa OCR → VLM → embedding; GPU 1 = PaddleOCR-VL; CPU = parse/chunk/store | `configure_ingestion_devices` [hardware.py:65-88](../rag_kaggle/hardware.py#L65) | Code |
| Unload model sửa OCR trước khi load VLM | [parsers.py:333](../rag_kaggle/parsers.py#L333) | Code |
| Unload OCR, VLM, model sửa OCR trước embedding; unload encoder sau đó | [pipeline.py:306-326](../rag_kaggle/pipeline.py#L306) | Code |
| Retrieve: GPU 0 = Qwen; GPU 1 = query embedding; CPU = reranker FP32, Qdrant, BM25 | `configure_retrieval_devices` [hardware.py:91-114](../rag_kaggle/hardware.py#L91) | Code |
| Qwen load lazy, không unload giữa các câu hỏi | `load()` trả về sớm nếu đã load [generation.py:50-52](../rag_kaggle/generation.py#L50); `ask()` không gọi unload [pipeline.py:370](../rag_kaggle/pipeline.py#L370) | Code |
| "Qwen 7B 4-bit vừa một T4, tensor-parallel phản tác dụng" | Docstring [hardware.py:15](../rag_kaggle/hardware.py#L15) | Code (comment) |
| Reranker ở CPU để tận dụng RAM hệ thống | Docstring [hardware.py:13-20](../rag_kaggle/hardware.py#L13) | Code (comment) |
| 1 GPU dồn về cuda:0; không GPU chạy CPU + cảnh báo | [hardware.py:80-88](../rag_kaggle/hardware.py#L80), [hardware.py:104-114](../rag_kaggle/hardware.py#L104) | Code |

## 06 · Bước 1: Tiếp nhận

| Thông tin | Nguồn | Loại |
|---|---|---|
| File đơn, folder đệ quy, ZIP | `discover_input_files` [ingestion.py:25](../rag_kaggle/ingestion.py#L25) | Code |
| Giải nén vào staging, chặn đường dẫn thoát ra ngoài | `is_relative_to` [ingestion.py:77-78](../rag_kaggle/ingestion.py#L77) | Code |
| 200 MB/file; ZIP ≤ 5.000 file, ≤ 4.096 MB | [config.py:15-17](../rag_kaggle/config.py#L15), kiểm tra [ingestion.py:65-74](../rag_kaggle/ingestion.py#L65), [ingestion.py:105](../rag_kaggle/ingestion.py#L105) | Config + Code |
| Signature PDF/OOXML/OLE, file mã hóa, ZIP hỏng | `validate_file` [ingestion.py:93-137](../rag_kaggle/ingestion.py#L93) | Code |
| PDF có mật khẩu | `pdf.needs_pass` [parsers.py:420](../rag_kaggle/parsers.py#L420) | Code |
| Macro chỉ ghi nhận, không chạy | [ingestion.py:133-134](../rag_kaggle/ingestion.py#L133) | Code |
| Cùng tên + cùng hash thì bỏ qua; khác hash thì giữ version | `skip_unchanged_documents` [pipeline.py:206](../rag_kaggle/pipeline.py#L206); tên file `stem__hash12` [pipeline.py:728-731](../rag_kaggle/pipeline.py#L728) | Code |
| Lock: lượt ingest thứ hai bị từ chối (speaker notes) | [pipeline.py:84](../rag_kaggle/pipeline.py#L84), [pipeline.py:111](../rag_kaggle/pipeline.py#L111) | Code |

## 07 · Bước 2: Parse

| Thông tin | Nguồn | Loại |
|---|---|---|
| PyMuPDF: text, font, bbox, bảng `find_tables()` | [parsers.py:424-470](../rag_kaggle/parsers.py#L424) | Code |
| Dưới 80 ký tự là trang scan; render 220 DPI | `scan_text_threshold` [config.py:25](../rag_kaggle/config.py#L25), [parsers.py:433](../rag_kaggle/parsers.py#L433); `render_dpi` [config.py:24](../rag_kaggle/config.py#L24), [parsers.py:576](../rag_kaggle/parsers.py#L576) | Config + Code |
| DOCX: heading qua style + `outlineLvl`; đọc chart XML | [parsers.py:1092](../rag_kaggle/parsers.py#L1092), [parsers.py:1298](../rag_kaggle/parsers.py#L1298) | Code |
| Excel: openpyxl mở 2 lần (công thức + giá trị) | [parsers.py:817-818](../rag_kaggle/parsers.py#L817) | Code |
| .xls: LibreOffice, fallback pandas + calamine | [parsers.py:1002-1014](../rag_kaggle/parsers.py#L1002) | Code |
| Model chung Document → Block → Relationship | [models.py](../rag_kaggle/models.py) | Code |
| Docling được dự kiến trong thiết kế ban đầu | [Architecture.md §4.1](../Architecture.md) | Doc |

## 08 · Bước 3: OCR

| Thông tin | Nguồn | Loại |
|---|---|---|
| PaddleOCR-VL-1.6 → `pipeline_version` v1.6 | [paddleocr_vl.py:17-18](../rag_kaggle/paddleocr_vl.py#L17), notebook 01 cell 2 `OCR_MODEL` | Code |
| Ảnh tối thiểu 220 × 120 px | [config.py:37-38](../rag_kaggle/config.py#L37), [parsers.py:403-404](../rag_kaggle/parsers.py#L403) | Config + Code |
| venv riêng, paddlepaddle-gpu 3.2.1 cu126 | Notebook 01, cell "3. Cài PaddleOCR-VL" | Code |
| Worker subprocess, JSON lines có prefix | [paddleocr_vl.py:14](../rag_kaggle/paddleocr_vl.py#L14), [paddleocr_vl.py:152-200](../rag_kaggle/paddleocr_vl.py#L152) | Code |
| Pin GPU bằng `CUDA_VISIBLE_DEVICES` | [paddleocr_vl.py:163](../rag_kaggle/paddleocr_vl.py#L163), giá trị "1" ở [hardware.py:69](../rag_kaggle/hardware.py#L69) | Code |
| Map layout label → heading / table / image | [parsers.py:51-54](../rag_kaggle/parsers.py#L51), [parsers.py:607-609](../rag_kaggle/parsers.py#L607) | Code |
| Lưu raw OCR | `ocr_raw_pages` [parsers.py:602](../rag_kaggle/parsers.py#L602) | Code |
| Cài lỗi thì tắt OCR (speaker notes) | Notebook 01 cell 3 (`except` → `OCR_PYTHON = None`), cell 4 `enable_ocr` | Code |

## 09 · Bước 4: Sửa chính tả OCR

| Thông tin | Nguồn | Loại |
|---|---|---|
| Qwen2.5-3B-Instruct, NF4, temperature 0 | [baseline.yaml:25-29](../configs/baseline.yaml#L25); NF4 [ocr_correction.py:196-205](../rag_kaggle/ocr_correction.py#L196) | Config + Code |
| Producer–consumer, buffer 8, thread riêng | [ocr_correction.py:283-290](../rag_kaggle/ocr_correction.py#L283), [ocr_correction.py:319-335](../rag_kaggle/ocr_correction.py#L319); `max_buffered_components` [config.py:59](../rag_kaggle/config.py#L59) | Code + Config |
| 600 ký tự context trước, khúc ≤ 1.200 ký tự | [config.py:66-67](../rag_kaggle/config.py#L66); prompt [ocr_correction.py:70-87](../rag_kaggle/ocr_correction.py#L70) | Config + Code |
| Chỉ nhận JSON `corrected_text` | `parse_correction_output` [ocr_correction.py:172-182](../rag_kaggle/ocr_correction.py#L172) | Code |
| Mất số/mã/email/URL hoặc tỷ lệ độ dài ngoài 0,35–3 thì từ chối | `validate_correction` [ocr_correction.py:156-169](../rag_kaggle/ocr_correction.py#L156) | Code |
| Retry 1 lần (bỏ context), rồi giữ OCR gốc | [ocr_correction.py:532-560](../rag_kaggle/ocr_correction.py#L532); `max_retries`, `fail_open` [config.py:60-61](../rag_kaggle/config.py#L60) | Code + Config |
| Label được sửa; bảng mặc định không sửa | [config.py:62-63](../rag_kaggle/config.py#L62) | Config |
| Heading đã sửa ghi ngược vào `section_path` | `_apply_corrected_headings` [parsers.py:203](../rag_kaggle/parsers.py#L203) | Code |
| BM25 khớp theo âm tiết | `tokenize_vi` [storage.py:730-741](../rag_kaggle/storage.py#L730) | Code |
| Tắt bằng `OCR_CORRECTION_MODEL = None` | Notebook 01 cell 2, cell 4 | Code |
| Glossary mặc định tắt (speaker notes) | [baseline.yaml:39](../configs/baseline.yaml#L39) | Config |

## 10 · Bước 5: VLM

| Thông tin | Nguồn | Loại |
|---|---|---|
| Qwen2.5-VL-3B, NF4, mặc định tắt | [baseline.yaml:43-44](../configs/baseline.yaml#L43); NF4 [vision.py:75-88](../rag_kaggle/vision.py#L75); notebook 01 `VISION_MODEL = None` | Config + Code |
| Chỉ cho flowchart / chart / diagram | `image_types` [config.py:83](../rag_kaggle/config.py#L83), [vision.py:122](../rag_kaggle/vision.py#L122) | Config + Code |
| Phân loại theo caption, nhãn OCR, mũi tên | `classify_image` [vision.py:18-28](../rag_kaggle/vision.py#L18) | Code |
| Chạy sau khi sửa OCR xong; unload model sửa trước | Deferred vision [parsers.py:119](../rag_kaggle/parsers.py#L119), [parsers.py:328-333](../rag_kaggle/parsers.py#L328) | Code |
| Đuôi 1 trang trước, 1.500 ký tự | `previous_page_chars` [config.py:85](../rag_kaggle/config.py#L85), [vision.py:131](../rag_kaggle/vision.py#L131); chọn trang trước [parsers.py:382](../rag_kaggle/parsers.py#L382) | Config + Code |
| Flowchart: nodes/edges; chart: observations tách summary | Prompt [vision.py:41-56](../rag_kaggle/vision.py#L41); render [vision.py:191-219](../rag_kaggle/vision.py#L191) | Code |
| Retry 2 lần rồi `needs_review` | [config.py:81](../rag_kaggle/config.py#L81), [vision.py:137-147](../rag_kaggle/vision.py#L137) | Config + Code |
| Block VLM chèn sau ảnh, quan hệ `derived_from` (speaker notes) | [parsers.py:358](../rag_kaggle/parsers.py#L358), [relationships.py:190-192](../rag_kaggle/relationships.py#L190) | Code |
| "Bật khi evaluation cho thấy cần" | Comment [baseline.yaml:43](../configs/baseline.yaml#L43) | Config (comment) |

## 11 · Bước 6: Hồ sơ tài liệu và quan hệ

| Thông tin | Nguồn | Loại |
|---|---|---|
| title, doc_number, issue_date, issuer, signers, keywords | `build_document_profile` [profile.py:157-184](../rag_kaggle/profile.py#L157) | Code |
| Entities person/location/other | `_classify` [profile.py:138-152](../rag_kaggle/profile.py#L138) | Code |
| Gom biến thể theo key bỏ dấu | `normalize_key` [profile.py:46](../rag_kaggle/profile.py#L46), `strip_diacritics` [entities.py:25](../rag_kaggle/entities.py#L25) | Code |
| 7 loại quan hệ | [relationships.py:52-192](../rag_kaggle/relationships.py#L52); danh sách [models.py:9-17](../rag_kaggle/models.py#L9) | Code |
| Lưu SQLite; gắn payload, BM25, prompt | Bảng `document_entities`, `document_keywords` [storage.py:69-83](../rag_kaggle/storage.py#L69); payload [models.py:111-135](../rag_kaggle/models.py#L111); `bm25_text` [storage.py:703-711](../rag_kaggle/storage.py#L703); prompt [generation.py:118-131](../rag_kaggle/generation.py#L118) | Code |
| Không dùng LLM | profile.py / entities.py chỉ dùng regex và thống kê | Code |

## 12 · Bước 7: Chunking

| Thông tin | Nguồn | Loại |
|---|---|---|
| `StructureAwareChunker` | [chunking.py:17](../rag_kaggle/chunking.py#L17) | Code |
| Text gộp tới 1.800 ký tự, không overlap | [config.py:96](../rag_kaggle/config.py#L96); [chunking.py:74-75](../rag_kaggle/chunking.py#L74); overlap = 0 [chunking.py:163](../rag_kaggle/chunking.py#L163) | Config + Code |
| Bảng ≤ 6.000 ký tự giữ nguyên; header kế thừa, caption | [chunking.py:103-126](../rag_kaggle/chunking.py#L103) | Code |
| Bảng lớn: preview 5 × 5, bảng đầy đủ ra .xlsx | [chunking.py:126-133](../rag_kaggle/chunking.py#L126); [config.py:100-101](../rag_kaggle/config.py#L100) | Code + Config |
| Ảnh = 1 chunk, ngưỡng 6.000 | [chunking.py:159](../rag_kaggle/chunking.py#L159), [config.py:103](../rag_kaggle/config.py#L103) | Code + Config |
| Header embed: Section / Sheet / Range | `_header` [chunking.py:172-180](../rag_kaggle/chunking.py#L172) | Code |
| Parent-child là thiết kế cũ; DB cũ bị từ chối (speaker notes) | [Architecture.md §6](../Architecture.md), `ARTIFACT_SCHEMA_VERSION = 2` [config.py:10](../rag_kaggle/config.py#L10), test `test_legacy_schema_is_rejected_with_a_clear_message` | Doc + Code |

## 13 · Bước 8: Dense embedding

| Thông tin | Nguồn | Loại |
|---|---|---|
| sentence-transformers, BAAI/bge-m3 | [storage.py:428-447](../rag_kaggle/storage.py#L428); [baseline.yaml:60](../configs/baseline.yaml#L60) `dense_model` | Code + Config |
| Normalize, cosine, named vector `dense` | [storage.py:491](../rag_kaggle/storage.py#L491), [storage.py:558](../rag_kaggle/storage.py#L558), [storage.py:22](../rag_kaggle/storage.py#L22) | Code |
| GPU 0 khi ingest, GPU 1 khi hỏi | [hardware.py:70](../rag_kaggle/hardware.py#L70), [hardware.py:95](../rag_kaggle/hardware.py#L95) | Code |
| Batch 4; OOM giảm batch rồi retry | [baseline.yaml:63](../configs/baseline.yaml#L63); [storage.py:485-498](../rag_kaggle/storage.py#L485) | Config + Code |
| Cache vector trong SQLite | Bảng `embedding_cache` [storage.py:105](../rag_kaggle/storage.py#L105), [storage.py:402-410](../rag_kaggle/storage.py#L402) | Code |
| Model/revision khóa trong manifest; notebook 02 phải khớp | [pipeline.py:542-551](../rag_kaggle/pipeline.py#L542), [pipeline.py:819-831](../rag_kaggle/pipeline.py#L819); notebook 02 `EMBEDDING_MODEL = None` | Code |
| Gợi ý bge-multilingual-gemma2 khi GPU ≥ 24 GB, phải ingest lại | [hardware.py:157-159](../rag_kaggle/hardware.py#L157) | Code |
| bge-multilingual-gemma2 là mục tiêu ban đầu | Comment [baseline.yaml:58-59](../configs/baseline.yaml#L58); [Architecture.md §8.1](../Architecture.md) | Config (comment) + Doc |

## 14 · Bước 9: Lưu trữ

| Thông tin | Nguồn | Loại |
|---|---|---|
| Qdrant local `QdrantClient(path=…)` | [storage.py:521](../rag_kaggle/storage.py#L521) | Code |
| Payload để lọc; payload index | `Chunk.payload` [models.py:111](../rag_kaggle/models.py#L111); `_create_payload_indexes` [storage.py:649](../rag_kaggle/storage.py#L649) | Code |
| BM25Okapi: nội dung + tên file + tiêu đề + entities không dấu | [storage.py:671-677](../rag_kaggle/storage.py#L671), `bm25_text` [storage.py:703-711](../rag_kaggle/storage.py#L703) | Code |
| Tokenizer giữ số, mã, dấu chấm, gạch chéo | `tokenize_vi` [storage.py:730-741](../rag_kaggle/storage.py#L730) | Code |
| SQLite: documents, blocks, chunks, relationships, cache | `CREATE TABLE` [storage.py:47-110](../rag_kaggle/storage.py#L47) | Code |
| Parquet; .xlsx cho bảng lớn | [pipeline.py:265-266](../rag_kaggle/pipeline.py#L265), `save_table_parquet` [parsers.py:1369](../rag_kaggle/parsers.py#L1369) | Code |
| Batch upsert 64, ID ổn định (speaker notes) | [storage.py:561-571](../rag_kaggle/storage.py#L561), `qdrant_point_id` [storage.py:744](../rag_kaggle/storage.py#L744) | Code |

## 15 · Bước 10: Freeze

| Thông tin | Nguồn | Loại |
|---|---|---|
| Manifest: corpus ID, schema version, embedding | `freeze_corpus` [pipeline.py:525-551](../rag_kaggle/pipeline.py#L525) | Code |
| Phiên bản parser/chunker | [pipeline.py:354-355](../rag_kaggle/pipeline.py#L354) | Code |
| checksums.json; đóng Qdrant, commit SQLite rồi mới zip; bundle nằm ngoài work_dir | [pipeline.py:614-649](../rag_kaggle/pipeline.py#L614) | Code |
| Restore: staging, kiểm tra path/schema/checksum, rollback | [pipeline.py:657-712](../rag_kaggle/pipeline.py#L657), [pipeline.py:792-809](../rag_kaggle/pipeline.py#L792) | Code |
| Corpus freeze không nhận ingest thêm | [pipeline.py:107-109](../rag_kaggle/pipeline.py#L107) | Code |
| Bảng "Vì sao tách 2 notebook" | Hàng VRAM và Mở chat: [pipeline.py:62-77](../rag_kaggle/pipeline.py#L62); hàng Tái lập: checksum ở trên. Hàng Debug là lập luận | Code + lập luận |

## 17 · Retrieve bước 1: Hybrid search

| Thông tin | Nguồn | Loại |
|---|---|---|
| Dense top 30, BM25 top 30, fused top 30, rrf_k 60 | [baseline.yaml:64-67](../configs/baseline.yaml#L64) | Config |
| RRF = Σ 1/(k + rank) | [retrieval.py:121-125](../rag_kaggle/retrieval.py#L121) | Code |
| Entity boost: ranking thêm, không phải filter | `match_entities` [storage.py:326](../rag_kaggle/storage.py#L326); [pipeline.py:399-407](../rag_kaggle/pipeline.py#L399) | Code |
| Câu gốc luôn được tìm | [retrieval.py:98](../rag_kaggle/retrieval.py#L98) | Code |
| Rewrite tắt; rewrite làm mất số thì bị bỏ | [baseline.yaml:84](../configs/baseline.yaml#L84); `sanitize_plan` [generation.py:215-228](../rag_kaggle/generation.py#L215) | Config + Code |
| Ví dụ "499.000" | Comment trong `tokenize_vi` [storage.py:738](../rag_kaggle/storage.py#L738) | Code (comment) |

## 18 · Retrieve bước 2: Rerank

| Thông tin | Nguồn | Loại |
|---|---|---|
| FlagReranker, bge-reranker-v2-m3 | [retrieval.py:24-44](../rag_kaggle/retrieval.py#L24) | Code |
| 30 → top 8 | [baseline.yaml:67-68](../configs/baseline.yaml#L67) | Config |
| T4 × 2: CPU, tắt FP16; 1 GPU: cuda:0 | [hardware.py:96-97](../rag_kaggle/hardware.py#L96), [hardware.py:107](../rag_kaggle/hardware.py#L107); `use_fp16` [retrieval.py:35](../rag_kaggle/retrieval.py#L35) | Code |
| Bằng chứng yếu (min_rerank_score 0,05) thì từ chối | [baseline.yaml:91](../configs/baseline.yaml#L91); `retrieval_confidence` [guardrails.py:54-67](../rag_kaggle/guardrails.py#L54); [pipeline.py:413-422](../rag_kaggle/pipeline.py#L413) | Config + Code |
| Tắt bằng `RERANKER_MODEL = None` | Notebook 02 cell 2, cell 3 | Code |

## 19 · Retrieve bước 3: Mở rộng context và tính toán

| Thông tin | Nguồn | Loại |
|---|---|---|
| 1 chunk kề mỗi phía, cùng section | [baseline.yaml:73](../configs/baseline.yaml#L73); `neighbor_chunks` [storage.py:309](../rag_kaggle/storage.py#L309); [retrieval.py:182](../rag_kaggle/retrieval.py#L182) | Config + Code |
| Bảng preview nạp đủ hàng, tối đa 12.000 ký tự | [retrieval.py:254-261](../rag_kaggle/retrieval.py#L254); [baseline.yaml:74](../configs/baseline.yaml#L74) | Code + Config |
| Theo quan hệ | `_related_blocks` [retrieval.py:283-290](../rag_kaggle/retrieval.py#L283) | Code |
| Khử trùng, sắp thứ tự, SOURCE_x, 28.000 ký tự | [retrieval.py:160-238](../rag_kaggle/retrieval.py#L160); [baseline.yaml:81](../configs/baseline.yaml#L81) | Code + Config |
| sum/avg/max/min/count theo từ khóa Việt/Anh | [computation.py:11-18](../rag_kaggle/computation.py#L11) | Code |
| Bỏ dòng "Tổng" | `TOTAL_ROW_PATTERN` [computation.py:17](../rag_kaggle/computation.py#L17), [computation.py:110](../rag_kaggle/computation.py#L110) | Code |
| Kết quả tính đặt trước context | [generation.py:119](../rag_kaggle/generation.py#L119) | Code |
| Prompt cấm tự cộng/trừ bảng | Quy tắc 7 [generation.py:22](../rag_kaggle/generation.py#L22) | Code |
| Chỉ một kết quả tốt nhất; chưa join/group-by | `compute_from_blocks` [computation.py:66](../rag_kaggle/computation.py#L66); [project-summary.md §11.1](project-summary.md) | Code + Doc |

## 20 · Retrieve bước 4: Sinh câu trả lời

| Thông tin | Nguồn | Loại |
|---|---|---|
| Qwen2.5-7B-Instruct, NF4 + double quant, compute FP16 | [baseline.yaml:78](../configs/baseline.yaml#L78); [generation.py:54-62](../rag_kaggle/generation.py#L54) | Config + Code |
| Temperature 0,1; 1.024 token; context 28.000 ký tự | [baseline.yaml:81-83](../configs/baseline.yaml#L81) | Config |
| Load lazy, GPU 0 | [generation.py:50-52](../rag_kaggle/generation.py#L50); [hardware.py:98](../rag_kaggle/hardware.py#L98) | Code |
| Quy tắc prompt (chỉ dùng NGỮ CẢNH, tài liệu là dữ liệu…) | [generation.py:14-23](../rag_kaggle/generation.py#L14) | Code |
| Source kèm file, tiêu đề, số hiệu, ngày, người ký, file bảng | [generation.py:118-136](../rag_kaggle/generation.py#L118) | Code |
| JSON answer + source_ids; backend map ID | [generation.py:159-166](../rag_kaggle/generation.py#L159), [generation.py:206-209](../rag_kaggle/generation.py#L206) | Code |
| "Qwen 7B 4-bit vừa một T4" | Docstring [hardware.py:15](../rag_kaggle/hardware.py#L15) | Code (comment) |
| Gợi ý Qwen2.5-14B khi GPU ≥ 24 GB | [hardware.py:161-163](../rag_kaggle/hardware.py#L161) | Code |
| `ask()` chưa dùng lịch sử chat (speaker notes) | Chữ ký `ask(self, query, filters)` [pipeline.py:370](../rag_kaggle/pipeline.py#L370) | Code |

## 21 · Retrieve bước 5: Guardrails, trace, UI

| Thông tin | Nguồn | Loại |
|---|---|---|
| Regex prompt injection | `INJECTION_PATTERNS` [guardrails.py:12-37](../rag_kaggle/guardrails.py#L12) | Code |
| Vô hiệu token chat template, marker SOURCE giả | `sanitize_context` [guardrails.py:40-43](../rag_kaggle/guardrails.py#L40) | Code |
| Từ chối khi không có hit / bằng chứng yếu | [guardrails.py:54-67](../rag_kaggle/guardrails.py#L54) | Code |
| Cảnh báo số không có trong context | `unsupported_numbers` [guardrails.py:70](../rag_kaggle/guardrails.py#L70); [pipeline.py:450](../rag_kaggle/pipeline.py#L450) | Code |
| Mask email, SĐT, số thẻ | [guardrails.py:29-52](../rag_kaggle/guardrails.py#L29) | Code |
| trace_id; ghi `query_traces.jsonl` | [tracing.py:18](../rag_kaggle/tracing.py#L18), [tracing.py:51](../rag_kaggle/tracing.py#L51) | Code |
| Trace nằm ngoài corpus frozen | `session_dir` [config.py](../rag_kaggle/config.py) (`RuntimeConfig.session_dir`) | Code |
| Gradio 5 | `gradio>=5.20,<6` [requirements-kaggle.txt](../requirements-kaggle.txt) | Config |
| Tab Ingestion / Chat / Evaluation / System; rank dense/BM25/RRF/rerank, gallery, .xlsx | [ui.py:197-246](../rag_kaggle/ui.py#L197), [ui.py:134-137](../rag_kaggle/ui.py#L134) | Code |
| `share=True` | Notebook 02 cell 4 | Code |

## 22 · Đánh giá

| Thông tin | Nguồn | Loại |
|---|---|---|
| Retrieval: recall@k, MRR, hit rate theo loại, block coverage | `retrieval_metrics` [evaluation.py:67-93](../rag_kaggle/evaluation.py#L67) | Code |
| Answer: correctness, numeric exact match, faithfulness proxy, citation, refusal | `answer_metrics` [evaluation.py:127-143](../rag_kaggle/evaluation.py#L127) | Code |
| Vận hành: parse success, lỗi OCR/VLM, throughput, latency, peak VRAM | [evaluation.py:193-245](../rag_kaggle/evaluation.py#L193), `operational_metrics` [evaluation.py:221](../rag_kaggle/evaluation.py#L221) | Code |
| Dataset mẫu 4 tình huống | [evaluation/dataset.sample.jsonl](../evaluation/dataset.sample.jsonl) | Code (data) |
| 67 test, 62 pass, 5 skip vì thiếu dependency parser/index | `python3 -m unittest discover -s tests` chạy lại ngày 09/10/2026: `Ran 67 tests … OK (skipped=5)`; lý do skip: `parser/index deps missing`, `index deps missing` | Chạy lại |
| Phần model trong test là fake | [tests/](../tests) | Code |
| Chưa có benchmark model thật | Không có output benchmark trong repo; [project-summary.md §19](project-summary.md) | Doc |
| smoke.py có sẵn (speaker notes) | `run_smoke_test` [smoke.py:42](../rag_kaggle/smoke.py#L42) | Code |

## 23 · Hướng tiếp theo

| Thông tin | Nguồn | Loại |
|---|---|---|
| 50–100 câu hỏi có ground truth | [Architecture.md §14.1](../Architecture.md) | Doc |
| Bật rewrite, VLM, glossary, sửa bảng OCR khi có số đo | Các cờ đang tắt trong [baseline.yaml](../configs/baseline.yaml) (dòng 36, 39, 43, 84) và comment của chúng | Config |
| Gemma2 embedding (phải ingest lại), Qwen 14B khi GPU 24 GB | [hardware.py:157-163](../rag_kaggle/hardware.py#L157) | Code |
| Tính năng còn thiếu (đa lượt, NER, nhiều bảng, phân quyền, Qdrant server) | [project-summary.md §19](project-summary.md) | Doc |

Các slide 05 và 16 là slide chuyển phần, chỉ nhắc lại tên bước; nguồn giống các slide chi tiết.

---

## Phần không kiểm chứng bằng code

Các nội dung dưới đây **không thể** chứng minh bằng code trong repo vì chúng nói về công cụ bên ngoài. Trên slide chúng nằm trong bảng có nhãn **"· nhận định chung"**. Riêng hàng "Đang dùng" của mỗi bảng đã được viết lại để chỉ chứa thông tin có trong code.

| Slide | Nội dung ngoài repo |
|---|---|
| 06 | Dedup theo tên file, theo thời gian sửa; LangChain DirectoryLoader |
| 07 | Docling, Unstructured, pdfplumber, pandas.read_excel |
| 08 | Tesseract, VietOCR, Qwen2.5-VL 7B đọc cả trang, Google/Azure OCR |
| 09 | Rule-based, BARTpho, Qwen2.5-7B cho bước sửa, phương án không sửa |
| 10 | Qwen2.5-VL-7B, InternVL, MiniCPM-V, Florence-2 |
| 11 | underthesea/PhoBERT NER, LLM trích xuất, GraphRAG |
| 12 | Fixed-size + overlap, semantic chunking, đánh giá parent-child |
| 13 | mE5-large, bi-encoder PhoBERT, OpenAI/Cohere embedding |
| 14 | FAISS, Chroma, Milvus/pgvector/Elastic |
| 17 | Weighted score fusion, chỉ dense, chỉ BM25, sparse của BGE-M3 |
| 18 | BGE reranker Gemma, Jina reranker v2, ms-marco MiniLM, Cohere Rerank |
| 20 | Llama 3.1 8B, Gemma 2 9B, Vistral-7B/SeaLLM |
| 21 | Gradio so với Streamlit / web app riêng |
| 15 | Hàng "Debug" trong bảng tách 2 notebook (lập luận) |

## Đã sửa trên slide để khớp code

| Slide | Trước | Sau | Lý do |
|---|---|---|---|
| 01 | Nho Dung | Huy Thinh & Nho Dung | Yêu cầu của người dùng |
| 02 | "2 GPU 16 GB, khoảng 30 GB RAM" | "Preset dual T4 trong hardware.py…" | Dung lượng GPU/RAM không có trong code |
| 02 | "Tên riêng, số hiệu… chính xác tuyệt đối" | "số, mã, email, URL tuyệt đối không được đổi" | Khớp `_protected_tokens` |
| 04 | "chỉ thêm độ trễ truyền dữ liệu" | "phản tác dụng với hỏi đáp tương tác" | Khớp docstring hardware.py |
| 06 | 4 GB | 4.096 MB | Khớp `max_archive_mb` |
| 08 | "VLM nhỏ (~0.9B)…" | Mô tả theo cách code map layout label | Kích thước model không có trong repo |
| 09 | "lệch độ dài", "title, heading…" | "tỷ lệ 0,35–3", tên label thật | Khớp `validate_correction`, `correct_labels` |
| 09 | Ví dụ "Quyết đinh" và "Chỉ vài GB VRAM" | Lý do theo tokenizer BM25; NF4 + validate | Không có trong repo |
| 12 | "(~450 token)" | bỏ | Comment yaml ghi ~450 nhưng `chars_per_token = 3.6` cho ra ~500; bỏ để tránh mâu thuẫn |
| 13 | "1024 chiều", "~568M, 8.192 token" | bỏ | Không có trong code (`dense_dimension` để None) |
| 17 | "15/2024/TT-NHNN" | "499.000" | Ví dụ lấy từ comment trong code |
| 18 | "~568M", "30 GB RAM" | FlagReranker, CPU, tắt FP16 | Không có trong code |
| 19 | "LLM 7B tự cộng 40 dòng dễ sai"; "Text-to-SQL đã cân nhắc" | Quy tắc 7 của prompt; giới hạn hiện tại | Khớp generation.py / computation.py |
| 20 | "license Apache-2.0", "context dài", "sát 16 GB" | NF4 + double quant; gợi ý trong hardware.py | Không có trong repo |
| 22 | "chạy local 08/10/2026" | "chạy lại 09/10/2026" | Đã chạy lại test |
