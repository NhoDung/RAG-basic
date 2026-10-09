# Tổng hợp những gì project RAG-basic đã thực hiện

> Ngày đối chiếu: **08/10/2026**. Tài liệu phản ánh mã nguồn trong working tree tại thời điểm đọc, bao gồm các thay đổi local đang có. “Đã triển khai” nghĩa là có logic trong code; không đồng nghĩa đã được benchmark thành công với model thật trên Kaggle.

## 1. Mục tiêu và kết quả triển khai

Project xây dựng một **hệ thống Multimodal RAG cho tài liệu tiếng Việt**, hướng tới chạy trên Kaggle. Hệ thống đọc PDF, DOCX và Excel; chuẩn hóa văn bản, bảng, hình ảnh và biểu đồ; lập chỉ mục tìm kiếm; sau đó dùng mô hình ngôn ngữ để trả lời câu hỏi kèm nguồn dẫn.

Các phần chính đã có:

- Nạp tài liệu từ file đơn, folder hoặc ZIP, kiểm tra đầu vào và ghi trạng thái xử lý.
- Parse riêng cho PDF có text, PDF scan, DOCX và các định dạng Excel.
- Tích hợp PaddleOCR-VL, LLM sửa chính tả OCR và VLM bổ sung cho hình phù hợp.
- Chuẩn hóa dữ liệu thành document, block, chunk và quan hệ giữa các block.
- Trích xuất hồ sơ tài liệu, từ khóa, tên riêng và biến thể tên bằng heuristic.
- Chunking theo cấu trúc; giữ bảng nhỏ nguyên vẹn, lưu bảng lớn đầy đủ ra `.xlsx`.
- Dense embedding, Qdrant local, BM25, hợp nhất bằng RRF và reranker.
- Mở rộng ngữ cảnh bằng chunk lân cận, bảng gốc và các quan hệ tài liệu.
- Tính toán một số phép tổng hợp bảng bằng Python.
- Sinh câu trả lời bằng Qwen, ánh xạ citation từ ID do backend cấp.
- Đóng băng, xuất và phục hồi corpus giữa hai session độc lập.
- Giao diện Gradio, trace, evaluation, smoke test và bộ test tự động.

Package Python có tên `rag-kaggle-baseline`, phiên bản `0.1.0`, yêu cầu Python từ `3.10` theo [pyproject.toml](../pyproject.toml).

## 2. Kiến trúc thực tế hiện tại

### 2.1. Tách ingestion và hỏi đáp

Hai notebook phục vụ hai giai đoạn độc lập:

```text
01_ingestion.ipynb
File / folder / ZIP
  → kiểm tra → parse + OCR → sửa OCR → VLM tùy chọn
  → quan hệ + hồ sơ tài liệu → chunking → embedding
  → Qdrant + BM25 + SQLite + tables/assets/parsed
  → freeze → corpus_bundle.zip

02_retrieve_answer.ipynb
corpus_bundle.zip
  → kiểm tra checksum/schema/embedding → phục hồi corpus
  → query → dense + BM25 → RRF → reranker
  → mở rộng context + tính toán bảng → Qwen
  → câu trả lời + citation + trace
```

Ba lớp public trong [pipeline.py](../rag_kaggle/pipeline.py):

- `IngestionPipeline`: parse, chunk, index, freeze và export; chặn hỏi đáp/evaluation.
- `RetrievalAnswerPipeline`: restore và hỏi đáp/evaluation; không khởi tạo parser, OCR, correction, VLM hoặc chunker; chặn ingest.
- `RAGPipeline`: chế độ `full` để tương thích và phục vụ test/API cũ.

Corpus đã frozen không nhận thêm tài liệu qua API ingest. Muốn xây corpus khác cần dùng work directory mới. Đây là ràng buộc ở tầng ứng dụng; storage vẫn được mở bằng SQLite/Qdrant local, không phải cơ chế read-only cưỡng chế ở filesystem.

### 2.2. Thay đổi so với thiết kế ban đầu

[Architecture.md](../Architecture.md) chứa cả định hướng thiết kế và mô tả cũ về parent-child. Implementation hiện tại dùng `StructureAwareChunker`, không còn truy xuất parent chunk. Ngữ cảnh được mở rộng từ chunk cùng mục, block liên quan và dữ liệu bảng đầy đủ.

Các mốc version trong code:

- Parser: `0.4.0`.
- Chunker: `1.0.0-structure-aware`.
- Artifact schema: `2`.
- Prompt sửa OCR: `ocr-correction-v2`.
- Document profile: version `1`.

`ChildChunk` còn là alias của `Chunk` để tương thích tên cũ. Database có schema parent-child cũ bị từ chối và yêu cầu ingest lại.

## 3. Tiếp nhận và kiểm tra tài liệu

Nguồn: [ingestion.py](../rag_kaggle/ingestion.py), [pipeline.py](../rag_kaggle/pipeline.py).

### 3.1. Phát hiện file

`discover_input_files()` hỗ trợ file đơn, duyệt folder đệ quy và giải nén ZIP vào staging. Danh sách được sắp xếp để thứ tự xử lý ổn định; đường dẫn trùng được loại bỏ. File tạm Office, `.DS_Store`, `thumbs.db` và extension không hỗ trợ được ghi vào danh sách skipped.

Các extension được parser/API nhận: `.pdf`, `.docx`, `.xlsx`, `.xlsm`, `.xltx`, `.xltm`, `.xls`. ZIP là phương tiện chứa các file này, không phải một modality được index. Ảnh được xử lý khi nằm trong tài liệu; chưa có parser cho ảnh upload độc lập.

Baseline giới hạn một file ở 200 MB; ZIP tối đa 5.000 entry và 4.096 MB tổng kích thước giải nén. Trước khi giải nén ZIP đầu vào, code kiểm tra đường dẫn không thoát khỏi staging.

### 3.2. Validation và quản lý phiên bản

- Kiểm tra file tồn tại, extension, dung lượng, file rỗng và signature PDF/OOXML/OLE.
- Phát hiện một số trường hợp file Office mã hóa, ZIP hỏng và PDF yêu cầu mật khẩu.
- Ghi nhận sự hiện diện của macro; không thực thi macro.
- Tính SHA-256 nội dung. Upload cùng tên gốc và cùng hash được bỏ qua khi bật `skip_unchanged_documents`.
- Cùng tên nhưng nội dung khác được giữ thành phiên bản khác, có `uploaded_at`, `content_hash`, `original_file_name`.
- Bản source trong runtime dùng tên chứa hash để tránh ghi đè file cùng tên.
- Lock ingestion từ chối một lượt ingest thứ hai đang chạy. UI có thêm `UploadIngestionGate` để chặn batch trùng/đồng thời.

API cho phép thêm tài liệu khi corpus chưa frozen. UI ingestion export và freeze ngay sau batch thành công, vì vậy không thể tiếp tục thêm vào corpus đó bằng lượt upload tiếp theo.

## 4. Parse theo định dạng

Nguồn chính: [parsers.py](../rag_kaggle/parsers.py).

### 4.1. PDF có text layer

- Dùng PyMuPDF đọc text, font, bounding box, bảng và ảnh.
- Ước lượng font body và phân cấp heading theo cỡ chữ, độ đậm, đánh số.
- Gắn `section_path`, trang, bbox vào block.
- Tách bảng bằng `page.find_tables()`; tránh lấy lại text đã nằm trong bảng.
- Giữ thứ tự text do PyMuPDF cung cấp, chèn bảng/ảnh vào luồng theo vị trí.
- Nhận diện caption bằng mẫu tiếng Việt/Anh.
- Trích xuất ảnh đủ kích thước và đưa qua OCR khi bật cấu hình tương ứng.

### 4.2. PDF scan hoặc trang rất ít text

Ngưỡng mặc định phân loại scan là ít hơn 80 ký tự text trên trang. Trang được render ở 220 DPI rồi gửi PaddleOCR-VL.

- Có layout output: ánh xạ title thành heading, table thành rows, chart/image thành block tương ứng và text thành block văn bản.
- Bỏ một số label header/footer/page number.
- Không có layout nhưng có text: tạo một block `ocr_page`.
- Lưu ảnh render, raw OCR, bbox và confidence nếu OCR trả về.
- Submit các component phù hợp cho correction theo reading order.
- OCR lỗi được lưu warning; không tự tạo nội dung thay thế.

Nhánh scan đọc layout OCR trực tiếp; không phải mọi chart trên trang scan đều đi qua VLM. VLM bổ sung được lên lịch từ đường xử lý visual qua `_add_visual()`.

### 4.3. DOCX

- Dùng python-docx duyệt paragraph/table theo thứ tự trong body.
- Nhận heading qua style tiếng Anh/Việt và `outlineLvl`; nhận caption qua style hoặc pattern.
- Lưu text header/footer trong metadata.
- Đọc bảng, bảng lồng và giá trị cell gộp.
- Trích ảnh gắn với paragraph; xử lý thêm media không nằm trong body flow ở cuối.
- Đọc chart XML để lấy loại chart, title, axes, series và cached data.
- Đọc workbook `.xlsx`/`.xlsm` nhúng trong DOCX, chuyển block về document chứa nó.

DOCX không được render thành trang để suy ra page citation; citation thường dựa vào mục. Media ngoài body flow không có bảo đảm đặt lại đúng vị trí layout gốc.

### 4.4. Excel hiện đại

- Dùng openpyxl mở workbook hai lần: một lần lấy công thức, một lần lấy cached value.
- Lưu formula, cached value, number format, tọa độ cell và display value.
- Chuẩn hóa ngày, boolean, số và phần trăm.
- Tách vùng dữ liệu bằng hàng/cột trống; phân loại thành text, table hoặc KPI bằng heuristic.
- Mang theo `sheet_name`, `cell_range`, section và origin của vùng.
- Đọc merged cells, hidden sheet/row/column; mặc định bỏ qua dữ liệu ẩn theo cấu hình.
- Đọc chart title, axes, categories, series, vùng nguồn và ảnh nhúng.
- Cảnh báo cell công thức thiếu cached value; không tự tính lại workbook.

### 4.5. Excel `.xls`

Ưu tiên chuyển `.xls` sang `.xlsx` bằng LibreOffice headless. Nếu thiếu LibreOffice, code có fallback `pandas.read_excel(..., engine="calamine")` để đọc sheet và bảng. Fallback này không cung cấp đầy đủ chart/formula/layout như nhánh openpyxl.

## 5. OCR, sửa chính tả và hiểu hình

### 5.1. Adapter PaddleOCR-VL

Nguồn: [paddleocr_vl.py](../rag_kaggle/paddleocr_vl.py).

- Có thể chạy in-process hoặc bằng Python trong môi trường riêng.
- Worker subprocess dùng giao thức JSON lines với prefix riêng; log thư viện được tách khỏi protocol output.
- Chọn GPU worker qua `CUDA_VISIBLE_DEVICES`.
- Chuẩn hóa output khác nhau của PaddleOCR thành text/layout blocks/raw data.
- Ghi nhớ lỗi load để tránh lặp lại load lỗi ở mọi trang.
- Hỗ trợ model directory local và thử constructor tương thích nhiều phiên bản package.

Nếu constructor không chấp nhận `pipeline_version`, adapter có thể dùng model mặc định của package và ghi warning. Tên cấu hình OCR vì vậy chưa tự bảo đảm chính xác model thực tế nếu dependency không được pin tương ứng.

### 5.2. Sửa OCR bằng LLM

Nguồn: [ocr_correction.py](../rag_kaggle/ocr_correction.py), [entities.py](../rag_kaggle/entities.py).

OCR producer và correction consumer chạy chồng lấp qua buffer giới hạn. Correction vẫn tuần tự theo component. Mỗi request chứa system instruction, quy tắc sửa lỗi, glossary tùy chọn, đuôi context trước và segment hiện tại; không mang toàn bộ lịch sử tài liệu.

- Baseline dùng Qwen2.5-3B-Instruct, 4-bit, temperature `0`.
- Buffer mặc định 8 component; context trước 600 ký tự; segment 1.200 ký tự.
- Segment dài được chia theo câu/dòng, có hard split khi cần.
- Chỉ nhận JSON có đúng trường `corrected_text`.
- Kiểm tra output rỗng, tỷ lệ chiều dài và việc mất token được bảo vệ như số, mã, email, URL.
- Retry thay đổi prompt bằng cách bỏ context/glossary; mặc định lỗi thì giữ raw text (`fail_open`).
- Tùy chọn sửa bảng kiểm tra số dòng/cell và ghi lại `metadata.rows` cùng Markdown.
- Lưu raw OCR, trạng thái, model/revision/prompt version, latency và token usage.
- Reset context giữa document; quản lý generation của worker để loại kết quả document đã abort khỏi tiến độ document mới.
- Heading sửa xong được cập nhật vào `section_path` của block con.
- Glossary gom các biến thể tên theo dạng bỏ dấu, chọn cách viết xuất hiện nhiều nhất; mặc định tắt.

Prompt yêu cầu bảo toàn giá trị nghiệp vụ, nhưng validation vẫn là heuristic. Kiểm tra token hiện tại chủ yếu phát hiện token gốc bị mất, không chứng minh mọi thay đổi đều đúng nghĩa.

### 5.3. VLM tùy chọn

Nguồn: [vision.py](../rag_kaggle/vision.py), [parsers.py](../rag_kaggle/parsers.py).

- Phân loại ảnh sơ bộ bằng caption, OCR labels, từ khóa và dấu mũi tên.
- Với visual phù hợp, Qwen2.5-VL tạo JSON flowchart (`nodes`, `edges`) hoặc chart (`observations`, `summary`).
- Chạy sau khi correction document hoàn tất; unload correction model trước khi load VLM.
- Nhận OCR đã sửa, caption, section và đuôi đúng một trang trước nếu có page metadata.
- Chèn block mô tả ngay sau image block trong reading order.
- Phân biệt quan sát và nhận xét suy luận khi render text để retrieval.
- Output sai có retry hữu hạn rồi đánh dấu `needs_review`; lỗi model ghi warning.

Baseline và notebook hiện tại tắt VLM. Code validation JSON chủ yếu kiểm tra một số kiểu trường; chưa kiểm chứng đầy đủ node/edge hoặc tính đúng của dữ liệu hình.

## 6. Mô hình dữ liệu và hồ sơ tài liệu

Nguồn: [models.py](../rag_kaggle/models.py), [profile.py](../rag_kaggle/profile.py), [relationships.py](../rag_kaggle/relationships.py).

### 6.1. Canonical data model

- `ParsedDocument`: định danh, tên file, loại file, hash, blocks, relationships, metadata và parser version.
- `Block`: loại nội dung, text, section, page/sheet/range, bbox, asset, raw content, reading order và confidence.
- `Chunk`: đơn vị retrieval, block IDs, ordinal, vị trí nguồn, entities, keywords, metadata và version.
- `SearchHit`: chunk cùng dense rank, sparse rank, RRF score và rerank score.
- `StageStatus`: stage/status/error/message/retryable/duration.
- `QueryPlan`: câu hỏi gốc, semantic queries, keyword query và filters.

Các loại block: heading, text, caption, table, image, chart, flowchart, KPI, OCR page và VLM.

### 6.2. Quan hệ giữa các block

Đã xây dựng và lưu các loại quan hệ:

- `next_in_reading_order`: thứ tự đọc.
- `belongs_to_section`: liên kết tới heading.
- `captioned_by`: bảng/hình với caption lân cận.
- `referenced_by`: câu văn nhắc đến hình/bảng theo số.
- `visualizes`: chart với vùng table/KPI nguồn trong sheet.
- `continues`: bảng nối tiếp trên hai trang kế nhau, có kiểm tra hình dạng/vị trí heuristic.
- `derived_from`: mô tả VLM với image gốc.

Caption và tối đa một số câu tham chiếu được gắn vào metadata để đi cùng chunk. Bảng tiếp trang có thể kế thừa header. Đây là graph quan hệ trong document, chưa phải hệ thống GraphRAG suy luận graph tổng quát.

### 6.3. Document profile

Sau correction và xây quan hệ, code trích title, số hiệu, ngày ban hành, đơn vị ban hành, người ký, keywords, entities và name variants bằng regex/thống kê/heuristic, không dùng LLM.

Entities có nhóm `person`, `location`, `other`; lưu canonical spelling, aliases, key bỏ dấu và số lần xuất hiện. Chunk nhận những entity thực sự được nhắc tới và keywords của document. Profile được lưu trong SQLite, một phần đưa vào Qdrant payload/BM25 và prompt trả lời.

Đây chưa phải NER model hoặc trích xuất metadata được bảo đảm chính xác, đặc biệt với tên viết hoa, người ký và tài liệu không theo mẫu hành chính.

## 7. Chunking theo cấu trúc

Nguồn: [chunking.py](../rag_kaggle/chunking.py).

### 7.1. Text

Gộp các đoạn liên tiếp cùng section/sheet đến `text_chunk_chars` (baseline 1.800 ký tự), không overlap. Khi không có section/sheet, page tham gia khóa nhóm. Đoạn riêng vượt ngưỡng được chia theo paragraph, dòng/bullet, câu rồi khoảng trắng. Heading đã xuất hiện trong section header và caption đã gắn với hình/bảng không tạo chunk trùng riêng.

### 7.2. Bảng

Một structured table tạo một chunk. Bảng vừa ngưỡng 6.000 ký tự giữ Markdown đầy đủ. Bảng lớn tạo preview gồm header, 5 hàng và 5 cột đầu theo baseline, danh sách tên cột và đường dẫn file đầy đủ.

Rows đầy đủ vẫn có trong block SQLite. File `tables/<document_id>/<block_id>.xlsx` được xuất để tải; cell dạng `=...` được lưu như text, tránh biến text tài liệu thành công thức khi export.

### 7.3. Hình, chart, flowchart và KPI

Chunk visual chứa OCR/text mô tả, caption và câu tham chiếu; chỉ chia khi vượt ngưỡng visual 6.000 ký tự. KPI và các loại block khác đi qua nhánh chunk đơn theo text budget. Vì vậy nguyên tắc một visual/một chunk có ngoại lệ khi nội dung quá dài.

Header dùng cho embedding gồm `Section`, `Sheet`, `Range` nếu có. Tên file, số trang và loại chunk nằm trong metadata; BM25 bổ sung một số thông tin này vào text sparse riêng.

## 8. Embedding và lưu trữ

Nguồn: [storage.py](../rag_kaggle/storage.py), [parsers.py](../rag_kaggle/parsers.py).

### 8.1. Dense index

- Dùng SentenceTransformer; baseline BAAI/bge-m3.
- Normalize embedding, Qdrant named vector `dense`, distance cosine.
- Hỗ trợ revision, device và query instruction riêng.
- Cache document embedding trong SQLite theo model/revision/text; query không dùng cache này.
- CUDA OOM: giảm batch size và retry hữu hạn.
- Upsert Qdrant theo ID ổn định, batch 64 point.
- Không cho trộn embedding model trong một index; query encoder phải khớp model index.
- Adapter có fallback model cấu hình được, nhưng baseline/notebook đặt fallback bằng model chính để lỗi load không đổi sang model khác.

### 8.2. BM25

BM25Okapi được dựng lại trên toàn corpus sau indexing và lưu `bm25.pkl`. Tokenizer lowercase + Unicode NFC theo từ/âm tiết, giữ số/mã và mở rộng token có dấu nối/chấm/slash. Sparse text gồm chunk content, tên file, title, entities (cả dạng bỏ dấu) và keywords.

### 8.3. SQLite và artifacts

SQLite chứa documents, blocks, chunks, relationships, document_entities, document_keywords, ingestion_status, settings và embedding_cache. Payload JSON giữ metadata chi tiết, gồm rows gốc của bảng.

Artifacts runtime gồm:

```text
work_dir/
├── source/                 # bản copy nguồn có hash
├── assets/                 # ảnh trích xuất / trang scan render
├── parsed/                 # document JSON sau parse
├── tables/                 # Parquet và bảng lớn .xlsx
├── qdrant/                 # vector store local
├── metadata.db
├── bm25.pkl
├── manifests/              # ingestion manifest theo run
├── manifest.json
├── config.json
├── corpus_manifest.json    # được tạo khi freeze
└── logs/                   # log worker OCR

session_dir/
└── query_traces.jsonl
```

Parquet được ghi khi bật cấu hình và dependency cho phép; nếu thiếu thư viện/engine, có thể bỏ qua hoặc ghi warning. Tính toán bảng hiện dùng `metadata.rows` của block, không cần đọc lại Parquet.

## 9. Freeze, export và restore corpus

Nguồn: [pipeline.py](../rag_kaggle/pipeline.py).

### 9.1. Freeze và export

- Từ chối freeze corpus không có chunk.
- Ghi corpus ID, cờ frozen, record counts, collection, phiên bản parser/chunker và chunk config.
- Ghi hợp đồng embedding: model, revision, dimension, normalization và query instruction.
- Commit metadata và đóng Qdrant local trước khi zip.
- Bundle chứa metadata, Qdrant, BM25, manifest/config, tables và assets; parsed mặc định được đưa vào.
- Source documents mặc định không được đưa vào; có thể bật `include_source_documents`.
- Tạo SHA-256 inventory trong `checksums.json`; bundle phải được ghi ngoài work directory.

### 9.2. Restore

Giải nén vào staging, kiểm tra đường dẫn, file bắt buộc, artifact schema, frozen flag, inventory và checksum. Model/revision/query instruction được kế thừa hoặc đối chiếu với cấu hình. Sau đó đóng store cũ, đổi directory, mở store mới và kiểm tra manifest với database; có nhánh backup/rollback nếu kích hoạt thất bại.

Traces query được đặt ở session directory riêng. Có alias tương thích `export_artifacts`, `restore_artifacts`, `from_artifacts`.

Checksum giúp phát hiện file thay đổi hoặc thiếu; không phải chữ ký xác thực nguồn bundle. BM25 được deserialize bằng pickle, nên bundle cần đến từ nguồn tin cậy. Giới hạn dung lượng/số entry của ZIP đầu vào tài liệu chưa được áp dụng tương tự trong nhánh restore corpus.

## 10. Truy xuất và mở rộng context

Nguồn: [retrieval.py](../rag_kaggle/retrieval.py), [generation.py](../rag_kaggle/generation.py), [pipeline.py](../rag_kaggle/pipeline.py).

1. Kiểm tra corpus và chuẩn hóa query.
2. Nếu bật rewrite, LLM tạo semantic queries, keyword query và filters. Lỗi thì fallback câu hỏi gốc. Query gốc luôn được tìm kiếm.
3. Kiểm tra rewrite giữ số/mã dạng số và chỉ chấp nhận một số filter có căn cứ trong query.
4. Dense + BM25 tìm ứng viên, có filter theo document, file, sheet, chunk type hoặc entity ở tầng index.
5. Entity trong query khớp document profile theo key bỏ dấu/không phân biệt hoa thường; thêm ranking giới hạn trong document đó để boost mềm.
6. RRF cộng `1 / (rrf_k + rank)` từ các ranking.
7. Reranker chấm query–chunk, sắp xếp và lấy top-k.
8. Kiểm tra evidence đủ mạnh; có thể từ chối trước generation.
9. Text hit thêm chunk text lân cận cùng document/section; bảng preview được render lại từ rows gốc trong budget.
10. Thêm caption, câu tham chiếu, vùng bảng nguồn chart, continuation và derived block qua graph.
11. Loại context trùng, giới hạn ký tự, sắp theo document/reading order và cấp `SOURCE_x`.

Baseline: dense top 30, BM25 top 30, fused top 30, rerank top 8, `rrf_k=60`, 1 neighbor mỗi phía, tối đa 12.000 ký tự cho phần bảng và 28.000 ký tự context generation.

Bảng lớn được mở rộng **đến budget**, không bảo đảm toàn bộ bảng đều vào prompt. File `.xlsx` và rows trong SQLite vẫn giữ dữ liệu đầy đủ. Chunk preview có thể bỏ sót nội dung chỉ xuất hiện ở hàng/cột sâu khi retrieval chưa tìm đúng bảng.

## 11. Tính toán bảng và sinh câu trả lời

### 11.1. Structured computation

Nguồn: [computation.py](../rag_kaggle/computation.py).

Hỗ trợ nhận diện `sum`, `avg`, `max`, `min`, `count` bằng từ khóa Việt/Anh. Code chọn cột số theo tỷ lệ parse được và độ khớp tên cột, lọc một số giá trị categorical được nhắc trong query, bỏ dòng tổng để tránh cộng hai lần, rồi thực hiện phép tính bằng Python.

Output có giá trị, phép tính/công thức, cột, điều kiện, rows dùng và vị trí nguồn; `max/min` kèm dòng tương ứng. Kết quả trở thành một nguồn context riêng.

Hiện chỉ chọn một kết quả tốt nhất từ các table/KPI block của hit đã truy xuất. Chưa có SQL planner, join nhiều bảng, group-by tổng quát hoặc thực thi Python do LLM sinh. Hàm parse số chưa quy đổi đơn vị “triệu/tỷ” thành hệ số; logic phụ thuộc cấu trúc header/rows.

### 11.2. Generation và citation

Nguồn: [generation.py](../rag_kaggle/generation.py).

- Dùng causal LM qua Transformers; baseline Qwen2.5-7B-Instruct với quantization 4-bit.
- Load model/tokenizer lazy; hỗ trợ revision/device và token usage.
- Prompt yêu cầu chỉ trả lời từ context, từ chối khi thiếu bằng chứng và dẫn `[SOURCE_x]`.
- Context có tên file, page/section, title, số hiệu, ngày, người ký và đường dẫn bảng đầy đủ.
- Ưu tiên context kết quả tính toán trước context retrieval.
- Parse JSON answer/source IDs, có fallback plain text.
- Backend chỉ ánh xạ ID hợp lệ sang citation; không tin file/page do model tự viết.
- Citation theo file + page/range trang, sheet + cell range hoặc section.
- Trả answer, citations, source IDs, trạng thái citation/refusal, cảnh báo, context preview, computation và latency/trace.

UI hiển thị lịch sử hội thoại nhưng `ask()` chỉ nhận query hiện tại; chưa dùng chat history cho truy xuất hoặc generation nhiều lượt.

## 12. Guardrails và quan sát

Nguồn: [guardrails.py](../rag_kaggle/guardrails.py), [tracing.py](../rag_kaggle/tracing.py).

- Phát hiện các pattern prompt injection tiếng Việt/Anh trong query/context và ghi cảnh báo.
- Vô hiệu hóa một số token chat template và marker nguồn giả trong context.
- Từ chối khi không có hit hoặc điểm evidence thấp theo cấu hình.
- Kiểm tra số trong answer so với prompt context và cảnh báo số thiếu căn cứ.
- Mask email, số điện thoại và một số chuỗi giống ID/card khi ghi trace.
- Query trace JSONL chứa plan, filters, ranking, hits, computation, usage, answer, citations, warnings và latency từng stage.
- Ingestion report có run ID, thời gian bắt đầu/kết thúc, elapsed time, documents/skips/errors/statuses và stats.
- Parser ghi timing theo trang PDF, block DOCX, sheet Excel, render/OCR/VLM.
- Notebook in thời gian start/end/elapsed của từng code cell bằng wrapper `try/finally`.

Guardrails hiện là regex, threshold và kiểm tra heuristic. Cảnh báo citation/số không tự loại bỏ mọi câu trả lời sai; PII masking chủ yếu áp dụng cho trace được ghi, không phải toàn bộ dữ liệu corpus hoặc nội dung trả về.

## 13. Giao diện Gradio

Nguồn: [ui.py](../rag_kaggle/ui.py).

- **Ingestion**: upload nhiều file/ZIP, reset option, nút xử lý, streaming log stage, report JSON và tải bundle.
- **Chat**: hỏi đáp, filter file/type, citation và warning.
- **Chi tiết retrieval**: dense/BM25 rank, RRF/rerank score và preview.
- **Context và nguồn**: preview context, gallery ảnh nguồn, tải bảng lớn `.xlsx`, trace JSON.
- **Evaluation**: upload JSONL, tùy chọn chạy answer generation và xem metrics.
- **System**: stats, danh sách document, ingestion statuses và upload/restore bundle trong chat runtime.

Có `launch_demo`, `launch_ingestion_demo`, `launch_chat_demo`; queue mặc định concurrency 1. UI hỗ trợ kiểu chatbot tương thích Gradio và whitelist đường dẫn runtime để phục vụ assets.

File chooser ingestion hiện liệt kê PDF/DOCX/XLSX/XLSM/XLS/ZIP; API discovery còn hỗ trợ XLTX/XLTM. Chưa có web backend triển khai riêng, tài khoản người dùng hoặc quản lý quyền theo tài liệu.

## 14. Kaggle notebooks và tài nguyên

Nguồn: [01_ingestion.ipynb](../notebooks/01_ingestion.ipynb), [02_retrieve_answer.ipynb](../notebooks/02_retrieve_answer.ipynb), [hardware.py](../rag_kaggle/hardware.py), [build_kaggle_notebook.py](../scripts/build_kaggle_notebook.py).

### 14.1. Những gì notebook đã tự động hóa

- Clone/pull source khi cần, import package và cài `requirements-kaggle.txt`.
- Cấu hình model và đường dẫn input ngay trong cell.
- Tạo môi trường Paddle riêng; thử `venv`, fallback `virtualenv`; cấu hình Paddle GPU `3.2.1`/`cu126` trong notebook hiện tại.
- Nếu cài OCR thất bại, in lỗi và disable OCR; người chạy cần kiểm tra scan không bị mất nội dung.
- Báo tài nguyên, phân bổ device, thống kê profile/entities/chunks và bảng preview.
- Ingest rồi freeze/export, hiển thị link tải bundle.
- Restore corpus và mở Chat UI trong notebook retrieval.
- Script tạo notebook dùng cell ID ổn định; tái sinh notebook sẽ ghi đè file notebook.

Đường dẫn dataset đang hardcode trong notebook local; cần đổi khi chạy với dataset khác. Notebook retrieval hiện nén một folder corpus bằng `shutil.make_archive(..., root_dir=CORPUS_SRC)` trước restore. Folder đó phải có cấu trúc corpus ở root; nếu đã có bundle ZIP thì có thể gán trực tiếp `CORPUS_BUNDLE`.

### 14.2. Phân bổ T4 x2 theo helper hiện tại

Ingestion: GPU 1 cho OCR worker; GPU 0 cho correction, VLM tùy chọn và embedding ở các giai đoạn tương ứng. OCR/correction được unload trước dense indexing; VLM document chạy sau correction.

Retrieval: GPU 0 cho Qwen generation/rewrite; GPU 1 cho query embedding; CPU cho reranker, Qdrant, BM25 và context expansion.

Có helper cho một GPU/CPU và helper full mode cũ. Phân bổ device không tự chứng minh mọi tổ hợp model vừa VRAM, đặc biệt một GPU có thể phải chứa model chồng lấp. `inspect_resources()` báo GPU/VRAM, RAM nếu có psutil và disk; `suggest_model_upgrades()` chỉ gợi ý, không tự thay model.

## 15. Cấu hình và dependency

Nguồn: [config.py](../rag_kaggle/config.py), [baseline.yaml](../configs/baseline.yaml), [requirements-kaggle.txt](../requirements-kaggle.txt).

Config được tách theo ingestion, parsing, OCR correction, vision, chunking, retrieval, generation, guardrails, runtime và artifacts. Có `from_dict`, `from_yaml`, `to_dict`, tạo directory và từ chối key không biết.

Baseline bật OCR, correction, reranker, relationship expansion, entity boost, structured computation và guardrails; tắt VLM, query rewrite, glossary, sửa bảng OCR, index dữ liệu ẩn và đóng gói source documents.

`PipelineConfig()` thuần khác preset YAML ở một số giá trị: correction mặc định tắt, batch embedding 8, fused top 24, rerank top 7, generation 700 token; YAML bật correction, batch 4, fused top 30, rerank top 8, generation 1.024 token. Notebook tiếp tục override theo model/path/device đã chọn.

Dependency gồm Qdrant client, SentenceTransformers, Transformers, Accelerate, BitsAndBytes, FlagEmbedding, rank-bm25, PyMuPDF, python-docx, openpyxl, pandas, python-calamine, pyarrow, Pillow, Gradio và PyYAML, cùng thư viện hỗ trợ model. PaddleOCR/Paddle được cài trong môi trường riêng bởi ingestion notebook.

## 16. Evaluation và kiểm thử đã có

### 16.1. Evaluation

Nguồn: [evaluation.py](../rag_kaggle/evaluation.py), [dataset.sample.jsonl](../evaluation/dataset.sample.jsonl).

Dataset nhận question, expected answer/document/location/block IDs, content type, category và no-answer. File mẫu có bốn tình huống: tra bảng, tổng hợp bảng, logic flowchart, câu hỏi không có đáp án; đây là mẫu schema, không phải corpus benchmark đã chạy.

Metrics hiện có:

- Retrieval: recall@k theo việc có ít nhất một hit phù hợp, MRR, hit rate theo content type và block coverage.
- Answer: correctness bằng substring/số, numeric exact match, faithfulness proxy, citation accuracy, refusal accuracy, false refusal rate.
- Phân nhóm theo category; latency median/max, parse success rate, issue OCR/VLM, indexing throughput và peak VRAM nếu có GPU.
- Xuất report JSON vào `work_dir/evaluation`.

Các metric correctness/faithfulness là heuristic/proxy, không phải đánh giá semantic bởi chuyên gia. Evaluation retrieval gọi retriever trực tiếp; các boost/rewrite do `ask()` chuẩn bị không được áp dụng giống hệt ở bước đo retrieval riêng. Report evaluation ghi trong work directory, cần lưu ý khi vận hành corpus frozen.

### 16.2. Bộ test tự động

Nguồn: [test_core.py](../tests/test_core.py), [test_ocr_correction.py](../tests/test_ocr_correction.py), [test_structure_pipeline.py](../tests/test_structure_pipeline.py).

Đã có test cho chunking, metadata round trip, numeric/text utilities, OCR output/worker protocol, config/device/mode, upload gate, caption/reference/continuation, computation, guardrails, query/answer parsing, vision helpers, evaluation metrics, correction ordering/retry/abort/context reset, table correction, heading propagation và deferred vision.

Test tích hợp có các scenario parse/index/ask/compute/export/restore, giữ phiên bản file, checksum hỏng, bảng lớn xuất workbook, full-table context, entity boost và legacy schema. Các scenario này dùng fake embedding/LLM cho phần model, không đánh giá chất lượng BGE/Qwen thật.

Kết quả kiểm tra local ngày 08/10/2026:

```text
Lệnh: python3 -m unittest discover -s tests
Ran 67 tests
OK (skipped=5)
```

62 test thực thi thành công; 5 test tích hợp bị skip do thiếu dependency parser/index theo điều kiện test. Có `ResourceWarning` về stream chưa đóng ở OCR subprocess. Kết quả trên chưa xác nhận model GPU, OCR thật, reranker thật hoặc toàn bộ workflow Kaggle.

### 16.3. Smoke test model thật

[smoke.py](../rag_kaggle/smoke.py) có `run_smoke_test()` tạo DOCX/XLSX/PDF/scan mẫu trong work directory riêng, kiểm tra environment, OCR, ingestion, dense index, retrieve/rerank, LLM answer, computation và VRAM. Kết quả theo stage là PASS/WARN/FAIL/SKIP. Chưa chạy smoke test GPU trong lần tổng hợp này.

## 17. Bản đồ source và tài liệu

- `rag_kaggle/__init__.py`: public API pipeline/config/device/resource helpers.
- `config.py`, `models.py`: cấu hình, version và schema dữ liệu.
- `ingestion.py`, `parsers.py`: discovery/validation và đọc tài liệu.
- `paddleocr_vl.py`, `ocr_correction.py`, `vision.py`: OCR, sửa lỗi, VLM.
- `entities.py`, `profile.py`, `relationships.py`: tên riêng, hồ sơ và graph block.
- `chunking.py`: chunking theo cấu trúc.
- `storage.py`: SQLite, embedding/cache, Qdrant, BM25.
- `retrieval.py`: fusion/rerank/context/citation.
- `computation.py`, `generation.py`, `guardrails.py`: tính toán, trả lời và kiểm tra evidence.
- `pipeline.py`: orchestration ingestion/ask/evaluate/freeze/export/restore.
- `tracing.py`, `hardware.py`: trace/timing và tài nguyên/device.
- `ui.py`, `smoke.py`: demo và kiểm tra end-to-end trên GPU.
- `notebooks/`: hai notebook chạy hai giai đoạn.
- `configs/`: preset Kaggle baseline.
- `evaluation/`: dataset JSONL mẫu.
- `tests/`: ba file test.
- `scripts/build_kaggle_notebook.py`: sinh notebook.
- `scripts/render_architecture_diagrams.py`: dựng ảnh sơ đồ bằng Pillow.
- [README.md](../README.md): hướng dẫn chạy/API/bundle.
- [Architecture.md](../Architecture.md): kiến trúc và định hướng, có các phần thiết kế cũ cần đối chiếu.
- [data-flow.md](data-flow.md): giải thích luồng text/table/visual theo implementation.
- `docs/diagrams/`: sáu hình về overview, parsing, chunking, retrieval, storage và GPU lifecycle; sơ đồ nên được đối chiếu với code trước khi dùng làm đặc tả.
- `parse-diagram.png`: ảnh sơ đồ bổ sung ở root.
- `.gitignore`: bỏ qua một số runtime artifacts, môi trường local, secrets và raw data.

## 18. Cách dùng các chức năng đã triển khai

### 18.1. Tạo corpus

```python
from pathlib import Path
from rag_kaggle import IngestionPipeline, PipelineConfig, configure_ingestion_devices

config = PipelineConfig.from_yaml("configs/baseline.yaml")
config.work_dir = Path("/kaggle/working/rag_ingestion")
configure_ingestion_devices(config)
# Khi dùng worker riêng, đặt config.parsing.ocr_python theo môi trường OCR đã cài.
pipeline = IngestionPipeline(config)
report = pipeline.ingest_sources(["/kaggle/input/my-documents"], reset=True)
if not report["ok"]:
    raise RuntimeError(report["errors"])
bundle = pipeline.export_corpus_bundle("/kaggle/working/corpus_bundle.zip")
```

### 18.2. Phục hồi và hỏi đáp

```python
from pathlib import Path
from rag_kaggle import PipelineConfig, RetrievalAnswerPipeline, configure_retrieval_devices

config = PipelineConfig.from_yaml("configs/baseline.yaml")
config.work_dir = Path("/kaggle/working/rag_corpus")
config.runtime.session_dir = "/kaggle/working/rag_session"
config.retrieval.dense_model = None
configure_retrieval_devices(config)
pipeline = RetrievalAnswerPipeline(config)
pipeline.restore_corpus_bundle("/kaggle/input/my-corpus/corpus_bundle.zip")
result = pipeline.ask("Quy trình phê duyệt gồm những bước nào?")
print(result["answer"])
print(result["citations"])
```

Đổi embedding yêu cầu xây lại corpus. Reranker và generation có thể đổi độc lập nếu tương thích adapter và phần cứng. Các notebook là nơi cung cấp đầy đủ bước setup Kaggle; ví dụ API trên giả định dependencies/model đã sẵn sàng.

## 19. Phạm vi đã làm và những phần chưa được xác nhận

Project đã có implementation cho một baseline RAG đa định dạng với persistence và demo hỏi đáp. Các khả năng mở rộng đã có code nhưng mặc định tắt gồm VLM, query rewrite, glossary và sửa bảng OCR.

Trong repository được đối chiếu, chưa có kết quả benchmark model thật/corpus thực chứng minh độ chính xác, tốc độ, peak VRAM hoặc chất lượng OCR ở quy mô sử dụng cụ thể. Notebook không lưu output thực thi để dùng làm bằng chứng chạy thành công. Bộ test local trong lần này còn skip các test tích hợp phụ thuộc môi trường.

Chưa có implementation riêng cho multi-user/authentication, phân quyền tài liệu, dịch vụ production/distributed Qdrant, API server độc lập, semantic NER chuyên dụng, truy vấn/tính toán nhiều bảng tổng quát hoặc hội thoại có memory. Những nội dung này không nên được xem là thành quả đã hoàn tất chỉ vì xuất hiện trong định hướng kiến trúc.
