# Kiểm tra luồng xử lý dữ liệu: trích xuất -> chunk -> embedding -> Qdrant -> retrieve

Cách kiểm tra: đọc code từng bước, rồi chạy tài liệu mẫu (DOCX, PDF, XLSX) qua pipeline thật và in kết quả từng giai đoạn.
Chạy lại trên tài liệu của bạn (không cần GPU, OCR tắt):

```bash
python scripts/trace_data_flow.py file.pdf file.docx file.xlsx --query "câu hỏi"
```

Giới hạn của lần kiểm tra này: chưa chạy trên Kaggle, nên chưa đo được chất lượng của LLM hiệu đính, PaddleOCR-VL, bge-m3 và
reranker thật. Embedding trong script là bag-of-words băm, chỉ dùng để xem nội dung chunk, payload, BM25 và cách ghép context.

## 1. Luồng hiện tại theo từng bước

| Bước | Văn bản | Bảng | Hình |
|---|---|---|---|
| **Trích xuất** | PDF số: khối text của PyMuPDF, heading theo cỡ chữ. PDF scan (< 80 ký tự text): render 220 dpi -> PaddleOCR-VL theo layout -> **LLM hiệu đính** (mới). DOCX: đoạn + style heading. XLSX: text/KPI theo vùng | PDF số: `find_tables`. Scan: bảng HTML/Markdown của OCR. DOCX: python-docx (ô gộp lặp lại, bảng lồng nhau). XLSX: vùng cách nhau bởi dòng/cột trống, đọc cả công thức và giá trị cache | Ảnh >= 220x120 -> OCR (+ VLM nếu bật và là flowchart/chart) -> phân loại ảnh. Chart DOCX/XLSX đọc từ XML/cache |
| **Quan hệ** | `referenced_by` (câu nhắc "Bảng 2") | caption, `continues` (bảng nối trang cùng số cột) | caption, `derived_from` (mô tả VLM) |
| **Chunk** (mới) | 1 đoạn = 1 chunk; đoạn ngắn gộp với đoạn kế tiếp; đoạn dài tách ở ranh giới câu | Nhỏ: 1 chunk Markdown. Lớn (> 60 hàng hoặc > 6.000 ký tự): 1 chunk xem trước + file `.xlsx` | 1 ảnh = 1 chunk (caption + OCR + đoạn tham chiếu) |
| **Embedding** | `chunk.content` = tiền tố (Document/Section/Page/Content type/Entities) + nội dung; `bge-m3`, chuẩn hóa, cache theo sha1 của text | giống văn bản | giống văn bản |
| **Lưu** | Qdrant: vector `dense` (cosine) + payload (gồm `content`, `entities`, `keywords`). SQLite: documents, blocks, sections (`parents`), chunks, relationships. BM25 pickle | thêm: `blocks.rows` giữ bảng đầy đủ; `tables/*.parquet` và `tables/*/*.xlsx` | `assets/` giữ file ảnh |
| **Retrieve** | dense top 30 + BM25 top 30 (+ `entity_search` khi câu hỏi nêu tên) -> RRF (k=60) -> top 30 -> reranker -> top 8 -> guardrail điểm | thêm: bảng lớn trúng -> lấy các dòng khớp câu hỏi từ `blocks.rows` | giống văn bản |
| **Context** | Section nếu <= 6.000 ký tự, nếu không thì chunk trúng + chunk liền trước/sau (mới) + block liên quan (caption, đoạn tham chiếu...) | + `TABLE ROWS` + `structured computation` đọc `blocks.rows` | + ảnh trả về UI qua `asset_paths` |

## 2. Phát hiện, đã kiểm chứng

| # | Phát hiện | Cách kiểm chứng | Trạng thái |
|---|---|---|---|
| 1 | Bảng bị cắt thành nhiều chunk theo 15 hàng x 8 cột; mỗi mảnh embed riêng, câu hỏi về cả bảng phải ghép lại từ nhiều chunk | đọc `chunking.py` cũ | **Đã sửa**: 1 bảng = 1 chunk, hoặc preview + xlsx |
| 2 | `split_text` cắt **giữa câu** với text xuống dòng cứng (kiểu PDF): 5/6 đoạn tách ra không kết thúc ở cuối câu | chạy lại với đoạn 2.929 ký tự wrap 80 cột | **Đã sửa** (`_split_unit` nối dòng bị wrap, giữ dòng ngắn kiểu danh sách); có test |
| 3 | Với thiết kế "chỉ giữ preview", dòng nằm xa không còn từ khóa để tìm ra bảng (ví dụ `SP123` trong bảng 150 dòng; danh sách khóa bị cắt ở 600 ký tự) | test e2e trả về `[]` | **Đã sửa**: BM25 index thêm toàn bộ giá trị cột đầu của bảng lớn (văn bản embed không đổi). Test fail khi tắt fix |
| 4 | Text OCR không được hiệu đính, tên riêng sai dấu rải rác và không nhất quán | đọc `_parse_scanned_page` | **Đã sửa** (mục 3 dưới đây) |
| 5 | Không có metadata cấp tài liệu (số hiệu, ngày, người ký, người được nhắc) trong DB/Qdrant | đọc `ChildChunk.payload` | **Đã sửa**: profile + `entities`/`keywords` ở chunk |
| 6 | PDF số: header/footer lặp lại mỗi trang **không bị lọc** (chỉ lọc ở nhánh OCR qua `OCR_SKIP_LABELS`), thành chunk/nhiễu BM25 | đọc `_parse_digital_page` | **Còn mở** |
| 7 | Ảnh **không có chữ OCR và không có VLM bị bỏ hẳn** (`_add_visual` trả `None`). Vì vậy "1 ảnh = 1 chunk" hiện chỉ đúng với ảnh có chữ | đọc `parsers.py:235` | **Còn mở** (cần quyết định, xem mục 4) |
| 8 | `tables/*.parquet` được ghi nhưng **không được đọc ở bước truy vấn** (`computation.py` đọc `blocks.rows` từ SQLite); chỉ `table_path` trong metadata trỏ tới nó | grep `read_parquet`/`table_path` | **Còn mở** (tốn dung lượng bundle, chưa có người dùng) |
| 9 | Header bảng luôn giả định là hàng đầu (`rows[0]`) ở chunker, parquet, computation. Bảng không có hàng tiêu đề sẽ cho "Tên các cột" sai | grep `rows[0]` | **Còn mở** |
| 10 | Qdrant payload chứa cả `content` trong khi retrieval chỉ cần `chunk_id` (nội dung đọc lại từ SQLite): bundle chứa text hai lần | đọc `dense_search` (`with_payload=["chunk_id"]`) | **Còn mở**, ưu tiên thấp |
| 11 | BM25 dùng token âm tiết, không tách từ, không bỏ stopword; tiền tố `Document:/Section:/Content type:` làm mọi chunk có chung các token đó (IDF thấp nên ảnh hưởng nhỏ) | chạy `tokenize_vi` trên chunk | **Còn mở**, cần đo bằng `pipeline.evaluate` |

## 3. Hiệu đính OCR và metadata (ý 1, 2, 3)

Cơ chế ngữ cảnh: mỗi lần gọi LLM được dựng lại từ đầu, không dùng lại hội thoại cũ.

```text
system + skill (cố định)  ~ 400 token
THUẬT NGỮ THỐNG NHẤT      <= memory_max_chars (800 ký tự)
ĐOẠN TRƯỚC (đã sửa)       <= context_chars (1.200 ký tự), tối đa 1 trang trước
ĐOẠN CẦN SỬA              <= segment_chars (1.500 ký tự)
```

Nên kích thước prompt bị chặn, bất kể tài liệu dài bao nhiêu. "Verify hậu OCR với đúng 1 trang phía trước" và "đoạn liền trước"
là cùng một cơ chế: ngữ cảnh lấy từ các đoạn đã sửa gần nhất, không bao giờ xa hơn `max_pages_back` (mặc định 1) trang.

Về băn khoăn "cần context window và độ khôn của LLM theo kịp" của bạn: phần glossary được thiết kế để **không** phụ thuộc vào
độ khôn của LLM.

- Glossary chỉ nhận thực thể mà chuỗi đó **thực sự có trong đoạn đã sửa** (LLM bịa ra tên thì bị bỏ).
- Hai cách viết chỉ khác dấu (Hùng/Hưng) **không bị gộp** (có thể là hai người khác nhau); chỉ được báo là "có thể bị OCR đọc sai".
- Bản sửa bị loại và giữ nguyên bản OCR nếu: đổi bất kỳ chữ số nào, đổi độ dài > 35%, giống bản gốc < 60%, rỗng, hoặc lặp lại prompt.
- Lỗi GPU/OOM liên tiếp 3 lần thì tự tắt bước này cho phần còn lại, không làm hỏng tài liệu.
- Bảng, ảnh, công thức, code không bao giờ đi qua LLM. Bản gốc luôn nằm ở `block.metadata.ocr_original_text`.

Kết quả của mỗi tài liệu (số đoạn sửa/loại/lỗi, lý do bị loại, glossary) nằm ở `documents.payload_json -> metadata.ocr_correction`
và trong `report["documents"][i]["ocr_correction"]`. Hãy xem `reject_reasons` ở lần chạy đầu trên Kaggle để biết LLM có đang bị loại quá nhiều không.

## 4. Quyết định cần bạn xem

1. **Ảnh không có chữ (phát hiện 7)**: giữ nguyên (không index ảnh trang trí/ảnh chụp) hay tạo chunk chỉ gồm caption + vị trí? Tạo chunk
   sẽ đúng "1 ảnh = 1 chunk" nhưng có thể thêm nhiều chunk vô nghĩa từ logo, ảnh minh họa.
2. **Bỏ overlap giữa các chunk văn bản** (trước đây 240 ký tự): tính liên tục giờ nhờ chunk liền trước/sau lúc retrieve. Chưa đo ảnh hưởng
   tới recall; nên chạy `pipeline.evaluate` với bộ câu hỏi thật trước và sau khi ingest lại.
3. **Mặc định hiệu đính bật trong `baseline.yaml`** (`correction.enabled: true`) nhưng tắt trong dataclass (`CorrectionConfig.enabled = False`).
   Notebook bật theo `CORRECTION_MODEL`.
4. **Sơ đồ `docs/diagrams/03-parent-child.png` chưa vẽ lại** (còn mô tả chiến lược cũ).
