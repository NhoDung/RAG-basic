from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "diagrams"
OUTPUT.mkdir(parents=True, exist_ok=True)

WIDTH = 1600
HEIGHT = 900

COLORS = {
    "background": "#F6F3EA",
    "ink": "#17202A",
    "muted": "#607080",
    "line": "#52606D",
    "input": "#D8EAFE",
    "input_border": "#2878B5",
    "process": "#FFF0C2",
    "process_border": "#C87500",
    "model": "#F8DCE8",
    "model_border": "#B8336A",
    "store": "#DDF2DF",
    "store_border": "#25814B",
    "output": "#E8E0F8",
    "output_border": "#6741A5",
    "neutral": "#E8ECEF",
    "neutral_border": "#667685",
    "danger": "#FADBD8",
    "danger_border": "#B03A2E",
    "white": "#FFFFFF",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(42, bold=True)
SUBTITLE_FONT = load_font(22)
SECTION_FONT = load_font(25, bold=True)
BOX_TITLE_FONT = load_font(23, bold=True)
BOX_BODY_FONT = load_font(19)
SMALL_FONT = load_font(17)


class Diagram:
    def __init__(self, title: str, subtitle: str):
        self.image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
        self.draw = ImageDraw.Draw(self.image)
        self.draw.text((70, 42), title, font=TITLE_FONT, fill=COLORS["ink"])
        self.draw.text((72, 98), subtitle, font=SUBTITLE_FONT, fill=COLORS["muted"])
        self.draw.line((70, 140, WIDTH - 70, 140), fill="#CBD2D8", width=2)

    def section(self, xy, size, title, fill="#EFECE3"):
        x, y = xy
        w, h = size
        self.draw.rounded_rectangle(
            (x, y, x + w, y + h), radius=24, fill=fill, outline="#CED3D6", width=2
        )
        self.draw.text((x + 25, y + 18), title, font=SECTION_FONT, fill=COLORS["ink"])

    def box(self, xy, size, title, body="", kind="process", radius=18):
        x, y = xy
        w, h = size
        fill = COLORS[kind]
        border = COLORS[f"{kind}_border"]
        self.draw.rounded_rectangle(
            (x, y, x + w, y + h), radius=radius, fill=fill, outline=border, width=3
        )
        title_lines = self._wrap_pixels(title, BOX_TITLE_FONT, w - 32)
        title_height = len(title_lines) * 29
        ty = y + max(16, (h - title_height - (40 if body else 0)) // 2)
        for line in title_lines:
            bbox = self.draw.textbbox((0, 0), line, font=BOX_TITLE_FONT)
            tw = bbox[2] - bbox[0]
            self.draw.text((x + (w - tw) / 2, ty), line, font=BOX_TITLE_FONT, fill=COLORS["ink"])
            ty += 29
        if body:
            ty += 5
            for line in self._wrap_pixels(body, BOX_BODY_FONT, w - 36):
                bbox = self.draw.textbbox((0, 0), line, font=BOX_BODY_FONT)
                tw = bbox[2] - bbox[0]
                self.draw.text(
                    (x + (w - tw) / 2, ty), line, font=BOX_BODY_FONT, fill=COLORS["muted"]
                )
                ty += 25

    def pill(self, xy, size, text, kind="neutral"):
        self.box(xy, size, text, kind=kind, radius=size[1] // 2)

    def arrow(self, start, end, color=None, width=4, dashed=False, label=None):
        color = color or COLORS["line"]
        x1, y1 = start
        x2, y2 = end
        if dashed:
            segments = 12
            for i in range(segments):
                if i % 2 == 0:
                    a = i / segments
                    b = (i + 1) / segments
                    self.draw.line(
                        (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                         x1 + (x2 - x1) * b, y1 + (y2 - y1) * b),
                        fill=color,
                        width=width,
                    )
        else:
            self.draw.line((x1, y1, x2, y2), fill=color, width=width)

        import math

        angle = math.atan2(y2 - y1, x2 - x1)
        length = 15
        spread = 0.55
        p1 = (x2, y2)
        p2 = (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread))
        p3 = (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread))
        self.draw.polygon([p1, p2, p3], fill=color)

        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            bbox = self.draw.textbbox((0, 0), label, font=SMALL_FONT)
            tw = bbox[2] - bbox[0]
            self.draw.rounded_rectangle(
                (mx - tw / 2 - 8, my - 15, mx + tw / 2 + 8, my + 12),
                radius=8,
                fill=COLORS["background"],
            )
            self.draw.text((mx - tw / 2, my - 12), label, font=SMALL_FONT, fill=COLORS["muted"])

    def note(self, xy, text, color=None, max_width=1300):
        x, y = xy
        color = color or COLORS["muted"]
        for line in self._wrap_pixels(text, SMALL_FONT, max_width):
            self.draw.text((x, y), line, font=SMALL_FONT, fill=color)
            y += 24

    def save(self, name: str):
        self.image.save(OUTPUT / name, format="PNG", optimize=True)

    def _wrap_pixels(self, text, font, max_width):
        lines = []
        for paragraph in text.splitlines() or [""]:
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                bbox = self.draw.textbbox((0, 0), candidate, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        return lines


def horizontal_chain(diagram, items, y, box_size=(205, 115), gap=35):
    total = len(items) * box_size[0] + (len(items) - 1) * gap
    x = (WIDTH - total) // 2
    positions = []
    for index, (title, body, kind) in enumerate(items):
        diagram.box((x, y), box_size, title, body, kind)
        positions.append((x, y))
        if index:
            previous_x = positions[index - 1][0]
            diagram.arrow(
                (previous_x + box_size[0], y + box_size[1] / 2),
                (x - 8, y + box_size[1] / 2),
            )
        x += box_size[0] + gap
    return positions


def render_overview():
    d = Diagram(
        "Bức tranh toàn hệ thống",
        "Hai luồng độc lập: chuẩn bị kho kiến thức và trả lời câu hỏi.",
    )
    d.section((55, 170), (1490, 270), "A. KHI CÓ TÀI LIỆU MỚI")
    ingest = [
        ("Tài liệu", "PDF · DOCX · Excel", "input"),
        ("Đọc nội dung", "Text · bảng · hình", "process"),
        ("Ghép quan hệ", "Đoạn · bảng · biểu đồ", "process"),
        ("Chia nhỏ", "Parent và child", "process"),
        ("Tạo vector", "Dense và BM25", "model"),
        ("Đóng băng", "Qdrant + metadata bundle", "store"),
    ]
    top_positions = horizontal_chain(d, ingest, 260, box_size=(205, 125), gap=32)

    d.section((55, 470), (1490, 300), "B. KHI NGƯỜI DÙNG ĐẶT CÂU HỎI")
    chat = [
        ("Câu hỏi", "Tiếng Việt", "input"),
        ("Tìm kiếm", "Ý nghĩa + từ khóa", "process"),
        ("Xếp hạng", "Chọn đoạn tốt nhất", "model"),
        ("Lấy ngữ cảnh", "Mở rộng parent", "process"),
        ("Qwen trả lời", "Chỉ dùng context", "model"),
        ("Kết quả", "Answer + nguồn", "output"),
    ]
    bottom_positions = horizontal_chain(d, chat, 575, box_size=(205, 125), gap=32)
    d.note((535, 447), "corpus_bundle.zip được chuyển sang session hỏi đáp; không ingest thêm.", max_width=720)
    d.note((70, 820), "Mục tiêu: tìm đúng bằng child chunk, trả lời đủ ngữ cảnh bằng parent và luôn chỉ ra nguồn.")
    d.save("01-system-overview.png")


def render_parsing():
    d = Diagram(
        "Đọc tài liệu đa định dạng",
        "Parser native lấy dữ liệu gốc; PaddleOCR-VL-1.6 đọc phần scan và nội dung trực quan.",
    )

    files = [(150, "PDF"), (675, "DOCX"), (1200, "Excel")]
    parsers = [
        ("Docling + PyMuPDF", "Scan: PaddleOCR-VL"),
        ("Docling + python-docx", "Đọc OOXML khi cần"),
        ("openpyxl + pandas", "Giữ formula và cell range"),
    ]
    for (x, name), (parser, body) in zip(files, parsers):
        d.box((x, 185), (250, 95), name, kind="input")
        d.box((x, 335), (250, 115), parser, body, "process")
        d.arrow((x + 125, 280), (x + 125, 325))

    types = [
        (170, "Text", "Lấy trực tiếp"),
        (490, "Bảng", "Rows + headers"),
        (810, "Hình / flowchart", "PaddleOCR-VL-1.6"),
        (1130, "Chart", "Data gốc / OCR-VL"),
    ]
    for x, title, body in types:
        kind = "model" if "Qwen" in body else "neutral"
        d.box((x, 560), (250, 110), title, body, kind)

    d.box((560, 490), (480, 70), "Chuẩn hóa thành các block", kind="process")
    for x, _ in files:
        d.arrow((x + 125, 450), (800, 480), color="#87939D", width=3)
    for x, _, _ in types:
        d.arrow((800, 560), (x + 125, 550), color="#87939D", width=3)

    d.box(
        (500, 745),
        (600, 100),
        "Canonical Document Model",
        "Một cấu trúc chung cho PDF, DOCX và Excel",
        "store",
    )
    for x, _, _ in types:
        d.arrow((x + 125, 670), (800, 735), color="#87939D", width=3)
    d.save("02-document-parsing.png")


def render_parent_child():
    d = Diagram(
        "Parent-child chunking",
        "Child giúp tìm chính xác; parent giúp giữ trọn ý nghĩa của nội dung.",
    )
    d.box(
        (330, 175),
        (940, 145),
        "PARENT: Quy trình phê duyệt khoản vay",
        "Đoạn giải thích + flowchart + bảng điều kiện + caption",
        "output",
    )

    children = [
        (90, "Child 1", "Đoạn giải thích"),
        (465, "Child 2", "Nội dung flowchart"),
        (840, "Child 3", "Bảng - hàng 1 đến 15"),
        (1215, "Child 4", "Bảng - hàng 16 đến 30"),
    ]
    for x, title, body in children:
        d.box((x, 410), (295, 115), title, body, "input")
        d.arrow((800, 320), (x + 147, 400), color="#7C6BA8", width=3)

    d.box((85, 665), (330, 115), "Câu hỏi", "CIC không đạt thì xử lý thế nào?", "process")
    d.box((520, 665), (270, 115), "Tìm thấy", "Child 2", "model")
    d.box((900, 665), (275, 115), "Mở rộng", "Lấy parent_id", "process")
    d.box((1280, 665), (245, 115), "Context", "Chỉ lấy phần liên quan", "store")
    d.arrow((415, 722), (510, 722))
    d.arrow((790, 722), (890, 722))
    d.arrow((1175, 722), (1270, 722))
    d.arrow((612, 525), (655, 655), dashed=True, label="retrieval hit")
    d.note((90, 835), "Không nhét cả section dài vào prompt: ưu tiên child trúng query và các block có quan hệ trực tiếp.")
    d.save("03-parent-child.png")


def render_retrieval():
    d = Diagram(
        "Một câu hỏi đi qua retrieval",
        "Dense tìm theo ý nghĩa; BM25 tìm từ khóa; RRF kết hợp ưu điểm của cả hai.",
    )
    d.box((625, 170), (350, 100), "Câu hỏi + query rewrite", "Giữ nguyên số, mã và tên riêng", "input")
    d.box((245, 360), (340, 120), "Dense search", "Tìm câu có cùng ý nghĩa", "model")
    d.box((1015, 360), (340, 120), "BM25 search", "Tìm đúng từ khóa và mã", "process")
    d.arrow((730, 270), (415, 350))
    d.arrow((870, 270), (1185, 350))

    d.box((630, 555), (340, 105), "RRF fusion", "Gộp hai bảng xếp hạng", "process")
    d.arrow((415, 480), (720, 545))
    d.arrow((1185, 480), (880, 545))

    chain = [
        (80, "Reranker", "Chọn 5-8 đoạn", "model"),
        (395, "Mở rộng parent", "Lấy phần liên quan", "process"),
        (710, "Tính toán", "Python nếu cần", "process"),
        (1025, "Qwen", "Viết câu trả lời", "model"),
        (1340, "Kết quả", "Answer + citation", "output"),
    ]
    for index, (x, title, body, kind) in enumerate(chain):
        d.box((x, 735), (210, 110), title, body, kind)
        if index:
            previous_x = chain[index - 1][0]
            d.arrow((previous_x + 210, 790), (x - 10, 790))
    d.arrow((800, 660), (185, 725), color="#87939D", width=3)
    d.save("04-query-retrieval.png")


def render_storage():
    d = Diagram(
        "Mỗi loại dữ liệu có một nơi phù hợp",
        "Qdrant dùng để tìm kiếm, không phải nơi lưu toàn bộ dữ liệu nguồn.",
    )
    d.box((600, 175), (400, 105), "Frozen corpus bundle", "Manifest + checksum + dữ liệu retrieval", "output")

    stores = [
        (65, "Qdrant", "Vector + child chunks", "Tìm kiếm"),
        (450, "SQLite", "Parent + document graph", "Nối ngữ cảnh"),
        (835, "JSON / Parquet", "Hàng và cột gốc", "Tính toán"),
        (1220, "Asset storage", "Ảnh crop cần thiết", "Xem nguồn"),
    ]
    for x, title, body, role in stores:
        d.box((x, 410), (315, 150), title, f"{body}\n{role}", "store")
        d.arrow((800, 280), (x + 157, 400), color="#87939D", width=3)

    d.box((270, 700), (420, 115), "Context cho LLM", "Nội dung + dữ liệu đã tính", "process")
    d.box((910, 700), (420, 115), "Nguồn hiển thị", "Trang, sheet, cell range, ảnh", "input")
    for x, _, _, _ in stores[:3]:
        d.arrow((x + 157, 560), (480, 690), color="#87939D", width=3)
    d.arrow((1377, 560), (1120, 690), color="#87939D", width=3)
    d.arrow((690, 757), (900, 757), label="answer + citation")
    d.save("05-storage-map.png")


def render_gpu():
    d = Diagram(
        "Phân bổ tài nguyên Kaggle T4 x2",
        "Hai session độc lập: ingestion không giữ Qwen answer; chat không load OCR/VLM.",
    )
    d.section((55, 175), (1490, 255), "INGESTION")
    ingest = [
        ("CPU parse", "Đọc file", "neutral"),
        ("GPU 1: OCR-VL", "Đọc scan/hình/chart", "model"),
        ("GPU 0: VLM", "Chỉ khi cần", "model"),
        ("GPU 0: embedding", "Tạo vectors", "model"),
        ("Freeze bundle", "Qdrant + checksum", "store"),
    ]
    horizontal_chain(d, ingest, 265, box_size=(235, 115), gap=42)

    d.section((55, 470), (1490, 255), "CHAT")
    chat = [
        ("GPU 1: BGE-M3", "Encode query", "model"),
        ("CPU search", "Qdrant + BM25", "neutral"),
        ("CPU reranker", "FP32, top candidates", "neutral"),
        ("GPU 0: Qwen 7B", "4-bit", "model"),
        ("Generate", "Answer + source", "output"),
    ]
    horizontal_chain(d, chat, 560, box_size=(235, 115), gap=42)

    d.box(
        (315, 785),
        (970, 75),
        "Quy tắc: embedding trong chat phải trùng manifest; corpus không nhận thêm dữ liệu",
        kind="danger",
    )
    d.save("06-kaggle-gpu-lifecycle.png")


def main():
    render_overview()
    render_parsing()
    render_parent_child()
    render_retrieval()
    render_storage()
    render_gpu()
    print(f"Rendered 6 diagrams to {OUTPUT}")


if __name__ == "__main__":
    main()
