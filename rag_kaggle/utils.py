from __future__ import annotations

import hashlib
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_id(*parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text)).lower()).strip()


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [list(row) + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:]
    lines = [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |" for row in body)
    return "\n".join(lines)


def escape_markdown_cell(value: str) -> str:
    return clean_text(str(value)).replace("|", "\\|").replace("\n", "<br>")


def table_xlsx_relpath(document_id: str, block_id: str) -> str:
    """Corpus-relative path of the full-table workbook for a large table block."""
    return f"tables/{document_id}/{block_id}.xlsx"


def estimate_tokens(text: str, chars_per_token: float = 3.6) -> int:
    return max(1, round(len(text) / chars_per_token)) if text else 0


_NUMBER_PATTERN = re.compile(r"\(?-?\d[\d.,]*\)?%?")


def parse_number(value: object, python_repr: bool = False) -> float | None:
    """Parse numbers written in Vietnamese or English conventions.

    ``python_repr`` is used for values serialized from Excel cells with ``str()``,
    where ``.`` is always the decimal separator.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if python_repr:
        try:
            return float(text)
        except ValueError:
            pass
    compact = re.sub(r"(?i)(vnđ|vnd|đồng|usd|đ|\$|€|triệu|tỷ|nghìn)", "", text).replace(" ", "")
    match = _NUMBER_PATTERN.fullmatch(compact)
    if not match:
        return None
    raw = match.group(0)
    negative = raw.startswith("(") and raw.endswith(")") or raw.startswith("-")
    percent = raw.endswith("%")
    digits = raw.strip("()%-")
    if not digits or not digits[0].isdigit():
        return None
    if "." in digits and "," in digits:
        decimal = "." if digits.rfind(".") > digits.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        digits = digits.replace(thousands, "").replace(decimal, ".")
    elif "." in digits or "," in digits:
        separator = "." if "." in digits else ","
        parts = digits.split(separator)
        if len(parts) > 2 or (len(parts[-1]) == 3 and parts[0] != "0"):
            digits = digits.replace(separator, "")
        else:
            digits = digits.replace(separator, ".")
    try:
        number = float(digits)
    except ValueError:
        return None
    if negative:
        number = -number
    if percent:
        number = number / 100
    return number


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in re.finditer(r"\d[\d.,]*%?", text or ""):
        token = match.group(0).rstrip(".,")
        number = parse_number(token)
        if number is not None:
            numbers.append(number)
    return numbers


def html_table_to_rows(html: str) -> list[list[str]]:
    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self.row: list[str] | None = None
            self.cell: list[str] | None = None
            self.colspan = 1

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row = []
            elif tag in {"td", "th"} and self.row is not None:
                self.cell = []
                self.colspan = int(dict(attrs).get("colspan") or 1)

        def handle_endtag(self, tag):
            if tag in {"td", "th"} and self.row is not None and self.cell is not None:
                text = clean_text("".join(self.cell))
                self.row.extend([text] * max(1, self.colspan))
                self.cell = None
            elif tag == "tr" and self.row is not None:
                if any(self.row):
                    self.rows.append(self.row)
                self.row = None

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

    parser = _TableParser()
    parser.feed(html or "")
    return parser.rows


def markdown_table_to_rows(markdown: str) -> list[list[str]]:
    rows = []
    for line in (markdown or "").splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [clean_text(cell) for cell in re.split(r"(?<!\\)\|", line.strip("|"))]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            continue
        rows.append(cells)
    return rows


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name or "A"


def excel_column_index(name: str) -> int:
    index = 0
    for char in name.upper():
        index = index * 26 + ord(char) - 64
    return index


def parse_cell_range(reference: str) -> tuple[str | None, int, int, int, int] | None:
    """Parse ``'Sheet 1'!$B$2:$C$9`` into (sheet, min_row, max_row, min_col, max_col)."""
    if not reference:
        return None
    sheet = None
    ref = reference.strip()
    if "!" in ref:
        sheet, ref = ref.rsplit("!", 1)
        sheet = sheet.strip("'").replace("''", "'")
    ref = ref.replace("$", "")
    parts = ref.split(":")
    coords = []
    for part in parts:
        match = re.fullmatch(r"([A-Za-z]{1,3})(\d+)", part)
        if not match:
            return None
        coords.append((int(match.group(2)), excel_column_index(match.group(1))))
    if len(coords) == 1:
        coords.append(coords[0])
    (row_a, col_a), (row_b, col_b) = coords
    return sheet, min(row_a, row_b), max(row_a, row_b), min(col_a, col_b), max(col_a, col_b)


def ranges_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    a_min_row, a_max_row, a_min_col, a_max_col = first
    b_min_row, b_max_row, b_min_col, b_max_col = second
    return not (
        a_max_row < b_min_row or b_max_row < a_min_row or a_max_col < b_min_col or b_max_col < a_min_col
    )
