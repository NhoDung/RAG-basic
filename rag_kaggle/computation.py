from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .models import Block
from .utils import normalize_for_match, parse_number


OPERATIONS = {
    "sum": ("tổng", "cộng lại", "tổng cộng", "sum", "total"),
    "avg": ("trung bình", "bình quân", "average", "mean"),
    "max": ("lớn nhất", "cao nhất", "nhiều nhất", "tối đa", "max", "highest", "largest"),
    "min": ("nhỏ nhất", "thấp nhất", "ít nhất", "tối thiểu", "min", "lowest", "smallest"),
    "count": ("đếm", "số lượng", "bao nhiêu dòng", "bao nhiêu mục", "có bao nhiêu", "count", "how many"),
}
TOTAL_ROW_PATTERN = re.compile(r"^(tổng|tổng cộng|cộng|total|grand total|sum)\b", re.IGNORECASE)
LABELS = {"sum": "Tổng", "avg": "Trung bình", "max": "Giá trị lớn nhất", "min": "Giá trị nhỏ nhất", "count": "Số dòng"}


@dataclass
class ComputationResult:
    operation: str
    column: str
    value: float
    formula: str
    block_id: str
    source_file: str
    sheet_name: str | None
    cell_range: str | None
    page: int | None
    rows_used: list[list[str]] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)
    label_of_extreme: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        lines = [
            "KẾT QUẢ TÍNH TOÁN (thực hiện bằng Python trên dữ liệu bảng gốc, không do LLM tự tính)",
            f"Phép tính: {LABELS[self.operation]} của cột \"{self.column}\"",
        ]
        if self.filters:
            lines.append("Điều kiện lọc: " + ", ".join(f"{key} = {value}" for key, value in self.filters.items()))
        lines.append(f"Công thức: {self.formula}")
        lines.append(f"Kết quả: {format_number(self.value)}")
        if self.label_of_extreme:
            lines.append(f"Dòng tương ứng: {self.label_of_extreme}")
        lines.append(f"Số dòng dữ liệu sử dụng: {len(self.rows_used)}")
        preview = self.rows_used[:15]
        if preview:
            lines.append("Các dòng nguồn:\n" + "\n".join(" | ".join(row) for row in preview))
        return "\n".join(lines)


def detect_operation(query: str) -> str | None:
    lowered = normalize_for_match(query)
    # "bao nhiêu" alone usually asks for a value lookup, not a count.
    for operation in ("avg", "max", "min", "sum", "count"):
        if any(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lowered) for keyword in OPERATIONS[operation]):
            return operation
    return None


def compute_from_blocks(query: str, blocks: list[Block]) -> ComputationResult | None:
    """Pick the best numeric column across candidate tables and compute in Python (§9.5)."""
    operation = detect_operation(query)
    if operation is None:
        return None
    best: tuple[float, ComputationResult] | None = None
    for block in blocks:
        result = _compute_on_block(query, operation, block)
        if result is None:
            continue
        score = result[0]
        if best is None or score > best[0]:
            best = result
    return best[1] if best else None


def _compute_on_block(query: str, operation: str, block: Block):
    rows = block.metadata.get("rows") or []
    header = block.metadata.get("inherited_header") or (rows[0] if rows else [])
    body = rows if block.metadata.get("inherited_header") else rows[1:]
    if not header or not body:
        return None
    python_repr = block.metadata.get("numeric_repr") == "python"
    query_norm = normalize_for_match(query)
    query_tokens = set(re.findall(r"\w+", query_norm))
    width = len(header)

    numeric_columns = []
    for column in range(width):
        values = [_value(row, column) for row in body]
        non_empty = [value for value in values if value]
        if not non_empty:
            continue
        parsed = [parse_number(value, python_repr=python_repr) for value in non_empty]
        ratio = sum(value is not None for value in parsed) / len(non_empty)
        if ratio >= 0.7:
            name_tokens = set(re.findall(r"\w+", normalize_for_match(header[column])))
            overlap = len(name_tokens & query_tokens) / max(len(name_tokens), 1)
            numeric_columns.append((overlap, column))
    if not numeric_columns and operation != "count":
        return None

    # Filter rows by categorical values mentioned verbatim in the question.
    filters: dict[str, str] = {}
    selected = [row for row in body if not TOTAL_ROW_PATTERN.match(_value(row, 0))]
    numeric_set = {column for _, column in numeric_columns}
    for column in range(width):
        if column in numeric_set:
            continue
        values = {normalize_for_match(_value(row, column)) for row in selected if _value(row, column)}
        mentioned = [value for value in values if len(value) >= 2 and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", query_norm)]
        if mentioned and len(mentioned) < len(values):
            target = max(mentioned, key=len)
            selected = [row for row in selected if normalize_for_match(_value(row, column)) == target]
            filters[header[column] or f"cột {column + 1}"] = target
    if not selected:
        return None

    base = {
        "block_id": block.block_id,
        "source_file": block.source_file,
        "sheet_name": block.sheet_name,
        "cell_range": block.cell_range,
        "page": block.page,
        "filters": filters,
    }
    if operation == "count":
        result = ComputationResult(
            operation="count",
            column=header[0] or "dòng",
            value=float(len(selected)),
            formula=f"COUNT(rows) = {len(selected)}",
            rows_used=selected,
            **base,
        )
        return (1.0 + len(filters), result)

    overlap, column = max(numeric_columns, key=lambda item: (item[0], -item[1]))
    numbers = [(parse_number(_value(row, column), python_repr=python_repr), row) for row in selected]
    numbers = [(value, row) for value, row in numbers if value is not None]
    if not numbers:
        return None
    values = [value for value, _ in numbers]
    label = None
    if operation == "sum":
        value = sum(values)
        formula = " + ".join(format_number(item) for item in values[:30]) + (" + ..." if len(values) > 30 else "")
        formula = f"SUM = {formula} = {format_number(value)}"
    elif operation == "avg":
        value = sum(values) / len(values)
        formula = f"AVERAGE = {format_number(sum(values))} / {len(values)} = {format_number(value)}"
    else:
        picker = max if operation == "max" else min
        value, row = picker(numbers, key=lambda item: item[0])
        label = " | ".join(row)
        formula = f"{operation.upper()}({len(values)} giá trị) = {format_number(value)}"
    result = ComputationResult(
        operation=operation,
        column=header[column],
        value=value,
        formula=formula,
        rows_used=[row for _, row in numbers],
        label_of_extreme=label,
        **base,
    )
    return (overlap + 0.5 * len(filters), result)


def _value(row: list[str], column: int) -> str:
    return str(row[column]).strip() if column < len(row) else ""


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}".replace(",", ".")
    integer, _, decimals = f"{value:,.4f}".rstrip("0").partition(".")
    return integer.replace(",", ".") + ("," + decimals if decimals else "")
