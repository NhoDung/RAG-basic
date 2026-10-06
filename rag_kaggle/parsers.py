from __future__ import annotations

import hashlib
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Iterable

from .config import PipelineConfig
from .models import Block, ParsedDocument
from .paddleocr_vl import PaddleOCRVLAdapter


LOGGER = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentParser:
    def __init__(self, config: PipelineConfig, ocr: PaddleOCRVLAdapter | None = None):
        self.config = config
        self.ocr = ocr

    def parse(self, path: str | Path) -> ParsedDocument:
        source = Path(path)
        extension = source.suffix.lower()
        if extension == ".pdf":
            return self._parse_pdf(source)
        if extension == ".docx":
            return self._parse_docx(source)
        if extension in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
            return self._parse_excel(source)
        if extension == ".xls":
            return self._parse_legacy_excel(source)
        raise ValueError(f"Unsupported file type: {extension}")

    def _new_document(self, source: Path) -> ParsedDocument:
        digest = file_sha256(source)
        document_id = stable_id(source.name, digest)
        return ParsedDocument(
            document_id=document_id,
            source_file=source.name,
            file_type=source.suffix.lower().lstrip("."),
            content_hash=digest,
        )

    def _asset_dir(self, document_id: str) -> Path:
        target = self.config.asset_dir / document_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _block(
        self,
        document: ParsedDocument,
        block_type: str,
        content: str,
        ordinal: object,
        **kwargs,
    ) -> Block:
        return Block(
            block_id=stable_id(document.document_id, block_type, ordinal),
            document_id=document.document_id,
            block_type=block_type,
            content=clean_text(content),
            source_file=document.source_file,
            **kwargs,
        )

    def _parse_pdf(self, source: Path) -> ParsedDocument:
        import fitz

        document = self._new_document(source)
        asset_dir = self._asset_dir(document.document_id)
        pdf = fitz.open(source)
        current_section: list[str] = []

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            page_text = clean_text(page.get_text("text"))
            is_scan = len(page_text) < self.config.parsing.scan_text_threshold

            if is_scan and self.config.parsing.enable_ocr:
                image_path = asset_dir / f"page_{page_number:04d}.png"
                pixmap = page.get_pixmap(dpi=self.config.parsing.render_dpi, alpha=False)
                pixmap.save(image_path)
                ocr_result = self._run_ocr(image_path)
                if ocr_result["text"]:
                    document.blocks.append(
                        self._block(
                            document,
                            "ocr_page",
                            ocr_result["text"],
                            f"page-{page_number}",
                            page=page_number,
                            asset_path=str(image_path),
                            metadata={"ocr": ocr_result["raw"], "is_scan": True},
                        )
                    )
                continue

            for block_index, raw in enumerate(page.get_text("blocks")):
                x0, y0, x1, y1, text, *_ = raw
                text = clean_text(text)
                if not text:
                    continue
                block_type = "heading" if len(text) < 120 and text.count("\n") <= 1 else "text"
                if block_type == "heading":
                    current_section = [text]
                document.blocks.append(
                    self._block(
                        document,
                        block_type,
                        text,
                        f"page-{page_number}-text-{block_index}",
                        section_path=list(current_section),
                        page=page_number,
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                    )
                )

            self._extract_pdf_tables(document, page, page_number, current_section)
            if self.config.parsing.ocr_images_in_digital_docs:
                self._extract_pdf_images(document, pdf, page, page_number, asset_dir, current_section)

        pdf.close()
        return document

    def _extract_pdf_tables(self, document, page, page_number, section_path):
        try:
            finder = page.find_tables()
        except (AttributeError, RuntimeError):
            return
        for table_index, table in enumerate(getattr(finder, "tables", [])):
            rows = table.extract()
            if not rows:
                continue
            normalized = [["" if cell is None else clean_text(str(cell)) for cell in row] for row in rows]
            markdown = rows_to_markdown(normalized)
            document.blocks.append(
                self._block(
                    document,
                    "table",
                    markdown,
                    f"page-{page_number}-table-{table_index}",
                    section_path=list(section_path),
                    page=page_number,
                    bbox=[float(value) for value in table.bbox],
                    metadata={"rows": normalized},
                )
            )

    def _extract_pdf_images(self, document, pdf, page, page_number, asset_dir, section_path):
        for image_index, image_info in enumerate(page.get_images(full=True)):
            xref = image_info[0]
            try:
                extracted = pdf.extract_image(xref)
            except (RuntimeError, ValueError):
                continue
            width = int(extracted.get("width", 0))
            height = int(extracted.get("height", 0))
            if width < self.config.parsing.min_image_width or height < self.config.parsing.min_image_height:
                continue
            extension = extracted.get("ext", "png")
            image_path = asset_dir / f"page_{page_number:04d}_image_{image_index:03d}.{extension}"
            image_path.write_bytes(extracted["image"])
            ocr_result = self._run_ocr(image_path)
            if not ocr_result["text"]:
                continue
            document.blocks.append(
                self._block(
                    document,
                    "image",
                    ocr_result["text"],
                    f"page-{page_number}-image-{image_index}",
                    section_path=list(section_path),
                    page=page_number,
                    asset_path=str(image_path),
                    metadata={"ocr": ocr_result["raw"], "width": width, "height": height},
                )
            )

    def _parse_docx(self, source: Path) -> ParsedDocument:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = self._new_document(source)
        docx = Document(source)
        current_section: list[str] = []
        ordinal = 0

        for item in iter_docx_blocks(docx):
            ordinal += 1
            if isinstance(item, Paragraph):
                text = clean_text(item.text)
                if not text:
                    continue
                style_name = (item.style.name or "").lower() if item.style else ""
                is_heading = style_name.startswith("heading") or style_name.startswith("tiêu đề")
                if is_heading:
                    current_section = [text]
                document.blocks.append(
                    self._block(
                        document,
                        "heading" if is_heading else "text",
                        text,
                        f"docx-{ordinal}",
                        section_path=list(current_section),
                        metadata={"style": item.style.name if item.style else None},
                    )
                )
            elif isinstance(item, Table):
                rows = [[clean_text(cell.text) for cell in row.cells] for row in item.rows]
                document.blocks.append(
                    self._block(
                        document,
                        "table",
                        rows_to_markdown(rows),
                        f"docx-{ordinal}",
                        section_path=list(current_section),
                        metadata={"rows": rows},
                    )
                )

        self._extract_docx_media(document, source, current_section)
        self._extract_docx_embedded_workbooks(document, source)
        return document

    def _extract_docx_media(self, document, source, section_path):
        asset_dir = self._asset_dir(document.document_id)
        supported = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        with zipfile.ZipFile(source) as archive:
            media = [name for name in archive.namelist() if name.startswith("word/media/")]
            for index, name in enumerate(media):
                suffix = Path(name).suffix.lower()
                if suffix not in supported:
                    continue
                image_path = asset_dir / f"docx_image_{index:03d}{suffix}"
                image_path.write_bytes(archive.read(name))
                if not self._image_is_large_enough(image_path):
                    continue
                ocr_result = self._run_ocr(image_path)
                if not ocr_result["text"]:
                    continue
                document.blocks.append(
                    self._block(
                        document,
                        "image",
                        ocr_result["text"],
                        f"docx-image-{index}",
                        section_path=list(section_path),
                        asset_path=str(image_path),
                        metadata={"ocr": ocr_result["raw"], "archive_path": name},
                    )
                )

    def _extract_docx_embedded_workbooks(self, document, source):
        with zipfile.ZipFile(source) as archive:
            workbook_names = [
                name
                for name in archive.namelist()
                if name.startswith("word/embeddings/") and Path(name).suffix.lower() in {".xlsx", ".xlsm"}
            ]
            for index, name in enumerate(workbook_names):
                embedded = self.config.source_dir / f"{document.document_id}_embedded_{index}.xlsx"
                embedded.write_bytes(archive.read(name))
                try:
                    parsed = self._parse_excel(embedded)
                except Exception as exc:  # Embedded chart workbooks are frequently partial.
                    LOGGER.warning("Could not parse embedded workbook %s: %s", name, exc)
                    continue
                for block in parsed.blocks:
                    block.document_id = document.document_id
                    block.source_file = document.source_file
                    block.block_id = stable_id(document.document_id, "embedded", index, block.block_id)
                    block.metadata["embedded_workbook"] = name
                    document.blocks.append(block)

    def _parse_excel(self, source: Path) -> ParsedDocument:
        import openpyxl
        from openpyxl.utils import get_column_letter

        document = self._new_document(source)
        asset_dir = self._asset_dir(document.document_id)
        formulas = openpyxl.load_workbook(source, data_only=False, read_only=False)
        values = openpyxl.load_workbook(source, data_only=True, read_only=False)

        for sheet_index, formula_sheet in enumerate(formulas.worksheets):
            if formula_sheet.sheet_state != "visible" and not self.config.parsing.include_hidden_sheets:
                continue
            value_sheet = values[formula_sheet.title]
            regions = split_excel_regions(value_sheet)
            for region_index, (min_row, max_row, min_col, max_col) in enumerate(regions):
                rows = []
                formula_map = {}
                for row_index in range(min_row, max_row + 1):
                    row = []
                    for column_index in range(min_col, max_col + 1):
                        value_cell = value_sheet.cell(row_index, column_index)
                        formula_cell = formula_sheet.cell(row_index, column_index)
                        value = value_cell.value
                        display_value = value
                        if display_value is None and isinstance(formula_cell.value, str):
                            display_value = formula_cell.value
                        row.append("" if display_value is None else str(display_value))
                        if isinstance(formula_cell.value, str) and formula_cell.value.startswith("="):
                            formula_map[formula_cell.coordinate] = formula_cell.value
                    rows.append(row)
                cell_range = (
                    f"{get_column_letter(min_col)}{min_row}:"
                    f"{get_column_letter(max_col)}{max_row}"
                )
                document.blocks.append(
                    self._block(
                        document,
                        "table",
                        rows_to_markdown(rows),
                        f"sheet-{sheet_index}-region-{region_index}",
                        section_path=[formula_sheet.title],
                        sheet_name=formula_sheet.title,
                        cell_range=cell_range,
                        metadata={"rows": rows, "formulas": formula_map},
                    )
                )

            for chart_index, chart in enumerate(getattr(formula_sheet, "_charts", [])):
                chart_type = type(chart).__name__
                series_refs = extract_chart_references(chart)
                content = f"Chart type: {chart_type}"
                if series_refs:
                    content += "\nSource ranges:\n" + "\n".join(f"- {ref}" for ref in series_refs)
                document.blocks.append(
                    self._block(
                        document,
                        "chart",
                        content,
                        f"sheet-{sheet_index}-chart-{chart_index}",
                        section_path=[formula_sheet.title],
                        sheet_name=formula_sheet.title,
                        metadata={"chart_type": chart_type, "source_ranges": series_refs},
                    )
                )

            for image_index, image in enumerate(getattr(formula_sheet, "_images", [])):
                try:
                    image_bytes = image._data()
                except (AttributeError, ValueError):
                    continue
                extension = Path(getattr(image, "path", "image.png")).suffix or ".png"
                image_path = asset_dir / f"sheet_{sheet_index}_image_{image_index}{extension}"
                image_path.write_bytes(image_bytes)
                if not self._image_is_large_enough(image_path):
                    continue
                ocr_result = self._run_ocr(image_path)
                if not ocr_result["text"]:
                    continue
                document.blocks.append(
                    self._block(
                        document,
                        "image",
                        ocr_result["text"],
                        f"sheet-{sheet_index}-image-{image_index}",
                        section_path=[formula_sheet.title],
                        sheet_name=formula_sheet.title,
                        asset_path=str(image_path),
                        metadata={"ocr": ocr_result["raw"]},
                    )
                )

        formulas.close()
        values.close()
        return document

    def _parse_legacy_excel(self, source: Path) -> ParsedDocument:
        import pandas as pd

        document = self._new_document(source)
        sheets = pd.read_excel(source, sheet_name=None, header=None, engine="calamine")
        for sheet_index, (sheet_name, frame) in enumerate(sheets.items()):
            frame = frame.dropna(how="all").dropna(axis=1, how="all")
            if frame.empty:
                continue
            rows = [
                ["" if pd.isna(value) else str(value) for value in row]
                for row in frame.itertuples(index=False, name=None)
            ]
            cell_range = f"A1:{excel_column_name(len(frame.columns))}{len(frame.index)}"
            document.blocks.append(
                self._block(
                    document,
                    "table",
                    rows_to_markdown(rows),
                    f"legacy-sheet-{sheet_index}",
                    section_path=[str(sheet_name)],
                    sheet_name=str(sheet_name),
                    cell_range=cell_range,
                    metadata={"rows": rows, "legacy_xls": True},
                )
            )
        return document

    def _run_ocr(self, image_path: Path) -> dict:
        if not self.config.parsing.enable_ocr or self.ocr is None:
            return {"text": "", "raw": [], "model": None}
        try:
            return self.ocr.predict(image_path)
        except Exception as exc:
            LOGGER.exception("OCR failed for %s", image_path)
            return {"text": "", "raw": [], "model": self.config.parsing.ocr_model_name, "error": str(exc)}

    def _image_is_large_enough(self, path: Path) -> bool:
        try:
            from PIL import Image

            with Image.open(path) as image:
                return (
                    image.width >= self.config.parsing.min_image_width
                    and image.height >= self.config.parsing.min_image_height
                )
        except Exception:
            return False


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:]
    lines = [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |" for row in body)
    return "\n".join(lines)


def escape_markdown_cell(value: str) -> str:
    return clean_text(value).replace("|", "\\|").replace("\n", "<br>")


def iter_docx_blocks(document) -> Iterable:
    from docx.document import Document as DocumentType
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent = document.element.body if isinstance(document, DocumentType) else document._tc
    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def split_excel_regions(sheet) -> list[tuple[int, int, int, int]]:
    non_empty_rows = []
    for row_index in range(1, sheet.max_row + 1):
        columns = [
            column_index
            for column_index in range(1, sheet.max_column + 1)
            if sheet.cell(row_index, column_index).value not in (None, "")
        ]
        non_empty_rows.append(columns)

    regions = []
    start = None
    for offset, columns in enumerate(non_empty_rows + [[]], start=1):
        if columns and start is None:
            start = offset
        elif not columns and start is not None:
            end = offset - 1
            used_columns = [column for row in non_empty_rows[start - 1 : end] for column in row]
            if used_columns:
                regions.append((start, end, min(used_columns), max(used_columns)))
            start = None
    return regions


def extract_chart_references(chart) -> list[str]:
    references = []
    for series in getattr(chart, "ser", []):
        for attribute_path in (
            ("val", "numRef", "f"),
            ("cat", "numRef", "f"),
            ("cat", "strRef", "f"),
            ("tx", "strRef", "f"),
        ):
            value = series
            for attribute in attribute_path:
                value = getattr(value, attribute, None)
                if value is None:
                    break
            if value:
                references.append(str(value))
    return list(dict.fromkeys(references))


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name or "A"


def save_parsed_document(document: ParsedDocument, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{document.document_id}.json"
    path.write_text(json.dumps(document.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
