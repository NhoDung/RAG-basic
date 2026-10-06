from __future__ import annotations

import re

from .models import Block, ParsedDocument, Relationship
from .parsers import CAPTION_PATTERN
from .utils import normalize_for_match, parse_cell_range, ranges_overlap


VISUAL_TYPES = {"image", "chart", "flowchart"}
CAPTION_KIND = {
    "bảng": {"table"},
    "table": {"table"},
}
REFERENCE_PATTERN = re.compile(
    r"\b(hình|bảng|biểu đồ|sơ đồ|lưu đồ|đồ thị|figure|fig\.|table|chart)\s*(\d+(?:[.\-]\d+)*)",
    re.IGNORECASE,
)


def build_relationships(document: ParsedDocument) -> list[Relationship]:
    """Build the document graph (§5.3) and attach caption/reference text to blocks.

    The graph is stored in SQLite; attached metadata lets the chunker embed captions
    and referencing sentences together with figures and tables (§6.2).
    """
    relations: dict[tuple[str, str, str], Relationship] = {}
    blocks = sorted(document.blocks, key=lambda block: block.reading_order)

    def add(source: Block, target: Block, relation_type: str, **metadata) -> None:
        if source.block_id == target.block_id:
            return
        key = (source.block_id, target.block_id, relation_type)
        relations.setdefault(
            key, Relationship(source.block_id, target.block_id, relation_type, document.document_id, metadata)
        )

    _reading_order(blocks, add)
    _sections(blocks, add)
    captions = _captions(blocks, add)
    _references(blocks, captions, add)
    _chart_sources(blocks, add)
    _table_continuations(blocks, add)
    _derived(blocks, add)

    document.relationships = list(relations.values())
    return document.relationships


def _reading_order(blocks, add):
    for previous, current in zip(blocks, blocks[1:]):
        add(previous, current, "next_in_reading_order")


def _sections(blocks, add):
    headings: dict[tuple[str, ...], Block] = {}
    for block in blocks:
        path = tuple(block.section_path)
        if block.block_type == "heading" and path:
            headings[path] = block
            continue
        heading = headings.get(path)
        if heading is not None:
            add(block, heading, "belongs_to_section")


def _caption_targets(caption: Block) -> set[str]:
    match = CAPTION_PATTERN.match(caption.content)
    if not match:
        return VISUAL_TYPES | {"table"}
    return CAPTION_KIND.get(match.group(1).lower(), VISUAL_TYPES)


def _captions(blocks, add) -> dict[str, Block]:
    """Link each figure/table to the nearest caption on the same page/section.

    Returns a mapping ``"<kind> <number>"`` -> captioned block for reference lookup.
    """
    labelled: dict[str, Block] = {}
    used_captions: set[str] = set()
    for index, block in enumerate(blocks):
        if block.block_type not in VISUAL_TYPES | {"table"} or block.metadata.get("derived_from"):
            continue
        candidates = []
        for offset in (-1, 1, -2, 2):
            position = index + offset
            if 0 <= position < len(blocks):
                other = blocks[position]
                if (
                    other.block_type == "caption"
                    and other.block_id not in used_captions
                    and other.page == block.page
                    and other.sheet_name == block.sheet_name
                    and block.block_type in _caption_targets(other)
                ):
                    candidates.append(other)
        if not candidates:
            continue
        caption = candidates[0]
        used_captions.add(caption.block_id)
        add(block, caption, "captioned_by")
        block.metadata["caption"] = caption.content
        caption.metadata["attached_to"] = block.block_id
        match = CAPTION_PATTERN.match(caption.content)
        if match:
            labelled[f"{_kind(match.group(1))} {match.group(2)}"] = block
    return labelled


def _kind(word: str) -> str:
    word = word.lower()
    return "table" if word in {"bảng", "table"} else "figure"


def _references(blocks, captions: dict[str, Block], add):
    if not captions:
        return
    for block in blocks:
        if block.block_type not in {"text", "ocr_page"}:
            continue
        for match in REFERENCE_PATTERN.finditer(block.content):
            target = captions.get(f"{_kind(match.group(1))} {match.group(2)}")
            if target is None:
                continue
            add(target, block, "referenced_by", mention=match.group(0))
            sentence = _sentence_around(block.content, match.start())
            references = target.metadata.setdefault("references", [])
            if sentence not in references and len(references) < 3:
                references.append(sentence)


def _sentence_around(text: str, position: int, limit: int = 300) -> str:
    start = max(text.rfind(".", 0, position), text.rfind("\n", 0, position)) + 1
    end_candidates = [index for index in (text.find(".", position), text.find("\n", position)) if index != -1]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()[:limit]


def _chart_sources(blocks, add):
    tables = [block for block in blocks if block.block_type in {"table", "kpi"} and block.sheet_name and block.cell_range]
    for chart in blocks:
        if chart.block_type != "chart":
            continue
        for reference in chart.metadata.get("source_ranges", []):
            parsed = parse_cell_range(reference)
            if not parsed:
                continue
            sheet, *bounds = parsed
            sheet = sheet or chart.sheet_name
            for table in tables:
                if normalize_for_match(table.sheet_name) != normalize_for_match(sheet or ""):
                    continue
                table_range = parse_cell_range(table.cell_range)
                if table_range and ranges_overlap(tuple(bounds), table_range[1:]):
                    add(chart, table, "visualizes", source_range=reference)


def _table_continuations(blocks, add):
    tables = [block for block in blocks if block.block_type == "table" and block.page is not None]
    for previous, current in zip(tables, tables[1:]):
        if current.page != previous.page + 1:
            continue
        if not _is_last_on_page(blocks, previous) or not _is_first_on_page(blocks, current):
            continue
        prev_rows, cur_rows = previous.metadata.get("rows") or [], current.metadata.get("rows") or []
        if not prev_rows or not cur_rows:
            continue
        if len(prev_rows[0]) != len(cur_rows[0]):
            continue  # Different schema: never merge.
        same_header = [normalize_for_match(c) for c in prev_rows[0]] == [normalize_for_match(c) for c in cur_rows[0]]
        add(current, previous, "continues", repeated_header=same_header)
        current.metadata["continues_from"] = previous.block_id
        if not same_header:
            current.metadata["inherited_header"] = prev_rows[0]


def _is_last_on_page(blocks, table):
    later = [b for b in blocks if b.page == table.page and b.reading_order > table.reading_order]
    return all(b.block_type in {"caption", "text"} and len(b.content) < 40 for b in later)


def _is_first_on_page(blocks, table):
    earlier = [b for b in blocks if b.page == table.page and b.reading_order < table.reading_order]
    return all(b.block_type in {"caption", "text", "heading"} and len(b.content) < 80 for b in earlier)


def _derived(blocks, add):
    by_id = {block.block_id: block for block in blocks}
    for block in blocks:
        source_id = block.metadata.get("derived_from")
        if source_id and source_id in by_id:
            add(block, by_id[source_id], "derived_from")
