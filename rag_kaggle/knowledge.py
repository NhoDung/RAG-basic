"""Document-level metadata: title, document numbers, dates, signers, entities, keywords.

The profile is built once per document after parsing from two sources:

* rules that need no model (document number, dates, "Người ký: ...", headings);
* the glossary collected by the OCR corrector, when it ran (people, places, ...).

It is stored with the document (SQLite), copied per chunk (``entities`` / ``keywords``,
also in the Qdrant payload) and shown to the answer model next to each source.
"""

from __future__ import annotations

import re
from typing import Any

from .models import ParsedDocument
from .utils import clean_text, normalize_for_match


CHUNK_ENTITY_KINDS = ("person", "signer", "place", "organization", "event")
MAX_ITEMS = 30

DOC_NUMBER_PATTERN = re.compile(r"\bsố\s*[:：]?\s*(\d[\w.\-]*(?:/[\w.\-]+)+)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\bngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)
SIGNER_PATTERN = re.compile(r"^(?:người ký|ký bởi|signed by)\s*[:：]\s*(.{2,80})$", re.IGNORECASE | re.MULTILINE)
NUMBERING_ONLY = re.compile(r"[\d\s.)(IVXLC-]+")


def build_document_profile(document: ParsedDocument) -> dict[str, Any]:
    """Collect searchable document metadata. Never raises on odd input; fields may be empty."""
    memory = (document.metadata.get("ocr_correction") or {}).get("memory") or {}
    entities: dict[str, list[str]] = {kind: [] for kind in CHUNK_ENTITY_KINDS}
    for kind in CHUNK_ENTITY_KINDS:
        entities[kind] = [group["value"] for group in memory.get(kind, [])]

    text_blocks = [block for block in document.blocks if block.block_type in ("heading", "text", "ocr_page", "caption")]
    # Document numbers and dates sit at the top of administrative documents.
    head_text = "\n".join(block.content for block in text_blocks[:40])
    all_text = "\n".join(block.content for block in text_blocks)

    doc_numbers = _unique(match.group(1).strip(".") for match in DOC_NUMBER_PATTERN.finditer(head_text))
    dates = _unique(
        f"{int(day):02d}/{int(month):02d}/{year}" for day, month, year in DATE_PATTERN.findall(head_text)
    )
    entities["signer"] = _unique([*entities["signer"], *(match.group(1).strip() for match in SIGNER_PATTERN.finditer(all_text))])

    headings = _unique(
        clean_text(block.content).split("\n")[0]
        for block in document.blocks
        if block.block_type == "heading" and 3 <= len(block.content.strip()) <= 80
        and not NUMBERING_ONLY.fullmatch(block.content.strip())
    )
    keywords = _unique([*(group["value"] for group in memory.get("keyword", [])), *headings])

    first = next((block for block in text_blocks if block.block_type in ("heading", "text")), None)
    title = clean_text(first.content).split("\n")[0][:160] if first is not None else ""
    return {
        "title": title,
        "doc_numbers": doc_numbers[:10],
        "dates": dates[:10],
        "entities": {kind: values[:MAX_ITEMS] for kind, values in entities.items() if values},
        "keywords": keywords[:MAX_ITEMS],
    }


def profile_terms(profile: dict[str, Any] | None) -> dict[str, list[tuple[str, re.Pattern]]]:
    """Compile the profile into word-boundary patterns used to tag individual chunks."""
    profile = profile or {}
    entity_values = [
        value for kind in CHUNK_ENTITY_KINDS for value in (profile.get("entities") or {}).get(kind, [])
    ]
    keyword_values = [*(profile.get("keywords") or []), *(profile.get("doc_numbers") or [])]
    return {"entities": _compile(entity_values), "keywords": _compile(keyword_values)}


def match_terms(terms: list[tuple[str, re.Pattern]], text: str, limit: int) -> list[str]:
    """Terms (original spelling) that literally occur in ``text``, in profile order."""
    if not terms or not text:
        return []
    normalized = normalize_for_match(text)
    found = [value for value, pattern in terms if pattern.search(normalized)]
    return found[:limit]


def profile_entities(profile: dict[str, Any] | None) -> list[str]:
    return list(
        dict.fromkeys(
            value for kind in CHUNK_ENTITY_KINDS for value in ((profile or {}).get("entities") or {}).get(kind, [])
        )
    )


def format_profile(profile: dict[str, Any] | None, max_chars: int = 400) -> str:
    """One compact line for the answer prompt; empty when the document has no metadata."""
    if not profile:
        return ""
    entities = profile.get("entities") or {}
    parts = []
    if profile.get("title"):
        parts.append(f"tiêu đề: {profile['title']}")
    if profile.get("doc_numbers"):
        parts.append("số hiệu: " + ", ".join(profile["doc_numbers"][:3]))
    if profile.get("dates"):
        parts.append("ngày: " + ", ".join(profile["dates"][:2]))
    if entities.get("signer"):
        parts.append("người ký: " + ", ".join(entities["signer"][:3]))
    mentioned = [value for kind in ("person", "place", "organization", "event") for value in entities.get(kind, [])]
    if mentioned:
        parts.append("nhắc tới: " + ", ".join(mentioned[:8]))
    return "; ".join(parts)[:max_chars]


def _compile(values: list[str]) -> list[tuple[str, re.Pattern]]:
    compiled = []
    for value in _unique(values):
        normalized = normalize_for_match(value)
        if len(normalized) >= 2:
            compiled.append((value, re.compile(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)")))
    return compiled


def _unique(values) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))
