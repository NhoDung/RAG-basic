"""Document profile: metadata, keywords, mentioned entities and signers.

Built deterministically from the (already OCR-corrected) blocks so it is cheap,
reproducible and independent of the LLM context window. The profile is stored with the
document, indexed in SQLite/Qdrant/BM25 and used to boost retrieval and to enrich
answer context.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .entities import EntityLedger, extract_entity_candidates, strip_diacritics
from .models import Block, Chunk, ParsedDocument


PROFILE_VERSION = 1
TEXTUAL_TYPES = {"text", "ocr_page", "caption", "heading", "image", "kpi", "chart", "flowchart"}

_DOC_NUMBER_RE = re.compile(
    r"\bS[ốo]\s*[:.]?\s*([0-9]+(?:[/\-][0-9A-Za-zĐđ]+(?:[-/][0-9A-Za-zĐđ.]+)*)+)", re.IGNORECASE
)
_DATE_TEXT_RE = re.compile(r"ng[àa]y\s+(\d{1,2})\s+th[áa]ng\s+(\d{1,2})\s+n[ăa]m\s+(\d{4})", re.IGNORECASE)
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
_LOCATION_RE = re.compile(
    r"\b(x[ãa]|ph[ưu][ờo]ng|qu[ậa]n|huy[ệe]n|t[ỉi]nh|th[àa]nh ph[ốo]|th[ịi] x[ãa]|th[ịi] tr[ấa]n)\s+"
    r"((?:[^\W\d_][\w]*\s?){1,4})",
    re.IGNORECASE,
)
_PERSON_PREFIX_RE = re.compile(r"\b(ông|bà|đồng chí|anh|chị)\s+((?:[^\W\d_]\w*\s?){2,5})", re.IGNORECASE)
_SIGNER_ROLES = (
    "chủ tịch", "phó chủ tịch", "giám đốc", "phó giám đốc", "tổng giám đốc", "trưởng phòng",
    "thủ trưởng", "người ký", "ký tên", "bộ trưởng", "thứ trưởng", "cục trưởng", "hiệu trưởng",
)
_SIGN_MARKERS = ("tm.", "kt.", "tl.", "q.", "nơi nhận")
_ISSUER_HINTS = ("ủy ban", "uỷ ban", "ubnd", "hđnd", "bộ ", "sở ", "công ty", "ngân hàng", "tổng cục", "cục ", "trường ", "ban ")
_STOPWORDS = frozenset(
    "và của các những một là có cho được trong với này đó khi để đã sẽ theo từ tại về như không "
    "hoặc nếu thì mà bị ra vào lên đến trên dưới sau trước cũng rất nhiều người ngày tháng năm".split()
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize_key(text: str) -> str:
    """Diacritic-free, lowercase, single-spaced key used to match names regardless of OCR."""
    return re.sub(r"\s+", " ", strip_diacritics(unicodedata.normalize("NFC", text)).lower()).strip()


def _body_blocks(document: ParsedDocument) -> list[Block]:
    return [b for b in sorted(document.blocks, key=lambda item: item.reading_order) if b.block_type in TEXTUAL_TYPES]


def _first_pages_text(blocks: list[Block], pages: int = 1, limit: int = 3000) -> str:
    first_page = min((b.page for b in blocks if b.page is not None), default=None)
    selected = [
        b for b in blocks if first_page is None or (b.page is not None and b.page < first_page + pages) or b.page is None
    ]
    return "\n".join(b.content for b in selected)[:limit]


def _detect_title(blocks: list[Block], fallback: str) -> str:
    for block in blocks:
        if block.block_type == "heading" and block.content.strip():
            return re.sub(r"^#+\s*", "", block.content.strip())[:200]
    for block in blocks:
        line = block.content.strip().split("\n", 1)[0]
        if len(line) >= 8:
            return line[:200]
    return fallback


def _detect_issue_date(text: str) -> str | None:
    match = _DATE_TEXT_RE.search(text)
    if match:
        day, month, year = (int(value) for value in match.groups())
    else:
        match = _DATE_NUMERIC_RE.search(text)
        if not match:
            return None
        day, month, year = (int(value) for value in match.groups())
    if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _detect_issuer(text: str) -> str | None:
    for line in text.splitlines()[:12]:
        stripped = line.strip()
        letters = [char for char in stripped if char.isalpha()]
        if len(stripped) < 4 or len(letters) < 4:
            continue
        if sum(char.isupper() for char in letters) / len(letters) > 0.8 and any(
            hint in stripped.lower() for hint in _ISSUER_HINTS
        ):
            return stripped[:160]
    return None


def _detect_signers(blocks: list[Block]) -> list[str]:
    """Names printed near a signing role in the tail of the document (low confidence)."""
    tail = [b for b in blocks if b.block_type in ("text", "ocr_page")][-10:]
    lines = [line.strip() for block in tail for line in block.content.splitlines() if line.strip()]
    signers: list[str] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        if not (any(role in lowered for role in _SIGNER_ROLES) or any(marker in lowered for marker in _SIGN_MARKERS)):
            continue
        for candidate_line in lines[index : index + 4]:
            for name in extract_entity_candidates(candidate_line.title() if candidate_line.isupper() else candidate_line):
                if 2 <= len(name.split()) <= 5 and name not in signers:
                    signers.append(name)
    return signers[:5]


def _extract_keywords(blocks: list[Block], ledger: EntityLedger, limit: int = 12) -> list[str]:
    keywords: list[str] = []
    for block in blocks:
        if block.block_type == "heading":
            heading = re.sub(r"^#+\s*", "", block.content.strip())
            if 3 <= len(heading) <= 80 and heading not in keywords:
                keywords.append(heading)
    bigrams: Counter[str] = Counter()
    for block in blocks:
        if block.block_type not in ("text", "ocr_page", "caption"):
            continue
        words = [w.lower() for w in _WORD_RE.findall(block.content)]
        for first, second in zip(words, words[1:]):
            if first in _STOPWORDS or second in _STOPWORDS or len(first) < 2 or len(second) < 2:
                continue
            bigrams[f"{first} {second}"] += 1
    keywords.extend(term for term, count in bigrams.most_common(limit) if count >= 3 and term not in keywords)
    keywords.extend(term for term in ledger.terms(limit=limit, max_chars=400) if term not in keywords)
    return keywords[: limit * 2]


def _classify(document_text: str, ledger: EntityLedger, signers: list[str]) -> list[dict]:
    locations = {normalize_key(f"{kind} {name}"): f"{kind} {name.strip()}" for kind, name in _LOCATION_RE.findall(document_text)}
    persons = {normalize_key(name): name.strip() for _, name in _PERSON_PREFIX_RE.findall(document_text)}
    signer_keys = {normalize_key(name) for name in signers}
    entities = []
    for item in ledger.entities():
        key = normalize_key(item["canonical"])
        if key in signer_keys:
            kind = "person"
        elif key in persons or any(key.endswith(p) for p in persons):
            kind = "person"
        elif any(loc.endswith(key) or key in loc for loc in locations):
            kind = "location"
        else:
            kind = "other"
        entities.append({**item, "kind": kind, "key": key})
    return entities


def build_document_profile(document: ParsedDocument, max_entities: int = 150) -> dict:
    """Return the profile dict stored as ``document.metadata['profile']``."""
    blocks = _body_blocks(document)
    ledger = EntityLedger(min_count=1)
    for block in blocks:
        ledger.add(block.content)
    head = _first_pages_text(blocks)
    number = _DOC_NUMBER_RE.search(head)
    signers = _detect_signers(blocks)
    entities = _classify("\n".join(b.content for b in blocks), ledger, signers)
    # Names repeated or typed with a role/place prefix are far more reliable than singletons.
    entities = [e for e in entities if e["count"] >= 2 or e["kind"] != "other"][:max_entities]
    return {
        "version": PROFILE_VERSION,
        "title": _detect_title(blocks, document.source_file),
        "doc_number": number.group(1) if number else None,
        "issue_date": _detect_issue_date(head),
        "issuer": _detect_issuer(head),
        "signers": signers,
        "keywords": _extract_keywords(blocks, ledger),
        "entities": entities,
        "name_variants": ledger.conflicts(),
    }


def summarize_profile(profile: dict) -> dict:
    """Small subset copied to each chunk's metadata/Qdrant payload."""
    return {key: profile.get(key) for key in ("title", "doc_number", "issue_date", "issuer", "signers")}


def attach_profile_to_chunks(chunks: list[Chunk], profile: dict, max_entities_per_chunk: int = 12) -> None:
    """Annotate chunks with the document keywords and the entities they actually mention."""
    entities = profile.get("entities") or []
    keywords = list(profile.get("keywords") or [])
    summary = summarize_profile(profile)
    for chunk in chunks:
        haystack = f" {normalize_key(chunk.content)} "
        mentioned = [e["canonical"] for e in entities if f" {e['key']} " in haystack]
        for alias_source in entities:
            if alias_source["canonical"] in mentioned:
                continue
            if any(f" {normalize_key(alias)} " in haystack for alias in alias_source.get("aliases") or []):
                mentioned.append(alias_source["canonical"])
        chunk.entities = mentioned[:max_entities_per_chunk]
        chunk.keywords = keywords
        chunk.metadata["profile"] = summary


def entity_keys(profile: dict) -> list[tuple[str, str, str, list[str], int]]:
    """(kind, canonical, key, aliases, mentions) rows for the document_entities table."""
    rows = []
    for item in profile.get("entities") or []:
        rows.append((item.get("kind", "other"), item["canonical"], item["key"], item.get("aliases") or [], item["count"]))
    return rows
