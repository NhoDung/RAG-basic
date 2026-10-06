from __future__ import annotations

import re

from .config import GuardrailConfig
from .models import SearchHit
from .utils import extract_numbers


REFUSAL_TEXT = "Tôi không tìm thấy đủ thông tin trong tài liệu được cung cấp."

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"you are now",
    r"\bact as (an?|the) ",
    r"(reveal|show|print) (the |your )?(system prompt|instructions)",
    r"jailbreak",
    r"bỏ qua (tất cả |mọi |các )?(hướng dẫn|chỉ dẫn|quy tắc|yêu cầu)( trước| ở trên)?",
    r"quên (hết |tất cả )?(các )?(hướng dẫn|quy tắc)",
    r"(tiết lộ|hiển thị|in ra) (system prompt|lời nhắc hệ thống|hướng dẫn hệ thống)",
    r"từ bây giờ bạn là",
    r"<\|im_(start|end)\|>",
    r"\[/?(system|inst)\]",
]
_INJECTION = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]

PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("ID", re.compile(r"\b\d{12}\b|\b\d{9}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){9,10}(?!\d)")),
]


def detect_prompt_injection(text: str) -> list[str]:
    return [pattern.pattern for pattern in _INJECTION if pattern.search(text or "")]


def sanitize_context(text: str) -> str:
    """Neutralize chat-template tokens and fake source markers inside documents."""
    text = re.sub(r"<\|im_(start|end)\|>", "<im>", text or "")
    return re.sub(r"\[(SOURCE_\d+)\]", r"(\1)", text)


def mask_pii(text: str) -> str:
    if not text:
        return text
    for label, pattern in PII_PATTERNS:
        text = pattern.sub(f"<{label}>", text)
    return text


def retrieval_confidence(hits: list[SearchHit], config: GuardrailConfig, reranker_enabled: bool) -> tuple[bool, str]:
    """Refuse when there is no evidence or the best evidence is too weak."""
    if not hits:
        return False, "no_hits"
    if not config.enabled:
        return True, "disabled"
    if reranker_enabled:
        best = max((hit.rerank_score for hit in hits if hit.rerank_score is not None), default=None)
        if best is not None and best < config.min_rerank_score:
            return False, f"low_rerank_score:{best:.3f}"
    best_rrf = max(hit.rrf_score for hit in hits)
    if best_rrf < config.min_rrf_score:
        return False, f"low_rrf_score:{best_rrf:.4f}"
    return True, "ok"


def unsupported_numbers(answer: str, context: str) -> list[str]:
    """Numbers in the answer that do not appear in the context (or are SOURCE ids)."""
    answer_wo_ids = re.sub(r"SOURCE_\d+", "", answer or "")
    context_numbers = {round(value, 6) for value in extract_numbers(context)}
    result = []
    for match in re.finditer(r"\d[\d.,]*%?", answer_wo_ids):
        token = match.group(0).rstrip(".,")
        values = extract_numbers(token)
        if not values:
            continue
        value = round(values[0], 6)
        if value in context_numbers:
            continue
        if len(token) <= 2 and value <= 31:
            continue  # Ordinals/list markers such as "bước 2".
        result.append(token)
    return list(dict.fromkeys(result))


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return "không tìm thấy đủ thông tin" in lowered or "không có thông tin" in lowered
