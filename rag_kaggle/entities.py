"""Deterministic proper-name extraction used for glossaries and document profiles.

No LLM is involved: names are Title Case word runs, grouped by their
diacritic-stripped form so that OCR variants of one name ("Nguyen Van A",
"Nguyễn Văn A") share a key and the best-attested spelling becomes canonical.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Words that start a sentence or introduce a name but are not part of it.
LEADING_STOPWORDS = frozenset(
    {
        "ông", "bà", "anh", "chị", "đồng", "chí", "theo", "căn", "cứ", "của", "về", "tại", "do",
        "và", "các", "những", "với", "cho", "trong", "khi", "nếu", "đã", "sẽ", "được", "bị",
        "kính", "gửi", "tên", "ngày", "tháng", "năm", "số",
    }
)
_FRAGMENT_SPLIT = re.compile(r"[^\w ]+")


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _is_name_word(word: str) -> bool:
    return len(word) > 1 and word[0].isupper() and any(char.islower() for char in word) and not word.isdigit()


def extract_entity_candidates(text: str) -> list[str]:
    """Return Title Case runs of at least two words (e.g. "Nguyễn Văn A", "Hòa Bình")."""
    text = unicodedata.normalize("NFC", text)
    found: list[str] = []
    for line in text.splitlines():
        for fragment in _FRAGMENT_SPLIT.split(line):
            run: list[str] = []
            for word in [*fragment.split(), ""]:
                if word and _is_name_word(word):
                    run.append(word)
                    continue
                while run and run[0].lower() in LEADING_STOPWORDS:
                    run.pop(0)
                if len(run) >= 2:
                    found.append(" ".join(run))
                run = []
    return found


def _spelling_score(variant: str, count: int) -> tuple[int, int]:
    """Prefer the most frequent spelling, then the one that kept more diacritics."""
    return count, sum(1 for char in variant if ord(char) > 127)


class EntityLedger:
    """Counts entity spellings across one document and reports canonical forms."""

    def __init__(self, min_count: int = 2):
        self.min_count = min_count
        self._variants: dict[str, Counter[str]] = {}

    def add(self, text: str) -> None:
        for candidate in extract_entity_candidates(text):
            key = strip_diacritics(candidate).lower()
            self._variants.setdefault(key, Counter())[candidate] += 1

    def canonical(self, key: str) -> str:
        variants = self._variants[key]
        return max(variants, key=lambda variant: _spelling_score(variant, variants[variant]))

    def terms(self, limit: int = 20, max_chars: int = 400) -> list[str]:
        """Canonical spellings seen at least ``min_count`` times, most frequent first."""
        ranked = sorted(
            (key for key, variants in self._variants.items() if sum(variants.values()) >= self.min_count),
            key=lambda key: (-sum(self._variants[key].values()), key),
        )
        result: list[str] = []
        used = 0
        for key in ranked:
            term = self.canonical(key)
            if len(result) >= limit or used + len(term) + 2 > max_chars:
                break
            result.append(term)
            used += len(term) + 2
        return result

    def conflicts(self) -> dict[str, list[str]]:
        """Canonical spelling -> other spellings of the same name seen in the document."""
        report: dict[str, list[str]] = {}
        for key, variants in self._variants.items():
            if len(variants) > 1:
                canonical = self.canonical(key)
                report[canonical] = sorted(variant for variant in variants if variant != canonical)
        return report

    def entities(self, min_count: int = 1) -> list[dict]:
        """All entities with their canonical form, aliases and mention count."""
        items = []
        for key, variants in self._variants.items():
            total = sum(variants.values())
            if total < min_count:
                continue
            canonical = self.canonical(key)
            items.append(
                {
                    "canonical": canonical,
                    "aliases": sorted(variant for variant in variants if variant != canonical),
                    "count": total,
                }
            )
        return sorted(items, key=lambda item: (-item["count"], item["canonical"]))
