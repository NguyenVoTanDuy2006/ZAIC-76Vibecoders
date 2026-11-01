from __future__ import annotations

import unicodedata


def normalize_text(text: str, config: dict | None = None) -> str:
    config = config or {}
    normalized = unicodedata.normalize(config.get("unicode_form", "NFC"), text)
    if config.get("lowercase", True):
        normalized = normalized.lower()
    if config.get("punctuation_to_space", True):
        normalized = "".join(" " if unicodedata.category(char).startswith("P") else char for char in normalized)
    if config.get("collapse_whitespace", True):
        normalized = " ".join(normalized.split())
    return normalized
