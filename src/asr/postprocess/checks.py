"""Conservative, deterministic checks for LM transcript edits."""
from __future__ import annotations

import math
import re
import unicodedata

from asr.common.metrics import edit_distance
from asr.common.text import normalize_text

NUMBERS = {
    "một", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy",
    "tám", "chín", "mười", "chục", "trăm", "nghìn", "ngàn", "triệu", "tỷ",
    "rưỡi", "linh", "lẻ",
}
LOGIC = {"không", "chưa", "đừng", "nếu", "thì", "trừ", "trước", "sau"}
UNITS = {"đồng", "nghìn", "ngàn", "triệu", "tỷ", "giờ", "phút", "ngày", "tháng"}
WORD_RE = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+", re.UNICODE)
NAME_RE = re.compile(r"(?<!\w)[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*(?!\w)")


def _words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(unicodedata.normalize("NFC", text))]


def _protected(text: str) -> tuple[list[str], list[str], list[str]]:
    words = _words(text)
    numbers = []
    for index, word in enumerate(words):
        is_number = any(c.isdigit() for c in word) or word in NUMBERS
        if is_number or (word in UNITS and index > 0 and
                         (any(c.isdigit() for c in words[index - 1]) or words[index - 1] in NUMBERS)):
            numbers.append(word)
    logic = [w for w in words if w in LOGIC]
    names = []
    for match in NAME_RE.finditer(text):
        prefix = text[:match.start()].rstrip()
        if prefix and prefix[-1] not in ".!?":
            names.append(match.group().lower())
    return numbers, logic, names


def hard_check(original: str, candidate: str, history: list[dict]) -> tuple[bool, str]:
    if not isinstance(candidate, str) or not candidate.strip():
        return False, "empty_candidate"
    if "\n" in candidate or "```" in candidate or "<|" in candidate:
        return False, "non_transcript_output"
    old = normalize_text(original).split()
    new = normalize_text(candidate).split()
    if not old or not new:
        return False, "empty_normalized_text"
    if len(new) < math.ceil(0.5 * len(old)) or len(new) > math.ceil(1.5 * len(old)):
        return False, "length_changed"
    if edit_distance(old, new) > max(2, math.ceil(0.4 * len(old))):
        return False, "too_many_word_edits"
    old_numbers, old_logic, old_names = _protected(original)
    new_numbers, new_logic, new_names = _protected(candidate)
    prior_text = " ".join(turn["text"] for turn in history).lower()
    if old_numbers != new_numbers:
        changed = [token for token in new_numbers if token not in old_numbers]
        if not changed or not all(token in _words(prior_text) for token in changed):
            return False, "number_or_time_changed"
    if old_logic != new_logic:
        return False, "logic_changed"
    if old_names != new_names:
        changed_names = [name for name in new_names if name not in old_names]
        if not changed_names or not all(name in prior_text for name in changed_names):
            return False, "name_changed"
    return True, "accepted"
