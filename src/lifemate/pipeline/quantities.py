"""Explicit Vietnamese money quantities with source offsets; no guessed values.

Hints may contain several amounts. Enforcement is restricted to one unambiguous
direct amount, never thresholds, arithmetic, corrections or multi-amount requests.
"""
from decimal import Decimal, InvalidOperation
import re
import unicodedata

VERSION = "explicit-vnd-quantities-v1"
_DIGITS = {"khong": 0, "mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4,
           "nam": 5, "lam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9}
_UNITS = {"nghin": 1000, "ngan": 1000, "k": 1000, "trieu": 1000000,
          "ty": 1000000000, "dong": 1, "vnd": 1}
_COUNT_UNITS = {"buoc", "luot", "nguoi", "diem", "met", "km", "kg", "gam", "lit",
                "giay", "phut", "gio", "ngay", "thang", "nam", "cai", "chiec", "lan"}


def _fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
                   if unicodedata.category(c) != "Mn")


def _cardinal(tokens):
    """Only standard units/tens/hundreds, not colloquial or digit-by-digit guesses."""
    if not tokens:
        return None
    value = 0
    if len(tokens) >= 2 and tokens[0] in _DIGITS and tokens[1] == "tram":
        value = _DIGITS[tokens[0]] * 100
        tokens = tokens[2:]
        if tokens and tokens[0] in ("le", "linh"):
            tokens = tokens[1:]
    if not tokens:
        return value
    if tokens[0] == "muoi":
        value += 10
        tokens = tokens[1:]
    elif len(tokens) >= 2 and tokens[0] in _DIGITS and tokens[1] == "muoi":
        value += _DIGITS[tokens[0]] * 10
        tokens = tokens[2:]
    if len(tokens) == 1 and tokens[0] in _DIGITS:
        return value + _DIGITS[tokens[0]]
    return value if not tokens else None


def _numeric(text):
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
        text = text.replace(".", "").replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def money_spans(text):
    # Normalization preserves character positions for Vietnamese precomposed text
    # by using original token offsets, rather than offsets into a folded string.
    tokens = [(m.group(), _fold(m.group()), m.start(), m.end())
              for m in re.finditer(r"\d+(?:[.,]\d+)*|[^\W\d_]+|₫", text)]
    spans = []
    for i, (_, unit, _, end) in enumerate(tokens):
        unit = "dong" if unit == "₫" else unit
        if unit not in _UNITS or i == 0:
            continue
        if ((unit == "dong" and tokens[i][0][0].isupper())
                or (i + 1 < len(tokens) and tokens[i + 1][1] in _COUNT_UNITS)):
            continue  # Proper names and nonmonetary quantities are not VND facts.
        prev = tokens[i - 1]
        start = i - 1
        base = _numeric(prev[0]) if prev[0][0].isdigit() else None
        if base is None:
            while (start > 0 and tokens[start - 1][1] in {*_DIGITS, "tram", "muoi", "le", "linh"}
                   and text[tokens[start - 1][3]:tokens[start][2]].isspace()):
                start -= 1
            cardinal = _cardinal([t[1] for t in tokens[start:i]])
            base = Decimal(cardinal) if cardinal is not None else None
        if base is None or not re.fullmatch(r"\s*", text[prev[3]:tokens[i][2]]):
            continue
        if ((i + 1 < len(tokens) and (tokens[i + 1][1] in {*_DIGITS, "ruoi", "may"}
                                      or tokens[i + 1][0][0].isdigit()))
                or (start > 0 and tokens[start - 1][1] in _UNITS)):
            continue  # Compound/colloquial amounts must not yield a partial hint.
        value = base * _UNITS[unit]
        if value <= 0 or value != value.to_integral_value():
            continue
        spans.append({"text": text[tokens[start][2]:end], "start": tokens[start][2], "end": end,
                      "value": int(value), "currency": "VND", "kind": "explicit_money"})
    return spans


def direct_amount(text):
    spans = money_spans(text)
    folded = _fold(text)
    # Do not bind a price/threshold, changed amount, range, fraction or relative
    # instruction to an action's amount. Leave ambiguous semantics to the model.
    if (len(spans) != 1 or re.search(
            r"\b(neu|duoi|tren|hon|kem|toi da|toi thieu|gia|phi|khoang|tu|den|"
            r"con lai|so du|mot nua|phan tram|a khong|khong phai|thay vi|doi|sua|them|bot|"
            r"va|voi|roi|con|thi|dong thoi|sau do)\b|[%/]", folded)
            or re.search(r"\b(usd|dollar|eur|euro)\b", folded)):
        return None
    return spans[0]
