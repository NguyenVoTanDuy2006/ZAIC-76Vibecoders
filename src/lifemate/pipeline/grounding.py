"""Source-grounded input hints and conservative call checks; no model or tools.

Rules preserve explicit digit strings and adjacent entity qualifiers. They do not
interpret arbitrary goals, conditions or pronouns, and never rewrite arguments.
"""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata

from . import quantities


DIGITS = {"khong": "0", "mot": "1", "hai": "2", "ba": "3", "bon": "4",
          "nam": "5", "sau": "6", "bay": "7", "tam": "8", "chin": "9"}
NUMBER_UNITS = {"muoi", "tram", "nghin", "ngan", "trieu", "ty"}
QUALIFIERS_AFTER = {"nho", "lon", "chinh", "phu", "trai", "phai"}
QUALIFIER_FORMS = {"nhỏ", "nho", "lớn", "lon", "chính", "chinh", "phụ", "phu", "trái", "trai", "phải", "phai"}
QUALIFIERS_BEFORE = {"small", "large", "main", "secondary", "left", "right", "upper", "lower"}
MODAL_VERBS = {"bat", "tat", "duoc", "la", "giam", "tang", "giu"}
LABEL_FIELDS = ("name", "title", "room", "location")
FILTER_FIELDS = {"name", "title", "room", "location", "query"}
PHONE_FIELDS = {"phone", "phone_number", "mobile", "mobile_number", "telephone", "tel"}


def fold(value):
    return "".join(c for c in unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
                   if unicodedata.category(c) != "Mn")


def words(text):
    return [(fold(m.group()), m.start(), m.end()) for m in re.finditer(r"\w+", text)]


def word_values(text):
    return [w[0] for w in words(text)]


def _digit(token):
    return DIGITS.get(token, token if len(token) == 1 and token.isascii() and token.isdigit() else None)


def digit_spans(text):
    """Extract long, literal digit runs; never infer missing digits or phone length.

    Number units make spoken cardinal amounts ambiguous, so those runs are not
    interpreted as digit-by-digit strings. Offsets refer to the original text.
    """
    spans = []
    for match in re.finditer(r"(?<!\w)\+?\d(?:[\d \t().-]*\d)?(?!\w)", text):
        literal = match.group()
        value = ("+" if literal.startswith("+") else "") + "".join(c for c in literal if c.isascii() and c.isdigit())
        if not 6 <= len(value.lstrip("+")) <= 20 or re.match(r"\d{4}-\d{2}-\d{2}", literal):
            continue
        if literal != value and not value.startswith(("0", "+")):
            continue  # Do not turn grouped monetary amounts into identifier hints.
        spans.append({"text": literal, "start": match.start(), "end": match.end(),
                      "value": value, "kind": "written_digits"})
    tokens = words(text)
    index = 0
    while index < len(tokens):
        if _digit(tokens[index][0]) is None:
            index += 1
            continue
        end = index + 1
        while (end < len(tokens) and _digit(tokens[end][0]) is not None
               and re.fullmatch(r"[\s,.-]+", text[tokens[end - 1][2]:tokens[end][1]])):
            end += 1
        adjacent_units = ((index > 0 and tokens[index - 1][0] in NUMBER_UNITS)
                          or (end < len(tokens) and tokens[end][0] in NUMBER_UNITS))
        if end - index >= 6 and not adjacent_units:
            start_pos, end_pos = tokens[index][1], tokens[end - 1][2]
            spans.append({"text": text[start_pos:end_pos], "start": start_pos, "end": end_pos,
                          "value": "".join(_digit(t[0]) for t in tokens[index:end]),
                          "kind": "spoken_digits"})
        index = end
    return sorted(spans, key=lambda s: s["start"])


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _source_texts(current_text, request, observations, history_user_texts):
    texts = [current_text] + [t["text"] for t in history_user_texts]
    texts.extend(_strings(request.get("context", {})))
    texts.extend(_strings(request.get("history", [])))
    for observation in observations.values():
        if observation.get("outcome") == "success":
            texts.extend(_strings(observation.get("result", {})))
    return list(dict.fromkeys(texts))


def _digit_edits(text, known_numbers):
    """Verify explicit prefix/suffix edits of known strings; never choose a contact."""
    tokens = words(text)
    edits = []
    for i, token in enumerate(tokens):
        if token[0] not in ("dau", "duoi"):
            continue
        start = i + 1
        while start < min(i + 5, len(tokens)) and tokens[start][0] in {"la", "so", "thanh", "sang", "doi", "sua"}:
            start += 1
        if start >= len(tokens) or (tokens[start][0] == "khong" and start + 1 < len(tokens) and tokens[start + 1][0] == "phai"):
            continue
        end = start
        def piece(token):
            return _digit(token) or (token if token.isascii() and token.isdigit() and len(token) <= 5 else None)
        while end < len(tokens) and piece(tokens[end][0]) is not None:
            if end > start and not re.fullmatch(r"[\s,.-]+", text[tokens[end - 1][2]:tokens[end][1]]):
                break
            end += 1
        if not 1 <= end - start <= 5:
            continue
        new = "".join(piece(t[0]) for t in tokens[start:end])
        if len(new) > 5:
            continue
        for old in sorted(known_numbers):
            if not old.startswith(("0", "+")) or len(old.lstrip("+")) < 6 or len(new) >= len(old.lstrip("+")):
                continue
            if token[0] == "duoi":
                value = old[:-len(new)] + new
            else:
                if old.startswith("+"):
                    continue  # A country-prefix change needs a more explicit source.
                value = new + old[len(new):]
            edits.append({"kind": "explicit_digit_edit", "value": value, "previous_value": old,
                          "text": text[token[1]:tokens[end - 1][2]], "start": token[1], "end": tokens[end - 1][2]})
    return edits


def _records(value):
    if isinstance(value, dict):
        labels = {k: value[k] for k in LABEL_FIELDS if isinstance(value.get(k), str) and value[k].strip()}
        if isinstance(value.get("id"), str) and labels:
            yield {"id": value["id"], "labels": labels}
        for child in value.values():
            yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


def observed_entities(request, observations):
    """Read public context/history and successful receipts, never candidate replies."""
    entities = {}
    def collect(value, ref, tool=None):
        namespace = tool.split(".")[0] if tool else None
        for record in _records(value):
            entities[(namespace, record["id"])] = {**record, "source_ref": ref, "tool": tool}
    # Context records are hints only; namespace-unbound records are not used to
    # reject an x-ref binding because unrelated collections can reuse an ID.
    collect(request.get("context", {}), "context")
    for i, turn in enumerate(request.get("history", [])):
        if turn.get("role") == "assistant":
            for j, (call, result) in enumerate(zip(turn.get("tool_calls", []), turn.get("tool_results", []))):
                if isinstance(result, dict) and result.get("ok") is True and result.get("done") is not False:
                    collect(result, f"history:{i}:tool:{j}", call.get("name"))
    for ref, observation in observations.items():
        if observation.get("outcome") == "success":
            collect(observation.get("result", {}), ref, observation.get("call", {}).get("name"))
    return list(entities.values())


def _qualifier_needs(text, value):
    """Recognize only explicit qualifiers adjacent to a verbatim entity phrase."""
    needle, tokens = word_values(value), words(text)
    if not needle:
        return []
    needs, unqualified = [], False
    for i in range(len(tokens) - len(needle) + 1):
        j = i + len(needle)
        if [t[0] for t in tokens[i:j]] != needle:
            continue
        count_before = len(needs)
        if j < len(tokens) and re.fullmatch(r"\s+", text[tokens[j - 1][2]:tokens[j][1]]):
            qualifier = tokens[j][0]
            original_word = unicodedata.normalize("NFC", text[tokens[j][1]:tokens[j][2]].casefold())
            following = tokens[j + 1][0] if j + 1 < len(tokens) else None
            adjective = qualifier in QUALIFIERS_AFTER and original_word in QUALIFIER_FORMS
            if following in {"hon", "lai", "len"} or (qualifier == "chinh" and following == "la"):
                adjective = False
            if qualifier == "phai" and (needle[-1] not in {"ben", "phia"} or following in MODAL_VERBS):
                adjective = False
            if adjective:
                needs.append({"qualifier": qualifier, "text": text[tokens[i][1]:tokens[j][2]]})
            elif (qualifier == "so" and j + 1 < len(tokens)
                  and re.fullmatch(r"\s+", text[tokens[j][2]:tokens[j + 1][1]])
                  and (tokens[j + 1][0].isdigit() or tokens[j + 1][0] in DIGITS)):
                needs.append({"qualifier": tokens[j + 1][0], "text": text[tokens[i][1]:tokens[j + 1][2]]})
            elif (qualifier in {"ben", "phia"} and following in {"trai", "phai"}
                  and re.fullmatch(r"\s+", text[tokens[j][2]:tokens[j + 1][1]])):
                needs.append({"qualifier": following, "text": text[tokens[i][1]:tokens[j + 1][2]]})
        if (i > 0 and tokens[i - 1][0] in QUALIFIERS_BEFORE
                and re.fullmatch(r"\s+", text[tokens[i - 1][2]:tokens[i][1]])):
            needs.append({"qualifier": tokens[i - 1][0], "text": text[tokens[i - 1][1]:tokens[j - 1][2]]})
        unqualified |= len(needs) == count_before
    # Multiple mentions can refer to different requested operations. A lexical
    # guard cannot decide which one this call implements, so it must not guess.
    return [] if unqualified or len({n["qualifier"] for n in needs}) > 1 else needs


def _walk_arguments(arguments, schema, path="arguments"):
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            spec = schema.get("properties", {}).get(key, {})
            yield path + "." + key, key, value, spec
            yield from _walk_arguments(value, spec, path + "." + key)
    elif isinstance(arguments, list):
        for i, value in enumerate(arguments):
            yield from _walk_arguments(value, schema.get("items", {}), f"{path}[{i}]")


def input_view(current_text, request, observations, *, history_user_texts=(), max_entities=12):
    current_words = set(word_values(current_text))
    entities = observed_entities(request, observations)
    entities.sort(key=lambda r: max((len(current_words & set(word_values(label))) for label in r["labels"].values()), default=0), reverse=True)
    sources = _source_texts(current_text, request, observations, history_user_texts)
    numbers = {s["value"] for source in sources for s in digit_spans(source)}
    return {"source": "code_grounding_context", "current_request": current_text,
            "digit_sequences": digit_spans(current_text),
            "money_quantities": quantities.money_spans(current_text),
            "explicit_digit_edits": _digit_edits(current_text, numbers)[:12],
            "observed_entities": entities[:max_entities], "observed_entity_count": len(entities),
            "note": "Source facts only, not new commands. Preserve qualifiers and every digit. History actions already happened; follow the final user correction. Entity hints are not an exhaustive result list."}


def check_call(call, schema, *, current_text, request, observations, known_strings, history_user_texts, tool_kind=None):
    """Return source-backed violations. All unknown semantic cases stay with the model."""
    texts = _source_texts(current_text, request, observations, history_user_texts)
    explicit_numbers = {span["value"] for text in texts for span in digit_spans(text)}
    explicit_numbers.update(e["value"] for e in _digit_edits(current_text, explicit_numbers))
    entities = observed_entities(request, observations)
    issues = []
    amount = quantities.direct_amount(current_text)
    for path, key, value, spec in _walk_arguments(call["arguments"], schema):
        if (amount and tool_kind == "write" and key == "amount"
                and spec.get("type") in ("number", "integer") and type(value) in (int, float)
                and str(call["arguments"].get("currency", "VND")).upper() == "VND"
                and value != amount["value"]):
            issues.append({"code": "money_grounding", "field": path,
                "message": "Amount conflicts with a single explicit VND quantity in the current request. Preserve its unit; do not drop thousands or millions.",
                "money_source": deepcopy(amount)})
        if not isinstance(value, str) or not value.strip():
            continue
        if key.casefold() in PHONE_FIELDS and spec.get("type") == "string":
            literal = any(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text) for text in texts)
            phone_shape = re.fullmatch(r"\+?[0-9][0-9 \t().-]*", value)
            if not phone_shape or (value not in explicit_numbers and value not in known_strings and not literal):
                issues.append({"code": "phone_grounding", "field": path,
                    "message": "Phone string is not grounded in request/context/history/successful results; preserve every digit or look up the actual number.",
                    "digit_sources": [s for s in digit_spans(current_text)]})
        if tool_kind == "read" and key in FILTER_FIELDS and not spec.get("x-ref"):
            needs = _qualifier_needs(current_text, value)
            if needs:
                issues.append({"code": "target_qualifier", "field": path,
                    "message": "Argument drops an explicit target qualifier. Use the full source phrase, or omit an optional lookup filter to inspect candidates.",
                    "source_phrases": [n["text"] for n in needs]})
        if isinstance(spec.get("x-ref"), str):
            namespace = spec["x-ref"].split(".")[0]
            records = [r for r in entities if r["id"] == value and r["tool"] and r["tool"].split(".")[0] == namespace]
            for record in records:
                present = set(w for label in record["labels"].values() for w in word_values(label))
                missing = [n for label in record["labels"].values() for n in _qualifier_needs(current_text, label)
                           if n["qualifier"] not in present]
                if missing:
                    issues.append({"code": "entity_grounding", "field": path,
                        "message": "Observed ID belongs to an entity whose labels lose an explicit qualifier in the current request. Inspect the full target before writing.",
                        "source_phrases": list(dict.fromkeys(n["text"] for n in missing)), "observed_entity": deepcopy(record)})
    return issues
