"""Small JSON predicate language. Missing evidence is unknown; never eval model code."""
from __future__ import annotations

from copy import deepcopy
import math


class PredicateError(ValueError):
    pass


UNKNOWN = object()
OPS = {"eq", "ne", "lt", "lte", "gt", "gte", "contains"}


def validate(expression, step_ids, *, depth=0):
    if depth > 6 or not isinstance(expression, dict):
        raise PredicateError("Predicate must be a JSON object, depth <= 6")
    op = expression.get("op")
    if op in ("all", "any"):
        if set(expression) != {"op", "items"} or not isinstance(expression["items"], list) or not 1 <= len(expression["items"]) <= 12:
            raise PredicateError("all/any needs 1..12 items")
        for child in expression["items"]:
            validate(child, step_ids, depth=depth + 1)
    elif op == "not":
        if set(expression) != {"op", "item"}:
            raise PredicateError("not needs item")
        validate(expression["item"], step_ids, depth=depth + 1)
    elif op in OPS:
        if set(expression) != {"op", "left", "right"}:
            raise PredicateError("Comparison needs op, left, right")
        for value in (expression["left"], expression["right"]):
            if not isinstance(value, dict):
                raise PredicateError("Operand needs value or step/path")
            if set(value) == {"value"}:
                continue
            if set(value) != {"step", "path"} or value["step"] not in step_ids:
                raise PredicateError("Operand references an unknown step")
            path = value["path"]
            if not isinstance(path, list) or len(path) > 16 or any(type(k) not in (str, int) or (type(k) is int and k < 0) for k in path):
                raise PredicateError("Path must contain field names/nonnegative indexes")
        if not any("step" in value for value in (expression["left"], expression["right"])):
            raise PredicateError("A predicate comparison must reference observed step data, not just literals")
    else:
        raise PredicateError("Unknown predicate operator")


def references(expression):
    if not expression:
        return set()
    if expression["op"] in ("all", "any"):
        return set().union(*(references(e) for e in expression["items"]))
    if expression["op"] == "not":
        return references(expression["item"])
    return {v["step"] for v in (expression["left"], expression["right"]) if "step" in v}


def operand(value, results):
    if "value" in value:
        return deepcopy(value["value"])
    current = results.get(value["step"], UNKNOWN)
    for key in value["path"]:
        if isinstance(current, dict) and type(key) is str and key in current:
            current = current[key]
        elif isinstance(current, list) and type(key) is int and key < len(current):
            current = current[key]
        else:
            return UNKNOWN
    return current


def _equal(left, right):
    # JSON booleans must not equal 0/1, including inside collections. Numeric
    # int/float values can represent the same quantity in result predicates.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def evaluate(expression, results):
    """Return true/false/unknown. Three-valued logic avoids false empty-lookup claims."""
    if expression is None:
        return "true"
    op = expression["op"]
    if op in ("all", "any"):
        values = [evaluate(e, results) for e in expression["items"]]
        decisive = "false" if op == "all" else "true"
        if decisive in values:
            return decisive
        return "unknown" if "unknown" in values else ("true" if op == "all" else "false")
    if op == "not":
        return {"true": "false", "false": "true", "unknown": "unknown"}[evaluate(expression["item"], results)]
    left, right = operand(expression["left"], results), operand(expression["right"], results)
    if left is UNKNOWN or right is UNKNOWN:
        return "unknown"
    if op in ("lt", "lte", "gt", "gte"):
        if type(left) not in (int, float) or type(right) not in (int, float):
            return "unknown"
        if (type(left) is float and not math.isfinite(left)) or (type(right) is float and not math.isfinite(right)):
            return "unknown"
    try:
        if op == "eq":
            answer = _equal(left, right)
        elif op == "ne":
            answer = not _equal(left, right)
        elif op == "contains":
            if not isinstance(left, (list, str, dict)):
                return "unknown"
            answer = (any(_equal(value, right) for value in left) if isinstance(left, list)
                      else isinstance(right, str) and right in left)
        else:
            answer = {"lt": lambda: left < right, "lte": lambda: left <= right,
                      "gt": lambda: left > right, "gte": lambda: left >= right}[op]()
    except (TypeError, ValueError):
        return "unknown"
    return "true" if answer else "false"
