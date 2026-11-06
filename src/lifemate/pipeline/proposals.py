"""Compile concise model proposals into the strict, versioned internal plan.

Only omitted protocol fields get defaults. No business conditions, dependencies,
tool arguments or predicates are inferred or silently repaired here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

from . import predicates
from .planning import ControllerError, identifier


STEP_DEFAULTS = {"depends_on": [], "when": None, "expected_arguments": {},
                 "postcondition": None, "operation_id": None}
STEP_REQUIRED = {"id", "goal_id", "kind", "tool"}
GOAL_REQUIRED = {"id", "description", "kind", "source"}


def operation_id(step_id):
    """Reserve the same identity before and after a deferred write is expanded."""
    return "op-" + hashlib.sha256(step_id.encode()).hexdigest()[:24]


def contract(plan_version):
    """Small model-facing contract; runtime validation remains independent."""
    return {"required": ["reason", "goals", "steps"],
            "reason": "one short sentence",
            "goal_required": sorted(GOAL_REQUIRED),
            "step_required": sorted(STEP_REQUIRED),
            "step_defaults_if_omitted": {k: deepcopy(v) for k, v in STEP_DEFAULTS.items() if k != "operation_id"},
            "step_kinds": ["read", "write", "answer", "maintain", "deferred"],
            "base_version_owned_by_code": plan_version,
            "operation_id_owned_by_code": "stable per step; preserve existing identities on revisions"}


def compile_proposal(value, *, plan_version, previous_steps=None):
    """Return (internal plan, applied defaults), or all independent shape faults.

    Explicit legacy versions/operation IDs are accepted but never coerced. The
    caller must still run parse_plan before installing anything.
    """
    if not isinstance(value, dict) or not {"reason", "goals", "steps"} <= set(value) or set(value) - {"reason", "goals", "steps", "base_version"}:
        raise ControllerError("plan_contract", "Proposal needs reason, goals, steps only; code supplies base_version", category="format")
    issues = []

    def add(code, path, message, *, sid=None, category="plan"):
        if len(issues) < 24:
            issues.append({"code": code, "field": path, "message": message,
                           "step_id": sid, "category": category})

    if "base_version" in value and (type(value["base_version"]) is not int or value["base_version"] != plan_version):
        add("stale_version", "base_version", f"Expected integer {plan_version}; omit this code-owned field")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        add("reason", "reason", "Expected a short nonempty string", category="format")
    for key in ("goals", "steps"):
        if not isinstance(value[key], list) or not value[key]:
            add("plan_size", key, "Expected a nonempty array of objects", category="format")
    result = deepcopy(value)
    result["base_version"] = plan_version
    defaults = ["base_version"] if "base_version" not in value else []
    for index, goal in enumerate(value["goals"] if isinstance(value["goals"], list) else []):
        if not isinstance(goal, dict) or set(goal) != GOAL_REQUIRED:
            add("goal_contract", f"goals[{index}]", "Expected exactly id, description, kind, source", category="format")
    candidates = result["steps"] if isinstance(result["steps"], list) else []
    step_ids = {s["id"] for s in candidates if isinstance(s, dict) and identifier(s.get("id"))}
    for index, step in enumerate(candidates):
        path = f"steps[{index}]"
        if not isinstance(step, dict):
            add("step_contract", path, "Expected an object", category="format")
            continue
        sid = step.get("id") if identifier(step.get("id")) else None
        missing, extra = STEP_REQUIRED - set(step), set(step) - STEP_REQUIRED - set(STEP_DEFAULTS)
        if missing or extra:
            add("step_contract", path, f"Missing fields: {sorted(missing)}; unexpected fields: {sorted(extra)}", sid=sid, category="format")
        for key, default in STEP_DEFAULTS.items():
            if key not in step:
                if key == "operation_id" and step.get("kind") == "write" and sid:
                    old = (previous_steps or {}).get(sid, {})
                    default = old.get("operation_id") or operation_id(sid)
                step[key] = deepcopy(default)
                defaults.append(f"{path}.{key}")
        deps = step["depends_on"]
        if not isinstance(deps, list) or any(not identifier(d) for d in deps) or len(deps) != len(set(deps)):
            add("dependencies", path + ".depends_on", "Expected an array of unique step ID strings; use [] for no dependencies, never null or a string", sid=sid)
        if not isinstance(step["expected_arguments"], dict):
            add("expected_arguments", path + ".expected_arguments", "Expected an object; use {} when arguments are not grounded yet", sid=sid)
        if step.get("kind") != "write" and step["operation_id"] is not None:
            add("operation_id", path + ".operation_id", "Only writes have operation IDs; omit this field for code to supply it", sid=sid)
        when = step["when"]
        if when is not None and (not isinstance(when, dict) or set(when) != {"source", "predicate"}):
            add("condition_contract", path + ".when", "Expected null or an object with source and predicate", sid=sid)
        expressions = [("postcondition", step["postcondition"])]
        if isinstance(when, dict) and "predicate" in when:
            expressions.append(("when.predicate", when["predicate"]))
        for field, expr in expressions:
            if expr is None:
                continue
            try:
                predicates.validate(expr, step_ids)
            except (predicates.PredicateError, TypeError) as error:
                add("predicate", path + "." + field, f"{error}; use a typed JSON predicate, not prose; literals use {{\"value\": ...}}", sid=sid)
    if issues:
        first = issues[0]
        raise ControllerError(first["code"], "; ".join(f"{i['field']}: {i['message']}" for i in issues),
                              step_id=first["step_id"], field=first["field"],
                              category=first["category"], violations=issues)
    return result, defaults
