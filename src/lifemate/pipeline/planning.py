"""Versioned, source-grounded interpretation and plan contracts; no model or tools."""
from __future__ import annotations

from copy import deepcopy
import json

from . import predicates


class ControllerError(ValueError):
    def __init__(self, code, message, *, step_id=None, field=None, category="plan", violations=None):
        self.code, self.step_id, self.field, self.category = code, step_id, field, category
        self.violations = deepcopy(violations)
        super().__init__(message)

    def as_dict(self):
        value = {"code": self.code, "message": str(self), "step_id": self.step_id,
                 "field": self.field, "category": self.category}
        if self.violations is not None:
            value["violations"] = deepcopy(self.violations)
        return value


def fail(code, message, **kw):
    raise ControllerError(code, message, **kw)


def source(value, sources, *, field="source"):
    if not isinstance(value, dict) or set(value) != {"ref", "quote"}:
        fail("source", "Source needs ref and an exact nonempty quote", field=field)
    ref, quote = value["ref"], value["quote"]
    if not isinstance(ref, str) or ref not in sources or not isinstance(quote, str) or not quote.strip() or quote not in sources[ref]:
        fail("source", "Source quote must occur verbatim in the cited user request/history", field=field)


def identifier(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 64


def parse_plan(value, *, sources, tools, max_tasks, previous=None, reinterpret=False):
    """Validate the entire candidate before installing it. No partial mutations."""
    if not isinstance(value, dict) or set(value) != {"base_version", "goals", "steps", "reason"}:
        fail("plan_contract", "Plan needs base_version, goals, steps, reason", category="format")
    expected = previous.version if previous else 0
    if type(value["base_version"]) is not int or value["base_version"] != expected:
        fail("stale_version", f"Expected base_version={expected}", field="base_version")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        fail("reason", "Plan revision needs a reason", field="reason", category="format")
    for key in ("goals", "steps"):
        if not isinstance(value[key], list) or not 1 <= len(value[key]) <= max_tasks:
            fail("plan_size", f"{key} needs 1..{max_tasks} objects", field=key, category="format")
    goals, steps = {}, {}
    for goal in value["goals"]:
        if not isinstance(goal, dict) or set(goal) != {"id", "description", "kind", "source"}:
            fail("goal_contract", "Goal needs id, description, kind, source", category="format")
        gid = goal["id"]
        if not identifier(gid) or gid in goals or not isinstance(goal["description"], str) or not goal["description"].strip():
            fail("goal_id", "Goal IDs must be unique with a nonempty description")
        source(goal["source"], sources)
        if goal["kind"] not in ("read", "write", "answer", "maintain", "unsupported"):
            fail("goal_kind", "Invalid goal kind", field=gid)
        goals[gid] = deepcopy(goal)
    if previous:
        if not set(previous.goals) <= set(goals):
            fail("goal_coverage", "A revision cannot remove a requested goal; reinterpret it under the same ID")
        for gid, old in previous.goals.items():
            if goals[gid]["source"] != old["source"]:
                fail("goal_source", "Existing goal source anchors cannot be erased", field=gid)
            if goals[gid] != old and not reinterpret:
                fail("interpretation_change", "Use reinterpret to correct a goal interpretation", field=gid)
    fields = {"id", "goal_id", "kind", "tool", "depends_on", "when", "expected_arguments", "postcondition", "operation_id"}
    for step in value["steps"]:
        if not isinstance(step, dict) or set(step) != fields:
            fail("step_contract", "Step needs " + ", ".join(sorted(fields)), category="format")
        sid = step["id"]
        if not identifier(sid) or sid in steps:
            fail("step_id", "Step IDs must be unique")
        if not isinstance(step["goal_id"], str) or step["goal_id"] not in goals or step["kind"] not in ("read", "write", "answer", "maintain", "deferred"):
            fail("step_goal", "Unknown goal_id or kind", step_id=sid)
        name, kind = step["tool"], step["kind"]
        if kind in ("read", "write"):
            if not isinstance(name, str) or name not in tools or tools[name].get("kind") != kind:
                fail("step_tool", "Step tool must be allowed and match its kind", step_id=sid)
        elif name is not None:
            fail("step_tool", "answer/maintain/deferred steps have tool=null", step_id=sid)
        deps = step["depends_on"]
        if not isinstance(deps, list) or any(not identifier(d) for d in deps) or len(deps) != len(set(deps)):
            fail("dependencies", "depends_on must contain unique step IDs", step_id=sid)
        if not isinstance(step["expected_arguments"], dict):
            fail("expected_arguments", "Expected arguments must be a JSON object", step_id=sid)
        if kind == "deferred" and (not deps or step["expected_arguments"] or step["postcondition"] is not None):
            fail("deferred_contract", "Deferred steps need prerequisite steps, empty expected_arguments and postcondition=null", step_id=sid)
        if kind == "write" and not identifier(step["operation_id"]):
            fail("operation_id", "A write needs a stable operation_id", step_id=sid)
        if kind != "write" and step["operation_id"] is not None:
            fail("operation_id", "Only write steps have operation_id", step_id=sid)
        steps[sid] = deepcopy(step)
        if kind == "write" and goals[step["goal_id"]]["kind"] != "write":
            fail("goal_kind", "A write step needs an explicitly interpreted write goal", step_id=sid)
    covered = {s["goal_id"] for s in steps.values()}
    if covered != set(goals):
        fail("goal_coverage", "Every goal must remain covered by at least one step")
    for gid, goal in goals.items():
        effect = "answer" if goal["kind"] == "unsupported" else goal["kind"]
        if not any(s["goal_id"] == gid and (s["kind"] == effect or
                (s["kind"] == "deferred" and goal["kind"] != "unsupported")) for s in steps.values()):
            fail("goal_coverage", "Goal needs a matching effect step; use unsupported when no available tool can fulfill it", field=gid)
    operations = [s["operation_id"] for s in steps.values() if s["kind"] == "write"]
    if len(operations) != len(set(operations)):
        fail("operation_id", "Each write operation_id must appear once")
    def visit(sid, ancestors):
        if sid not in steps or sid in ancestors:
            fail("dependencies", "Unknown/cyclic dependency", step_id=sid)
        result = set()
        for dep in steps[sid]["depends_on"]:
            result.add(dep)
            result.update(visit(dep, ancestors | {sid}))
        return result
    for sid, step in steps.items():
        ancestors = visit(sid, set())
        when = step["when"]
        if when is not None:
            if not isinstance(when, dict) or set(when) != {"source", "predicate"}:
                fail("condition_contract", "when needs source and predicate", step_id=sid)
            source(when["source"], sources, field="when.source")
        for field, expression in (("when", when["predicate"] if when else None), ("postcondition", step["postcondition"])):
            if expression is None:
                continue
            try:
                predicates.validate(expression, set(steps))
            except (predicates.PredicateError, TypeError) as error:
                fail("predicate", str(error), step_id=sid, field=field)
            refs = predicates.references(expression)
            allowed = ancestors if field == "when" or step["kind"] in ("answer", "maintain") else ancestors | {sid}
            if not refs <= allowed:
                fail("condition_dependency", "Predicate inputs must be prior dependency steps; a lookup cannot depend on its own answer", step_id=sid, field=field)
            if field == "when" and not refs:
                fail("condition_evidence", "A condition must read observed step evidence", step_id=sid, field=field)
        if step["kind"] == "maintain" and step["when"] is None and step["postcondition"] is None:
            fail("maintain_evidence", "Unconditional maintain needs an observed postcondition", step_id=sid)
    if previous:
        for sid, old in previous.steps.items():
            if previous.statuses.get(sid) in ("done", "uncertain"):
                if sid not in steps or steps[sid] != old:
                    fail("execution_immutable", "Executed/uncertain steps cannot be removed or rewritten", step_id=sid)
        for op, receipt in previous.successful_operations.items():
            matching = [s for s in steps.values() if s["operation_id"] == op]
            if len(matching) != 1 or matching[0]["id"] != receipt.step_id:
                fail("operation_immutable", "Successful operations must retain their original step identity")
    # Also proves the entire contract is JSON-safe before any state mutation.
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        fail("json_values", str(error), category="format")
    return goals, steps
