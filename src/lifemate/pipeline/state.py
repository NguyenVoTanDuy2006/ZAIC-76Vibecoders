"""Controller-owned versioned plans and append-only execution receipts."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import json

from . import predicates, proposals
from .planning import ControllerError


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class Receipt:
    ref: str
    step_id: str
    goal_id: str
    operation_id: str | None
    version: int
    kind: str
    call_json: str
    result_json: str
    outcome: str

    def snapshot(self):
        return {"ref": self.ref, "step_id": self.step_id, "goal_id": self.goal_id,
                "operation_id": self.operation_id, "plan_version": self.version,
                "kind": self.kind, "call": json.loads(self.call_json),
                "result": json.loads(self.result_json), "outcome": self.outcome}


@dataclass
class ControllerState:
    request_json: str
    sources_json: str
    phase: str = "plan"
    version: int = 0
    interpretation_version: int = 0
    goals: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)
    statuses: dict = field(default_factory=dict)
    receipts: dict = field(default_factory=dict)
    successful_operations: dict = field(default_factory=dict)
    initial_operations: set = field(default_factory=set)
    revisions: list = field(default_factory=list)
    failures: Counter = field(default_factory=Counter)
    stalls: Counter = field(default_factory=Counter)
    plan_revisions: int = 0
    interpretation_revisions: int = 0
    active_id: str | None = None
    issue: dict | None = None
    bad_reply: str | None = None
    candidate_final: str | None = None
    pending_review: dict | None = None
    exhausted: list = field(default_factory=list)

    @classmethod
    def from_input(cls, request, speech):
        sources = {"request": speech["current_text"]}
        for turn in speech["history_user_texts"]:
            sources[f"history:{turn['history_index']}"] = turn["text"]
        return cls(encoded({"request": request, "speech": speech}), encoded(sources))

    @property
    def sources(self):
        return json.loads(self.sources_json)

    def results(self):
        return {r.step_id: json.loads(r.result_json) for r in self.receipts.values() if r.outcome == "success"}

    def goal_statuses(self):
        resolved = {"done", "skipped"}
        answer = {}
        for gid, goal in self.goals.items():
            owned = [sid for sid, step in self.steps.items() if step["goal_id"] == gid]
            unsupported = goal["kind"] == "unsupported" and any(
                self.steps[sid]["kind"] == "answer" and self.statuses[sid] != "skipped" for sid in owned)
            answer[gid] = "done" if not unsupported and all(self.statuses[sid] in resolved for sid in owned) else "pending"
        return answer

    def install(self, goals, steps, reason, *, reinterpret=False):
        old_statuses, old_steps = self.statuses, self.steps
        self.goals, self.steps = deepcopy(goals), deepcopy(steps)
        self.statuses = {sid: (old_statuses[sid] if sid in old_steps and old_steps[sid] == step
                              and old_statuses[sid] in ("done", "uncertain") else "pending")
                         for sid, step in steps.items()}
        if not self.version:
            self.initial_operations = {s["operation_id"] for s in steps.values() if s["kind"] == "write"}
            self.initial_operations.update(proposals.operation_id(sid) for sid, s in steps.items()
                if s["kind"] == "deferred" and goals[s["goal_id"]]["kind"] == "write")
        self.version += 1
        self.interpretation_version += int(reinterpret or self.interpretation_version == 0)
        self.revisions.append({"version": self.version, "interpretation_version": self.interpretation_version,
                               "reason": reason, "goals": deepcopy(goals), "steps": deepcopy(steps)})
        self.active_id, self.pending_review, self.candidate_final = None, None, None
        self.phase, self.issue, self.bad_reply = "act", None, None
        self.refresh()

    def refresh(self):
        results = self.results()
        for _ in range(len(self.steps) + 1):
            changed = False
            for sid, step in self.steps.items():
                if self.statuses[sid] in ("done", "uncertain"):
                    continue
                when = predicates.evaluate(step["when"]["predicate"] if step["when"] else None, results)
                dependencies = all(self.statuses[d] == "done" for d in step["depends_on"])
                status = "skipped" if when == "false" else ("deferred" if step["kind"] == "deferred" else "pending")
                if when == "true" and dependencies and step["kind"] == "maintain":
                    if predicates.evaluate(step["postcondition"], results) == "true":
                        status = "done"
                if self.statuses[sid] != status:
                    self.statuses[sid], changed = status, True
            if not changed:
                break

    def ready(self):
        self.refresh()
        results = self.results()
        return [sid for sid, step in self.steps.items() if self.statuses[sid] == "pending"
                and step["kind"] != "deferred"
                and all(self.statuses[d] == "done" for d in step["depends_on"])
                and predicates.evaluate(step["when"]["predicate"] if step["when"] else None, results) == "true"]

    def guard(self, sid, call, version):
        if version != self.version or sid not in self.ready():
            raise ControllerError("not_ready", "Step/version is no longer ready", step_id=sid, category="plan")
        step = self.steps[sid]
        if call["name"] != step["tool"]:
            raise ControllerError("wrong_step_tool", "Call must implement the active step; revise the plan before using another tool", step_id=sid, category="action")
        if any(key not in call["arguments"] or encoded(call["arguments"][key]) != encoded(value)
               for key, value in step["expected_arguments"].items()):
            raise ControllerError("argument_constraint", "Call contradicts expected_arguments", step_id=sid,
                                  field="arguments", category="action")
        if step["kind"] == "write":
            if any(r.kind == "write" and self.statuses.get(r.step_id) == "uncertain" for r in self.receipts.values()):
                raise ControllerError("uncertain_write", "Reconcile outstanding write uncertainty before executing another write", step_id=sid, category="uncertain")
            if step["operation_id"] in self.successful_operations:
                raise ControllerError("duplicate_write", "This logical operation already succeeded", step_id=sid, category="plan")
            for receipt in self.receipts.values():
                if (receipt.kind == "write" and receipt.outcome == "success" and receipt.call_json == encoded(call)
                        and step["operation_id"] not in self.initial_operations):
                    raise ControllerError("duplicate_write", "A new operation ID cannot replay a successful write", step_id=sid, category="plan")

    def record(self, sid, call, result, ref, version, *, uncertain=False):
        if ref in self.receipts or version != self.version:
            raise ControllerError("receipt", "Duplicate receipt or stale execution", step_id=sid, category="uncertain")
        step = self.steps[sid]
        outcome = "uncertain" if uncertain else ("error" if result.get("ok") is not True else
                                                 "no_op" if result.get("done") is False else "success")
        receipt = Receipt(ref, sid, step["goal_id"], step["operation_id"], version, step["kind"],
                          encoded(call), encoded(result), outcome)
        self.receipts[ref] = receipt
        before = self.goal_statuses()
        if outcome == "success":
            if step["kind"] == "write":
                self.successful_operations[step["operation_id"]] = receipt
            verified = predicates.evaluate(step["postcondition"], self.results())
            self.statuses[sid] = "done" if verified == "true" else "uncertain"
        else:
            self.statuses[sid] = "uncertain" if outcome == "uncertain" else "pending"
        self.refresh()
        after = self.goal_statuses()
        for gid, status in after.items():
            if status == "done" and before.get(gid) != "done":
                self.stalls[gid] = 0
        if outcome != "success" or self.statuses[sid] != "done":
            self.stalls[step["goal_id"]] += 1
        return receipt

    def snapshot(self):
        return {"phase": self.phase, "plan_version": self.version, "interpretation_version": self.interpretation_version,
                "goals": list(deepcopy(self.goals).values()), "goal_statuses": self.goal_statuses(),
                "steps": [{**deepcopy(step), "status": self.statuses[sid]} for sid, step in self.steps.items()],
                "ready_steps": self.ready(), "active_step": self.active_id,
                "observations": [r.snapshot() for r in self.receipts.values()],
                "issue": deepcopy(self.issue), "previous_bad_reply": self.bad_reply,
                "plan_revisions": self.plan_revisions, "interpretation_revisions": self.interpretation_revisions,
                "stalls": dict(self.stalls), "failure_counts": dict(self.failures),
                "exhausted": deepcopy(self.exhausted)}
