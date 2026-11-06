"""Catalog-driven repair feedback. Code writes feedback; the model writes BTC calls.

This module explains rejected proposals. It does not validate/execute application
calls, choose replacement values, or add a new model-output protocol.
"""
from __future__ import annotations

from copy import deepcopy
from difflib import get_close_matches

from .tools import prompt_schema, reference_lookup_hints

VERSION = "catalog-call-repair-v1"


def _missing(value):
    return value is None or value == "" or value == [] or value == {}


def schema_facts(name, tools, arguments=None):
    spec = tools[name]["parameters"]
    properties = spec.get("properties", {})
    arguments = arguments if isinstance(arguments, dict) else {}
    groups = spec.get("x-one-of", [])
    return {"tool_name": name, "valid_params": sorted(properties),
            "required_params": list(spec.get("required", [])),
            "missing_required_params": [k for k in spec.get("required", []) if _missing(arguments.get(k))],
            "argument_alternatives": deepcopy(groups),
            "missing_alternative_group": bool(groups) and not any(
                all(not _missing(arguments.get(k)) for k in group) for group in groups),
            "unexpected_argument_names": sorted(set(arguments) - set(properties)),
            "parameters": prompt_schema(spec),
            "lookup_hints": reference_lookup_hints(list(tools.values()), spec)}


def missing_info_problem(arguments, tools):
    """Check the meaning of sys.ask_missing_info's target, separately from its schema."""
    name, params = arguments.get("tool_name"), arguments.get("params")
    if not isinstance(name, str) or name not in tools:
        return {"code": "missing_info_tool_not_allowed", "field": "arguments.tool_name",
                "message": f"sys.ask_missing_info target {name!r} is not an installed tool; choose its real allowed name."}
    if not isinstance(params, list) or not params:
        return {"code": "missing_info_params_shape", "field": "arguments.params",
                "message": "sys.ask_missing_info params must be a nonempty array of input parameter names."}
    valid = tools[name]["parameters"].get("properties", {})
    invalid = [p for p in params if not isinstance(p, str) or p not in valid]
    if invalid:
        return {"code": "invalid_missing_info_params", "field": "arguments.params",
                "message": f"Tool {name!r} is installed, but params {invalid!r} are not its input parameters. Valid input parameters: {sorted(valid)!r}."}
    return None


def feedback(call, tools, *, message, issue=None, lookup_hints=None, grounding=None):
    """Explain a rejection using only the actual available catalog and validator error."""
    name = call.get("name") if isinstance(call, dict) else None
    args = call.get("arguments", {}) if isinstance(call, dict) else {}
    args = args if isinstance(args, dict) else {}
    code, category = (issue.code, issue.category) if issue is not None else ("invalid_arguments", "schema")
    data = {"version": VERSION, "code": code, "category": category, "failed_tool": name, "executed": False,
            "field": issue.field if issue is not None else None,
            "validator_message": message,
            "expected_output": "One tool call with exactly name and arguments, or a brief plain final when no actionable work remains; never return this feedback JSON.",
            "next_step": "Use grounded values to propose a corrected call. If input can be retrieved, use an allowed lookup. If user input is genuinely missing, call sys.ask_missing_info for the intended tool's actual input parameters; do not substitute a plain question."}
    if not isinstance(name, str):
        data.update(code=code if issue is not None else "call_format", category="format",
                    allowed_tools=sorted(tools))
        return data
    if name not in tools:
        data.update(code="tool_not_allowed", category="schema", allowed_tools=sorted(tools),
                    name_hints=get_close_matches(name, sorted(tools), n=5, cutoff=0.35))
        data["next_step"] = "Choose a tool from allowed_tools that implements the user's intended action. Name hints are catalog spellings, not evidence of semantic equivalence. Do not invent a replacement tool or claim this call executed."
        return data
    facts = schema_facts(name, tools, args)
    data["call_schema"] = facts
    if issue is None:
        if facts["missing_required_params"]:
            data["code"] = "missing_required_arguments"
        elif facts["missing_alternative_group"]:
            data["code"] = "missing_argument_alternative"
    if name == "sys.ask_missing_info":
        problem = missing_info_problem(args, tools)
        if issue is None and problem and not facts["missing_required_params"]:
            data.update(code=problem["code"], field=problem["field"])
        target = args.get("tool_name")
        data["missing_info_target"] = target
        if isinstance(target, str) and target in tools:
            target_facts = schema_facts(target, tools)
            # Values for the target call were not supplied; do not pretend every
            # required field is missing in the actual user's task.
            target_facts.pop("missing_required_params")
            target_facts.pop("missing_alternative_group")
            target_facts.pop("unexpected_argument_names")
            data["target_schema"] = target_facts
            params = args.get("params")
            data["invalid_params"] = ([p for p in params if not isinstance(p, str) or p not in target_facts["valid_params"]]
                                      if isinstance(params, list) else [])
        else:
            data["allowed_tools"] = sorted(tools)
        data["next_step"] = "params names missing INPUT arguments of the intended tool, not fields you hope to obtain from its output. Check target_schema and lookup_hints; choose a lookup or a corrected sys.ask_missing_info call. Never guess values or mechanically rename the parameter."
    if lookup_hints:
        data["lookup_required"] = deepcopy(lookup_hints)
    if grounding:
        data["source_constraints"] = deepcopy(grounding)
    return data
