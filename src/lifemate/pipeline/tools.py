"""Allowed-tool selection, model-facing schemas, lookup routing and ID provenance."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import re

from .config import AgentInputError


def select_available_tools(request, catalog):
    installed = request["context"]["installed_apps"]
    names = {name for spec in installed.values() for name in spec["tools"]}
    names.add("sys.ask_missing_info")
    selected = []
    for app in catalog["apps"]:
        for tool in app["tools"]:
            if tool["name"] in names:
                if app["id"] not in installed and app["id"] != "sys":
                    raise AgentInputError("Tool khai báo ở app không được cài")
                selected.append(deepcopy(tool))
    selected_names = [tool["name"] for tool in selected]
    if len(selected_names) != len(set(selected_names)) or set(selected_names) != names:
        raise AgentInputError("Catalog thiếu tool hoặc có tên tool trùng")
    return selected


def reference_lookup_hints(available_tools, schema):
    """Use catalog descriptions to name installed read tools, never invent one.

    Hints guide the model; the controller does not pick a search query, call a
    tool automatically, or select an entity on the user's behalf.
    """
    read_names = {tool["name"] for tool in available_tools if tool.get("kind") == "read"}
    hints = []

    def walk(spec, path):
        if "x-ref" in spec:
            mentioned = re.findall(r"\b[a-z_]+\.[a-z_]+\b", spec.get("description", ""))
            lookups = list(dict.fromkeys(name for name in mentioned if name in read_names))
            hints.append({"parameter": path, "reference": spec["x-ref"], "lookup_tools": lookups})
        for key, child in spec.get("properties", {}).items():
            walk(child, f"{path}.{key}")
        if "items" in spec:
            walk(spec["items"], f"{path}[]")

    walk(schema, "arguments")
    return hints


def _description_without_examples(description):
    """Keep format guidance while omitting literal example values from prompts."""
    return re.split(r"\b(?:ví dụ|for example)\b|\be\.g\.", description,
                    maxsplit=1, flags=re.I)[0].rstrip(" ,;:")


def prompt_schema(schema):
    """Copy a schema for the model without examples; runtime schema stays intact."""
    if not isinstance(schema, dict):
        return deepcopy(schema)
    result = deepcopy(schema)
    for key in ("example", "examples"):
        result.pop(key, None)
    if isinstance(result.get("description"), str):
        result["description"] = _description_without_examples(result["description"])
    for key in ("properties", "$defs", "definitions", "patternProperties"):
        if isinstance(result.get(key), dict):
            result[key] = {name: prompt_schema(child) for name, child in result[key].items()}
    for key in ("items", "additionalProperties", "not", "contains"):
        if isinstance(result.get(key), dict):
            result[key] = prompt_schema(result[key])
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if isinstance(result.get(key), list):
            result[key] = [prompt_schema(child) for child in result[key]]
    return result


def qwen_tools(available_tools):
    result = []
    for tool in available_tools:
        description = _description_without_examples(tool.get("description", ""))
        if tool.get("kind"):
            description += f"\nLoại thao tác: {tool['kind']}."
        if tool.get("returns"):
            description += f"\nDữ liệu trả về: {tool['returns']}"
        hints = reference_lookup_hints(available_tools, tool["parameters"])
        if hints:
            description += "\nNếu chưa biết ID x-ref, tra cứu trước và chờ kết quả: " + json.dumps(hints, ensure_ascii=False)
        result.append({"type": "function", "function": {
            "name": tool["name"], "description": description,
            "parameters": prompt_schema(tool["parameters"]),
        }})
    return result


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def unverified_references(arguments, schema, known_strings, utterances):
    """Provenance heuristic, not proof that an ID belongs to the right entity.

    Block an x-ref value unless it was observed in context/history/tool results,
    or stated literally by the user. Does not access hidden app state.
    """
    unknown = []

    def walk(value, spec, path):
        if "x-ref" in spec:
            values = value if isinstance(value, list) else [value]
            for candidate in values:
                if not isinstance(candidate, str):
                    continue
                literal = any(re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", text)
                              for text in utterances)
                if candidate not in known_strings and not literal:
                    unknown.append(path)
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, spec.get("properties", {}).get(key, {}), f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, spec.get("items", {}), f"{path}[{index}]")

    walk(arguments, schema, "arguments")
    return unknown
