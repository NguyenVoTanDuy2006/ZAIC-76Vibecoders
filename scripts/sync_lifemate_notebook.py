"""Refresh reviewed embedded Kaggle sources without fetching or running models.

Run from repo root: python scripts/sync_lifemate_notebook.py
Historical reviewed agent snapshots remain accepted; unknown user edits do not.
"""
import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def assigned(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def sync():
    path = ROOT / "notebooks/lifemate/test_lifemate.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    bootstrap = "".join(nb["cells"][4]["source"])
    assignment = assigned(ast.parse(bootstrap), "AGENT_CALL_FIX_SOURCE")
    script = ast.literal_eval(assignment.value)
    patch_assignment = assigned(ast.parse(script), "_AGENT_CALL_PATCH")
    payload = json.loads(patch_assignment.value.args[0].value)
    # Every stored replacement is reversible to a reviewed source version.
    previous = payload["replacements"][0]["after"]
    versions = []
    for entry in [payload] + payload.get("upgrades", []):
        source = previous
        for replacement in reversed(entry["replacements"]):
            if source.count(replacement["after"]) != 1:
                raise ValueError("Cannot reconstruct reviewed source")
            source = source.replace(replacement["after"], replacement["before"], 1)
        if digest(source) != entry["base_sha256"]:
            raise ValueError("Reviewed source checksum mismatch")
        versions.append(source)
    current = (ROOT / "src/lifemate/agent.py").read_text(encoding="utf-8")
    if digest(previous) != digest(current):
        versions.append(previous)
    versions = list(dict.fromkeys(versions))
    def entry(source):
        return {"base_sha256": digest(source), "replacements": [{"before": source, "after": current}]}
    updated = {**entry(versions[0]), "patched_sha256": digest(current),
               "upgrades": [entry(s) for s in versions[1:]], "files": []}
    helpers = {file["path"]: file for file in payload["files"]}
    # A pinned monolithic checkout also needs every extracted pipeline module.
    # Keep workflow first: the bootstrap exposes its hash for diagnostics.
    for file_path in sorted((ROOT / "src/lifemate/pipeline").glob("*.py")):
        helpers.setdefault(file_path.relative_to(ROOT).as_posix(), {"path": file_path.relative_to(ROOT).as_posix()})
    for file in helpers.values():
        source = (ROOT / file["path"]).read_text(encoding="utf-8")
        accepted = set(file.get("accepted_sha256", []))
        if "sha256" in file:
            accepted.add(file["sha256"])
        accepted.discard(digest(source))
        updated["files"].append({"path": file["path"], "source": source,
                                 "sha256": digest(source), "accepted_sha256": sorted(accepted)})
    lines = script.splitlines(keepends=True)
    script = ("".join(lines[:patch_assignment.lineno - 1])
              + '_AGENT_CALL_PATCH = json.loads(' + repr(json.dumps(updated, ensure_ascii=False)) + ')\n'
              + "".join(lines[patch_assignment.end_lineno:]))
    old_check = '''if _target.exists() and _target.read_text(encoding="utf-8") != _entry["source"]:
        raise RuntimeError("Helper source differs; refusing to overwrite your changes.")'''
    new_check = '''if _target.exists():
        _existing_hash = hashlib.sha256(_target.read_text(encoding="utf-8").encode()).hexdigest()
        if _existing_hash not in {_entry["sha256"], *_entry.get("accepted_sha256", [])}:
            raise RuntimeError("Helper source differs; refusing to overwrite your changes.")'''
    script = script.replace(old_check, new_check)
    marker = 'AGENT_WORKFLOW_SHA256 = _AGENT_CALL_PATCH["files"][0]["sha256"]'
    if "AGENT_PIPELINE_SHA256 =" not in script:
        script = script.replace(marker, marker + '\nAGENT_PIPELINE_SHA256 = {f["path"]: f["sha256"] for f in _AGENT_CALL_PATCH["files"]}\n'
            'AGENT_BUNDLE_SHA256 = hashlib.sha256(json.dumps({"agent": AGENT_CALL_FIX_SHA256, "helpers": AGENT_PIPELINE_SHA256}, sort_keys=True).encode()).hexdigest()')
    # Reload dependencies before the coordinator, including on a warm Kaggle kernel.
    # Existing backend instances keep their module globals and reuse the loaded weights.
    reload_modules = ["lifemate.workflow"] + ["lifemate.pipeline." + name for name in (
        "config", "inputs", "tools", "predicates", "planning", "proposals", "state", "recovery",
        "adaptive", "quantities", "grounding", "call_repair", "protocol", "messages", "replies", "backend", "controller", "checkpoint", "task_reasoning",
        "actor", "execution", "results", "runner",
    )] + ["lifemate.pipeline", "lifemate.agent"]
    script = re.sub(
        r'for _module_name in \([^\n]*\):',
        "for _module_name in " + repr(tuple(reload_modules)) + ":", script,
    )
    lines = bootstrap.splitlines(keepends=True)
    bootstrap = ("".join(lines[:assignment.lineno - 1]) + "AGENT_CALL_FIX_SOURCE = " + repr(script) + "\n"
                 + "".join(lines[assignment.end_lineno:]))
    nb["cells"][4]["source"] = bootstrap.splitlines(keepends=True)
    support = "".join(nb["cells"][6]["source"])
    assignment = assigned(ast.parse(support), "TRAIN_EVAL_SOURCE")
    lines = support.splitlines(keepends=True)
    support = ("".join(lines[:assignment.lineno - 1]) + "TRAIN_EVAL_SOURCE = "
               + repr((ROOT / "src/lifemate/train_eval.py").read_text(encoding="utf-8")) + "\n"
               + "".join(lines[assignment.end_lineno:]))
    nb["cells"][6]["source"] = support.splitlines(keepends=True)
    # Visible runtime marker distinguishes a current warm kernel from old traces.
    run_source = "".join(nb["cells"][12]["source"])
    marker = 'print("Grounding trước khi chạy:", agent_config.grounding_enabled)'
    if 'print("Call repair:"' not in run_source:
        run_source = run_source.replace(marker, marker + '\nfrom lifemate.pipeline import call_repair, checkpoint, quantities\n'
            'print("Call repair:", call_repair.VERSION)\n'
            'print("Money grounding:", quantities.VERSION)\n'
            'print("Terminal checkpoint:", checkpoint.VERSION, agent_config.terminal_check_enabled, "limit:", agent_config.max_terminal_checks)\n'
            'print("Source bundle:", globals().get("AGENT_BUNDLE_SHA256"))')
    if 'print("Task reasoning:"' not in run_source:
        run_source = run_source.replace('print("Source bundle:", globals().get("AGENT_BUNDLE_SHA256"))',
            'from lifemate.pipeline import task_reasoning\n'
            'print("Task reasoning:", task_reasoning.VERSION, agent_config.task_reasoning_enabled)\n'
            'print("Source bundle:", globals().get("AGENT_BUNDLE_SHA256"))')
    nb["cells"][12]["source"] = run_source.splitlines(keepends=True)
    nb["cells"][18]["source"] = "".join(nb["cells"][18]["source"]).replace(
        'two inference runs', 'four inference variants').replace(
        'three inference variants', 'four inference variants').splitlines(keepends=True)
    reader = "".join(nb["cells"][20]["source"])
    reader = reader.replace('"execution_reconciled"}',
        '"execution_reconciled", "task_brief", "write_check", "write_check_limit", "terminal_check", "terminal_check_limit"}')
    if 'print("TASK REASONING:"' not in reader:
        reader = reader.replace('        effective_input = diagnostics.get("first_model_input")',
            '        if call_protocol and call_protocol.get("task_reasoning"):\n'
            '            print("TASK REASONING:", json.dumps(call_protocol["task_reasoning"], ensure_ascii=False, indent=2))\n'
            '        effective_input = diagnostics.get("first_model_input")')
        reader = reader.replace('            print("FIRST MODEL INPUT SHA256:", effective_input.get("sha256"))',
            '            print("FIRST MODEL INPUT PHASE:", effective_input.get("phase", "not saved"))\n'
            '            print("FIRST MODEL INPUT SHA256:", effective_input.get("sha256"))')
    nb["cells"][20]["source"] = reader.splitlines(keepends=True)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "kaggle-cell", "exec")
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Refreshed agent/pipeline/workflow/evaluation source in", path.relative_to(ROOT))


if __name__ == "__main__":
    sync()
