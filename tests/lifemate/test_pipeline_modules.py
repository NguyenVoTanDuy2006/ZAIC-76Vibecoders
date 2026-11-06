"""Module-boundary and pinned-notebook integration checks; no model inference."""
from contextlib import redirect_stdout
import io
import hashlib
import json
import subprocess
import sys
from textwrap import dedent
import unittest

import test_agent_calls
from test_predictions import REPO, fixture


class PipelineModuleTests(unittest.TestCase):
    def run_isolated(self, source_root, code):
        result = subprocess.run(
            [sys.executable, "-B", "-I", "-c",
             "import sys; sys.path.insert(0, " + repr(str(source_root)) + ");\n" + dedent(code)],
            cwd=REPO, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stages_import_without_coordinator_or_heavy_model_packages(self):
        # Blocking these imports proves the new modules do not reach back through
        # agent.py or require GPU libraries just to inspect or validate a stage.
        self.run_isolated(REPO / "src", '''
            import importlib
            import importlib.abc
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "lifemate.agent" or fullname.split(".")[0] in {"torch", "transformers"}:
                        raise AssertionError("Unexpected dependency: " + fullname)
            sys.meta_path.insert(0, Block())
            for name in ("config", "inputs", "tools", "messages", "replies", "backend",
                         "controller", "actor", "execution", "results", "runner",
                         "predicates", "planning", "proposals", "state", "recovery", "adaptive", "grounding", "call_repair", "protocol"):
                importlib.import_module("lifemate.pipeline." + name)
        ''')

    def test_pinned_notebook_installs_importable_modules_and_reloads_a_live_backend(self):
        _, script, payload = test_agent_calls.KaggleSourcePatchTests().patch_source()
        with fixture() as (root, _, __, ___):
            package = root / "src/lifemate"
            package.mkdir(parents=True)
            (package / "agent.py").write_text(payload["replacements"][0]["before"], encoding="utf-8")
            for name in ("__init__.py", "paths.py"):
                (package / name).write_text((REPO / "src/lifemate" / name).read_text(encoding="utf-8"), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                setup_namespace = {"PROJECT_ROOT": root}
                exec(compile(script, "notebook-setup", "exec"), setup_namespace)
            expected_hashes = {f["path"]: f["sha256"] for f in payload["files"]}
            self.assertEqual(setup_namespace["AGENT_PIPELINE_SHA256"], expected_hashes)
            expected_bundle = hashlib.sha256(json.dumps({"agent": payload["patched_sha256"],
                "helpers": expected_hashes}, sort_keys=True).encode()).hexdigest()
            self.assertEqual(setup_namespace["AGENT_BUNDLE_SHA256"], expected_bundle)
            setup_path = root / "setup.py"
            setup_path.write_text(script, encoding="utf-8")
            self.run_isolated(root / "src", '''
                from pathlib import Path
                from lifemate import agent
                from lifemate.pipeline import backend
                expected_root = Path(sys.path[0]).resolve()
                assert Path(agent.__file__).resolve().is_relative_to(expected_root)
                assert agent.AgentConfig is backend.AgentConfig
                class Env:
                    def get_tool_calls(self): return []
                    def get_state_hash(self): return "canonical"
                class Backend:
                    def generate(self, *args, **kwargs): return agent.ModelReply("Done")
                request = {"id": "sample", "history": [], "context": {"installed_apps": {}}}
                speech = {"id": "sample", "current_text": "Hello", "history_user_texts": []}
                tools = [{"name": "sys.ask_missing_info", "parameters": {"type": "object"}}]
                result = agent.run_agent(request, speech, tools, Env(), backend=Backend(),
                                         config=agent.AgentConfig(workflow_enabled=False))
                assert result["status"] == "finished"
                assert result["submission"] == {"id": "sample", "tool_calls": [], "state_hash": "canonical"}
                unified = agent.run_agent(request, speech, tools, Env(), backend=Backend())
                assert unified["diagnostics"]["workflow_protocol"] == "btc-tool-call-v1"
                assert unified["diagnostics"]["controller"] is None
                assert unified["diagnostics"]["call_protocol"]["execution_count"] == 0
                # Old backend objects must raise the current exception class after
                # dependency reload, so context-limit failures keep their status.
                weights = object()
                live_backend = backend.HFBackend(weights, None)
                original_class = type(live_backend)
                root = expected_root.parent
                exec(compile((root / "setup.py").read_text(encoding="utf-8"), "setup", "exec"),
                     {"PROJECT_ROOT": root})
                assert live_backend.model is weights
                assert original_class.generate.__globals__["ContextLimitError"] is agent.ContextLimitError
                assert agent.ModelReply is backend.ModelReply
            ''')

    def test_unknown_extracted_module_is_preserved_before_any_upgrade_write(self):
        _, script, payload = test_agent_calls.KaggleSourcePatchTests().patch_source()
        with fixture() as (root, _, __, ___):
            source = root / "src/lifemate/agent.py"
            source.parent.mkdir(parents=True)
            baseline = payload["replacements"][0]["before"]
            source.write_text(baseline, encoding="utf-8")
            custom = source.parent / "pipeline/actor.py"
            custom.parent.mkdir()
            custom.write_text("# local actor changes\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                exec(compile(script, "notebook-setup", "exec"), {"PROJECT_ROOT": root})
            self.assertEqual(source.read_text(encoding="utf-8"), baseline)
            self.assertEqual(custom.read_text(encoding="utf-8"), "# local actor changes\n")
            self.assertFalse((source.parent / "workflow.py").exists())
