"""Module-boundary and source-snapshot integration checks; no model inference."""
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

    def test_notebook_installs_one_importable_agent_and_evaluation_snapshot(self):
        helper = test_agent_calls.KaggleSourcePatchTests()
        with fixture() as (root, _, __, ___):
            namespace = helper.install(root)
            source_root = namespace["PROJECT_ROOT"] / "src"
            setup_path = root / "setup.py"
            setup_path.write_text(helper.bootstrap_source(), encoding="utf-8")
            self.run_isolated(source_root, '''
                from pathlib import Path
                from lifemate import agent
                from lifemate.pipeline import backend
                from lifemate.evaluation import runner, experiment
                expected_root = Path(sys.path[0]).resolve()
                assert Path(agent.__file__).resolve().is_relative_to(expected_root)
                assert Path(runner.__file__).resolve().is_relative_to(expected_root)
                assert runner.AgentSession is agent.AgentSession
                assert agent.AgentConfig is backend.AgentConfig
                assert agent.ModelLoadConfig(revision="checkpoint-sha").revision == "checkpoint-sha"
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
                weights = object()
                live_backend = backend.HFBackend(weights, None)
                original_class = type(live_backend)
                # Re-running the same immutable source does not reload modules or weights.
                workspace = expected_root.parent.parent
                exec(compile((workspace / "setup.py").read_text(encoding="utf-8"), "setup", "exec"),
                     {"SOURCE_WORK_ROOT": workspace, "qwen_backend": live_backend})
                assert live_backend.model is weights
                assert type(live_backend) is original_class
                assert original_class.generate.__globals__["ContextLimitError"] is agent.ContextLimitError
                assert agent.ModelReply is backend.ModelReply
                assert "torch" not in sys.modules and "transformers" not in sys.modules
            ''')

    def test_previously_imported_other_source_is_rejected_before_snapshot_changes(self):
        helper = test_agent_calls.KaggleSourcePatchTests()
        with fixture() as (root, _, __, ___):
            setup_path = root / "setup.py"
            setup_path.write_text(helper.bootstrap_source(), encoding="utf-8")
            self.run_isolated(REPO / "src", f'''
                from pathlib import Path
                import lifemate
                workspace = Path({str(root)!r})
                try:
                    exec(compile((workspace / "setup.py").read_text(encoding="utf-8"), "setup", "exec"),
                         {{"SOURCE_WORK_ROOT": workspace / "source"}})
                except RuntimeError as error:
                    assert "Restart Session" in str(error)
                else:
                    raise AssertionError("Mixed imported sources must be rejected")
                assert not (workspace / "source").exists()
            ''')
