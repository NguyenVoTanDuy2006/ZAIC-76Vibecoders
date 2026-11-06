"""Checkpoint identity handoff without importing Torch or loading model weights."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lifemate.pipeline.backend import load_qwen_backend
from lifemate.pipeline.config import AgentConfig, ModelLoadConfig
from lifemate.pipeline.messages import SYSTEM_PROMPT


REPO = Path(__file__).resolve().parents[2]


class ModelLoadingTests(unittest.TestCase):
    def test_revision_is_optional_and_preserves_existing_positional_arguments(self):
        config = ModelLoadConfig("local-checkpoint", "cuda:1", True)
        self.assertEqual(config.device, "cuda:1")
        self.assertTrue(config.local_files_only)
        self.assertIsNone(config.revision)
        for invalid in ("", "  ", 42, True):
            with self.subTest(revision=invalid), self.assertRaises(ValueError):
                ModelLoadConfig(revision=invalid)

    def test_tokenizer_and_model_receive_the_same_revision_and_offline_policy(self):
        for revision in (None, "checkpoint-commit-sha"):
            with self.subTest(revision=revision):
                model = Mock()
                model.generation_config = SimpleNamespace(prefill_chunk_size=None, use_cache=False)
                model.parameters.return_value = [SimpleNamespace(numel=lambda: 4_000_000_000)]
                tokenizer = object()
                tokenizer_loader = Mock(return_value=tokenizer)
                model_loader = Mock(return_value=model)
                torch = SimpleNamespace(float16="fake-float16", cuda=SimpleNamespace(
                    is_available=lambda: True, device_count=lambda: 2))
                transformers = SimpleNamespace(
                    AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
                    AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader))
                config = ModelLoadConfig("local-checkpoint", "cuda:1", True, revision)
                options = AgentConfig(prefill_chunk_size=128)
                with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
                    backend = load_qwen_backend(config, options)
                tokenizer_loader.assert_called_once_with(
                    "local-checkpoint", local_files_only=True, revision=revision)
                model_loader.assert_called_once_with(
                    "local-checkpoint", local_files_only=True, revision=revision,
                    dtype="fake-float16", device_map={"": 1}, attn_implementation="sdpa")
                model.eval.assert_called_once_with()
                self.assertIs(backend.model, model)
                self.assertIs(backend.tokenizer, tokenizer)
                self.assertEqual(model.generation_config.prefill_chunk_size, 128)
                self.assertTrue(model.generation_config.use_cache)

    def test_evaluation_config_matches_the_current_prompt_and_agent_budget(self):
        config = json.loads((REPO / "configs/lifemate/agent-evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(config["prompt_sha256"], hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest())
        self.assertEqual(config["generation"]["max_steps"], AgentConfig().max_steps)
        self.assertEqual(config["generation"]["max_calls"], AgentConfig().max_calls)


if __name__ == "__main__":
    unittest.main()
