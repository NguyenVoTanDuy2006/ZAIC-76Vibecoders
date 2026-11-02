import copy
import tempfile
import unittest
from pathlib import Path

import torch


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = torch.nn.ModuleDict(
            {
                "q_proj": torch.nn.Linear(16, 16, bias=False),
                "v_proj": torch.nn.Linear(16, 16, bias=False),
            }
        )
        self.decoder = torch.nn.ModuleDict(
            {
                "q_proj": torch.nn.Linear(16, 16, bias=False),
                "v_proj": torch.nn.Linear(16, 16, bias=False),
            }
        )

    def forward(self, x):
        return self.decoder["v_proj"](self.audio_encoder["q_proj"](x))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        try:
            import peft  # noqa: F401 - dependency availability probe
        except ImportError:
            self.skipTest("requires dedicated Transformers 5 / PEFT environment")
        from asr.finetuning.config import load_config

        self.c = load_config()
        self.settings = self.c["pissa"]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pissa_export_preserves_live_state_and_loads_on_original_base(self):
        from peft import PeftModel
        from asr.finetuning.pissa import PiSSAManager

        torch.manual_seed(7)
        original = TinyBase()
        fresh = copy.deepcopy(original)
        manager = PiSSAManager(self.settings)
        model = manager.attach(original)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.add_(0.01)
        model.train()
        x = torch.randn(2, 16)
        expected = model(x).detach()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        manager.save_initialization(self.root / "pissa_initial")
        manager.export(self.root / "adapter")
        self.assertTrue(model.training)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        self.assertEqual(model.peft_config["default"].r, 8)
        loaded = PeftModel.from_pretrained(fresh, self.root / "adapter")
        self.assertTrue(torch.allclose(loaded(x), expected, atol=1e-5, rtol=1e-5))

    def test_checkpoint_restores_residual_optimizer_and_next_update(self):
        from asr.finetuning.pissa import PiSSAManager
        from asr.finetuning.checkpoints import CheckpointManager

        torch.manual_seed(9)
        base = TinyBase()
        fresh = copy.deepcopy(base)
        method = PiSSAManager(self.settings)
        model = method.attach(base)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        x = torch.randn(2, 16)
        model(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        checkpoints = CheckpointManager(self.root, "config-hash", "data-hash")
        checkpoints.save_latest(
            epoch=1,
            method=method,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            history=[],
            best=None,
            rng_by_rank=[checkpoints.capture_rng()],
            progress={"updates": 1},
        )
        payload = checkpoints.read(self.root / "latest.pt")
        resumed = PiSSAManager(self.settings)
        other = resumed.attach(fresh, payload["method_state"])
        opt2 = torch.optim.AdamW(
            [p for p in other.parameters() if p.requires_grad], lr=1e-3
        )
        checkpoints.restore(payload, resumed, opt2, None, None, rank=0, world_size=1)
        self.assertTrue(torch.allclose(model(x), other(x), atol=0, rtol=0))
        for m, o in [(model, optimizer), (other, opt2)]:
            m(x).square().mean().backward()
            o.step()
            o.zero_grad()
        self.assertTrue(torch.allclose(model(x), other(x), atol=1e-7, rtol=1e-6))
        bad = CheckpointManager(self.root, "other-config", "data-hash")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            bad.read(self.root / "latest.pt")

    def test_failed_checkpoint_write_preserves_previous_commit(self):
        from unittest.mock import patch
        from asr.finetuning.pissa import PiSSAManager
        from asr.finetuning.checkpoints import CheckpointManager

        method = PiSSAManager(self.settings)
        model = method.attach(TinyBase())
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad]
        )
        checkpoints = CheckpointManager(self.root, "c", "d")
        kwargs = dict(
            epoch=1,
            method=method,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            history=[],
            best=None,
            rng_by_rank=[checkpoints.capture_rng()],
            progress={},
        )
        checkpoints.save_latest(**kwargs)
        before = (self.root / "latest.pt").read_bytes()
        with patch("torch.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                checkpoints.save_latest(**kwargs)
        self.assertEqual((self.root / "latest.pt").read_bytes(), before)
        self.assertEqual(list(self.root.glob(".latest-*.pt")), [])


if __name__ == "__main__":
    unittest.main()
