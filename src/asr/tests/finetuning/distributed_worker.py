"""Executed by torchrun from test_distributed; real Gloo collectives/backward."""

import json
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from asr.finetuning.config import load_config
from asr.finetuning.optimization import OptimizerController
from asr.finetuning.runtime import DistributedRuntime


def main():
    c = load_config()
    c["train"].update(world_size=2, precision="fp32")
    c["model"]["dtype"] = "float32"
    rt = DistributedRuntime("cpu").initialize(c)
    try:
        model = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(1.0)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        control = OptimizerController(opt, scheduler, None, 100.0, rt)
        # CE on both ranks, SCST has zero advantage on rank 0. Partial group of 2/8.
        for micro in range(2):
            with model.no_sync():
                ce = model(torch.ones(1, 1)).sum()
                control.backward(ce, 2)
            context = (
                model.no_sync()
                if micro == 0
                else __import__("contextlib").nullcontext()
            )
            with context:
                policy = model(torch.ones(1, 1)).sum() * (rt.rank * 2.0)
                control.backward(policy, 2)
        control.step(model)
        assert torch.allclose(model.module.weight, torch.tensor([[0.8]])), (
            model.module.weight
        )
        # One rank fails a step; all ranks must skip optimizer and scheduler.
        model.module.weight.grad = torch.tensor([[float("inf") if rt.rank else 1.0]])
        before = scheduler.last_epoch
        step = control.step(model)
        assert step["skipped"] and scheduler.last_epoch == before
        # Uneven strided evaluation has every ID once, no DistributedSampler padding.
        ids = [f"s{i}" for i in range(5)]
        gathered = [i for shard in rt.gather(ids[rt.rank :: 2]) for i in shard]
        assert len(gathered) == len(set(gathered)) == 5
        try:
            rt.call(lambda: 1 / 0 if rt.rank else 1, "semantic scorer")
        except RuntimeError as exc:
            assert "semantic scorer" in str(exc)
        else:
            raise AssertionError("scorer failure was hidden")
        if rt.main:
            Path(os.environ["ZAIC_TEST_RESULT"]).write_text(
                json.dumps({"status": "ok"})
            )
    finally:
        rt.close()


if __name__ == "__main__":
    main()
