from __future__ import annotations

import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


class DistributedRuntime:
    def __init__(self, device="cuda"):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = (
            torch.device("cuda", self.local_rank)
            if device == "cuda"
            else torch.device("cpu")
        )

    @property
    def main(self):
        return self.rank == 0

    def initialize(self, config):
        if self.world_size != config["train"]["world_size"]:
            raise ValueError(
                "launch with torchrun --nproc_per_node matching train.world_size"
            )
        if self.device.type == "cuda":
            if (
                not torch.cuda.is_available()
                or self.local_rank >= torch.cuda.device_count()
            ):
                raise RuntimeError("configured CUDA GPU is unavailable")
            torch.cuda.set_device(self.local_rank)
        elif config["train"]["precision"] != "fp32":
            raise ValueError("CPU development requires fp32/float32 config")
        torch.set_num_threads(config["runtime"]["cpu_threads"])
        if self.world_size > 1:
            dist.init_process_group(
                "nccl" if self.device.type == "cuda" else "gloo",
                timeout=timedelta(minutes=30),
            )
        seed = config["train"]["seed"] + self.rank
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return self

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def agree(self, flag):
        if self.world_size == 1:
            return bool(flag)
        x = torch.tensor(int(flag), device=self.device)
        dist.all_reduce(x, op=dist.ReduceOp.MIN)
        return bool(x.item())

    def call(self, function, label):
        result = None
        error = None
        try:
            result = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self.gather(error)
        if any(e is not None for e in errors):
            raise RuntimeError(f"{label} failed across ranks: {errors}")
        return result

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
