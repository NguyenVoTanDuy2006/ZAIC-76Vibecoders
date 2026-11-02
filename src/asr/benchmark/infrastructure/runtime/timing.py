from __future__ import annotations

import time


def synchronize_device(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def timed_call(function, *, device: str) -> tuple[object, float]:
    synchronize_device(device)
    start = time.perf_counter()
    result = function()
    synchronize_device(device)
    return result, time.perf_counter() - start
