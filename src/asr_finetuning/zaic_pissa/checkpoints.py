from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, root, config_hash, data_hash):
        self.root = Path(root)
        self.config_hash = config_hash
        self.data_hash = data_hash

    @staticmethod
    def capture_rng():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    @staticmethod
    def restore_rng(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def save_latest(
        self,
        *,
        epoch,
        method,
        optimizer,
        scheduler,
        scaler,
        history,
        best,
        rng_by_rank,
        progress,
    ):
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "epoch": epoch,
            "config_fingerprint": self.config_hash,
            "data_fingerprint": self.data_hash,
            "method_state": method.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "history": history,
            "best": best,
            "rng_by_rank": rng_by_rank,
            "progress": progress,
        }
        fd, tmp = tempfile.mkstemp(prefix=".latest-", suffix=".pt", dir=self.root)
        os.close(fd)
        try:
            with open(tmp, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.root / "latest.pt")
        finally:
            Path(tmp).unlink(missing_ok=True)

    def read(self, path):
        # Native training checkpoints are trusted project artifacts (pickle).
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (
            state.get("schema_version") != 1
            or state.get("config_fingerprint") != self.config_hash
            or state.get("data_fingerprint") != self.data_hash
        ):
            raise ValueError(
                "training checkpoint schema/config/data fingerprint mismatch"
            )
        return state

    def restore(self, state, method, optimizer, scheduler, scaler, *, rank, world_size):
        if len(state["rng_by_rank"]) != world_size:
            raise ValueError("resume requires the same world size")
        method.restore_adapter(state["method_state"])
        optimizer.load_state_dict(state["optimizer"])
        if (state["scheduler"] is None) != (scheduler is None) or (
            state["scaler"] is None
        ) != (scaler is None):
            raise ValueError("resume scheduler/scaler configuration differs")
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(state["scaler"])
        self.restore_rng(state["rng_by_rank"][rank])
