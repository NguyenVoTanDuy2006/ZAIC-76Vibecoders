"""Process-level leases for run mutations on Linux/Kaggle."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class RunLock:
    def __init__(self, root):
        root = Path(root).resolve()
        self.path = root.parent / f".{root.name}.lock"
        self.fd = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(f"run is already in use: {self.path}") from None
        self.fd = fd
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        return self

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        # Keep the inode: unlinking could allow another process to lock a new inode.

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
