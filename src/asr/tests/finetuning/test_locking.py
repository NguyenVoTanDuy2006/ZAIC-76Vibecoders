import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LockTests(unittest.TestCase):
    def test_overlapping_process_and_bundle_are_rejected_and_os_releases_lock(self):
        from asr.finetuning.locking import RunLock
        from asr.finetuning.bundles import create_bundles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from asr.finetuning.locking import RunLock; import sys; "
                    'lock=RunLock(sys.argv[1]); lock.acquire(); print("locked",flush=True); sys.stdin.read()',
                    str(root),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with self.assertRaisesRegex(RuntimeError, "in use"):
                    RunLock(root).acquire()
                with self.assertRaisesRegex(RuntimeError, "in use"):
                    create_bundles(root, Path(tmp) / "bundles")
            finally:
                child.communicate(input="", timeout=10)
            with RunLock(root):
                pass
