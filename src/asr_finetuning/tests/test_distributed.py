import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DistributedTests(unittest.TestCase):
    def test_two_rank_zero_advantage_partial_group_failure_and_uneven_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.json"
            env = dict(os.environ, ZAIC_TEST_RESULT=str(result))
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(Path(__file__).with_name("distributed_worker.py")),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(
                process.returncode, 0, process.stdout + "\n" + process.stderr
            )
            self.assertEqual(json.loads(result.read_text())["status"], "ok")


if __name__ == "__main__":
    unittest.main()
