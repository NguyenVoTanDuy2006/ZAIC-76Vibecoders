import json
import tempfile
import unittest
import wave
from pathlib import Path


class DataConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        records = []
        for i in range(10):
            turns = []
            for j in range(3):
                name = f"audio/c{i}_{j}.wav"
                p = self.root / name
                p.parent.mkdir(exist_ok=True)
                with wave.open(str(p), "wb") as f:
                    f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    f.writeframes((i * 3 + j + 1).to_bytes(2, "little") * 160)
                turns.append(
                    {"role": "user", "audio": name, "audio_text": f"Bật đèn {i} {j}."}
                )
            records.append(
                {
                    "id": f"c{i}",
                    "audio": turns[2]["audio"],
                    "audio_text": turns[2]["audio_text"],
                    "history": [
                        turns[0],
                        {"role": "assistant", "text": "not a target"},
                        turns[1],
                    ],
                }
            )
        self.path = self.root / "train.jsonl"
        self.path.write_text("\n".join(json.dumps(r) for r in records))

    def test_extract_and_split_keep_all_user_audio_and_conversations(self):
        from zaic_pissa.data import prepare_records, split_records

        records = prepare_records(self.root)
        self.assertEqual(len(records), 30)
        self.assertEqual(sum(r["is_latest"] for r in records), 10)
        a = split_records(records, seed=1234)
        self.assertEqual(a, split_records(records, seed=1234))
        self.assertEqual(
            {k: len(v) for k, v in a.items()}, {"train": 24, "validation": 3, "test": 3}
        )
        groups = [{r["conversation_id"] for r in a[k]} for k in a]
        self.assertTrue(
            all(not groups[i] & groups[j] for i in range(3) for j in range(i))
        )

    def test_pcm_duplicate_across_splits_is_rejected(self):
        from zaic_pissa.data import prepare_records, split_records

        records = prepare_records(self.root)
        for r in records:
            r["pcm_sha256"] = "same-audio"
        with self.assertRaisesRegex(ValueError, "PCM"):
            split_records(records)

    def test_audio_cannot_escape_root(self):
        from zaic_pissa.data import prepare_records

        self.path.write_text(
            json.dumps(
                {"id": "bad", "audio": "../outside.wav", "audio_text": "xin chào"}
            )
        )
        with self.assertRaisesRegex(ValueError, "root"):
            prepare_records(self.root)

    def test_fingerprint_is_portable_but_sensitive_to_training_settings(self):
        from zaic_pissa.config import load_config, config_fingerprint

        config = load_config()
        original = config_fingerprint(config)
        config["dataset"]["root"] = "/different/mount"
        config["runtime"]["cache_dir"] = "/different/cache"
        config["output"]["root"] = "/different/output"
        self.assertEqual(original, config_fingerprint(config))
        config["scst"]["weight"] = 1.0
        self.assertNotEqual(original, config_fingerprint(config))

    def test_defaults_match_approved_plan(self):
        from zaic_pissa.config import load_config

        c = load_config()
        self.assertEqual((c["train"]["epochs"], c["scst"]["weight"]), (3, 0.1))
        self.assertEqual(
            c["model"]["revision"], "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
        )
        self.assertEqual(
            c["semantic"]["revision"], "01daacda68afe13d83023d16ec647239e344a1e6"
        )


if __name__ == "__main__":
    unittest.main()
