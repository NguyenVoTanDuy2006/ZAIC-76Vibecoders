import unittest
import torch


class ObjectiveTests(unittest.TestCase):
    def test_advantage_changes_gradient_in_the_right_direction(self):
        from asr.finetuning.objective import scst_loss

        for advantage in [0.2, -0.2, 0.0]:
            logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
            logp = logits.log_softmax(-1)[0, 0]
            loss = scst_loss(
                logp.unsqueeze(0), torch.tensor([advantage]), torch.tensor([0.0])
            )
            loss.backward()
            self.assertAlmostEqual(float(logits.grad[0, 0]), -advantage / 2, places=6)

    def test_logprob_includes_first_eos_and_masks_padding(self):
        from asr.finetuning.objective import sequence_log_probabilities

        # Prompt length 2; generated [1, EOS=2, PAD=0].
        logits = torch.zeros(1, 5, 4, requires_grad=True)
        lp, mask = sequence_log_probabilities(
            logits, 2, torch.tensor([[1, 2, 0]]), [2], [3]
        )
        self.assertEqual(mask.tolist(), [[True, True, False]])
        self.assertAlmostEqual(
            float(lp.detach()), -2 * torch.log(torch.tensor(3.0)).item(), places=6
        )
        lp.sum().backward()
        self.assertEqual(float(logits.grad[0, 3:].abs().sum()), 0)
        self.assertEqual(float(logits.grad[..., 3].abs().sum()), 0)

    def test_forbidden_sample_is_not_accepted(self):
        from asr.finetuning.objective import sequence_log_probabilities

        with self.assertRaisesRegex(ValueError, "forbidden"):
            sequence_log_probabilities(
                torch.zeros(1, 3, 4), 1, torch.tensor([[3, 2]]), [2], [3]
            )

    def test_reward_has_no_gradient_and_weight_is_point_one(self):
        from asr.finetuning.objective import scst_loss, mixed_loss

        logp = torch.tensor([-2.0], requires_grad=True)
        sampled = torch.tensor([0.9], requires_grad=True)
        greedy = torch.tensor([0.7], requires_grad=True)
        pg = scst_loss(logp, sampled, greedy)
        total = mixed_loss(torch.tensor(1.0), pg, 0.1)
        total.backward()
        self.assertAlmostEqual(float(total.detach()), 1.04, places=6)
        self.assertAlmostEqual(float(logp.grad), -0.02, places=6)
        self.assertIsNone(sampled.grad)
        self.assertIsNone(greedy.grad)

    def test_best_selection_prioritizes_semantic_even_when_wer_worsens(self):
        from asr.finetuning.metrics import is_better

        old = {
            "epoch": 1,
            "latest": {
                "normalized_bertscore_f1": 0.90,
                "normalized_wer": 0.10,
                "normalized_cer": 0.08,
            },
        }
        new = {
            "epoch": 2,
            "latest": {
                "normalized_bertscore_f1": 0.91,
                "normalized_wer": 0.20,
                "normalized_cer": 0.18,
            },
        }
        self.assertTrue(is_better(new, old))
        new["latest"]["normalized_bertscore_f1"] = 0.9000001
        self.assertFalse(is_better(new, old))

    def test_metric_normalization_and_corpus_counts(self):
        from asr.finetuning.metrics import normalize, error_rates

        self.assertEqual(normalize(" HẸN lúc 26 GIỜ! "), "hẹn lúc 26 giờ")
        rates = error_rates([("a b", "a c"), ("d", "")])
        self.assertAlmostEqual(rates["wer"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
