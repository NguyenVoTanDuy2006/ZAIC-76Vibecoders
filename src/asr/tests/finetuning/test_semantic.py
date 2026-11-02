import types
import unittest
import torch


class Tokenizer:
    model_max_length = 6

    def encode(self, text, add_special_tokens=True, **kw):
        ids = [{"a": 2, "b": 3, "x": 4}[w] for w in text.split()]
        return [0, *ids, 1] if add_special_tokens else ids

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def prepare_for_model(self, ids, **kw):
        return {
            "input_ids": [0, *ids, 1],
            "attention_mask": [1] * (len(ids) + 2),
            "special_tokens_mask": [1, *([0] * len(ids)), 1],
        }


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(5), freeze=False)
        self.config = types.SimpleNamespace(max_position_embeddings=8)

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeScorer:
    def __init__(self):
        self._model = Encoder()
        self._tokenizer = Tokenizer()

    def score(self, hyps, refs, **kw):
        f = torch.tensor([1.0 if a == b else 0.5 for a, b in zip(hyps, refs)])
        return f, f, f


class SemanticTests(unittest.TestCase):
    def make(self):
        from asr.finetuning.semantic import PhoBERTReward

        return PhoBERTReward(
            {"batch_size": 2, "reference_cache_size": 2},
            "cpu",
            scorer=FakeScorer(),
            segmenter=lambda s: s,
        )

    def test_empty_predictions_order_and_freezing(self):
        scorer = self.make()
        scores = scorer.score_pairs([("a", "a"), ("a", ""), ("", ""), ("a", "b")])
        self.assertEqual([s.f1 for s in scores], [1.0, 0.0, 1.0, 0.5])
        self.assertFalse(any(p.requires_grad for p in scorer.model.parameters()))
        self.assertFalse(scorer.model.training)

    def test_long_prediction_penalizes_tail_instead_of_truncating(self):
        scorer = self.make()
        short, long = scorer.score_pairs([("a b", "a b"), ("a b", "a b x x x x x")])
        self.assertFalse(short.chunked)
        self.assertTrue(long.chunked)
        self.assertAlmostEqual(long.f1, 4 / 9, places=6)
        self.assertIsNone(scorer.model.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
