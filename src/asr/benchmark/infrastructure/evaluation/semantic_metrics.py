from __future__ import annotations


class PhoBERTBERTScore:
    """Contextual token similarity, with the word segmentation PhoBERT expects."""

    def __init__(self, model_path: str, config: dict):
        from bert_score import BERTScorer

        self.config = config
        self.scorer = BERTScorer(
            model_type=model_path,
            num_layers=config["num_layers"],
            batch_size=config["batch_size"],
            device=config["device"],
            idf=False,
            rescale_with_baseline=False,
        )
        self.model_parameter_count = sum(p.numel() for p in self.scorer._model.parameters())

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        if self.config.get("word_segment", True):
            from pyvi import ViTokenizer

            def segment(text: str) -> str:
                return ViTokenizer.tokenize(text) if text.strip() else ""

            pairs = [(segment(ref), segment(hyp)) for ref, hyp in pairs]
        scores = [0.0] * len(pairs)
        positions = [i for i, (ref, hyp) in enumerate(pairs) if ref.strip() and hyp.strip()]
        for i, (ref, hyp) in enumerate(pairs):
            if not ref.strip() and not hyp.strip():
                scores[i] = 1.0
        if positions:
            references = [pairs[i][0] for i in positions]
            hypotheses = [pairs[i][1] for i in positions]
            _, _, f1 = self.scorer.score(hypotheses, references, batch_size=self.config["batch_size"])
            for position, value in zip(positions, f1.tolist()):
                scores[position] = float(value)
        return scores
