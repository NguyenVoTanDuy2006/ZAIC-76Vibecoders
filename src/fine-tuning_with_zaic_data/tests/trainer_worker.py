import sys

from zaic_pissa.config import load_config
from zaic_pissa.runtime import DistributedRuntime
from zaic_pissa.semantic import SemanticScore
from zaic_pissa.trainer import Trainer


class DistributedReward:
    def __init__(self, rank):
        self.rank = rank

    def score_pairs(self, pairs, normalized=True):
        # Rank 0 has zero advantage. Rank 1 must still average the PG gradient with it.
        return [
            SemanticScore(0.4 if self.rank == 1 and len(pairs) == 2 and i == 1 else 0.5)
            for i, _ in enumerate(pairs)
        ]


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    runtime = DistributedRuntime("cpu").initialize(config)
    trainer = Trainer(
        config, runtime, reward_factory=lambda c, rt: DistributedReward(rt.rank)
    )
    try:
        trainer.train()
    finally:
        trainer.release()
        runtime.close()
