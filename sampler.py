import numpy as np

from config import Config
from preprocess import preprocess_corpus


class NegativeSampler:
    def __init__(self, counts: list[int], power: float = 0.75, seed: int = 42, table_size: int = 10 ** 7):
        self.counts = np.array(counts, dtype=np.float64)
        self.power = power
        self.rng = np.random.default_rng(seed)
        self.vocab_size = len(counts)

        adjusted_counts = self.counts ** self.power
        self.probabilities = adjusted_counts / adjusted_counts.sum()

        self.table_size = table_size

        cdf = np.cumsum(self.probabilities)
        cdf[-1] = 1.0

        positions = np.linspace(0.0, 1.0, num=self.table_size, endpoint=False)
        self.table = np.searchsorted(cdf, positions).astype(np.int32)

    def sample(
        self,
        num_negative: int,
        positive_id: int | None = None,
    ) -> np.ndarray:
        indices = self.rng.integers(0, self.table_size, size=num_negative)
        negative_ids = self.table[indices]

        if positive_id is not None:
            mask = negative_ids == positive_id
            while np.any(mask):
                resampled = self.table[self.rng.integers(0, self.table_size, size=mask.sum())]
                negative_ids[mask] = resampled
                mask = negative_ids == positive_id

        return negative_ids


if __name__ == "__main__":
    config = Config()
    data = preprocess_corpus(config)

    sampler = NegativeSampler(
        counts=data.counts,
        power=config.negative_sampling_power,
        seed=config.seed,
    )

    print("vocab size:", sampler.vocab_size)

    print("First 10 sampling probabilities:")
    for i in range(10):
        print(i, data.id_to_word[i], sampler.probabilities[i])

    print("Example negative samples:")
    samples = sampler.sample(num_negative=config.num_negative, positive_id=0)
    print(samples)

    print("Example sampled words:")
    print([data.id_to_word[i] for i in samples])
