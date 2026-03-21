import numpy as np

from config import Config
from preprocess import preprocess_corpus


class NegativeSampler:
    def __init__(self, counts: list[int], power: float = 0.75, seed: int = 42):
        self.counts = np.array(counts, dtype=np.float64)
        self.power = power
        self.rng = np.random.default_rng(seed)

        adjusted_counts = self.counts ** self.power
        self.probabilities = adjusted_counts / adjusted_counts.sum()

        self.vocab_size = len(counts)

    def sample(self, num_negative: int, positive_id: int | None = None) -> list[int]:
        negative_ids = []

        while len(negative_ids) < num_negative:
            sampled_id = int(self.rng.choice(self.vocab_size, p=self.probabilities))

            if positive_id is not None and sampled_id == positive_id:
                continue

            negative_ids.append(sampled_id)

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
