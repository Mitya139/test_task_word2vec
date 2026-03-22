from dataclasses import dataclass


@dataclass
class Config:
    # Dataset info
    dataset_slug: str = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    dataset_filename: str = "IMDB Dataset.csv"

    # Model / optimization
    min_count: int = 5
    embedding_dim: int = 128
    window_size: int = 2
    num_negative: int = 10
    negative_sampling_power: float = 0.75
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    epochs: int = 1

    # Reproducibility / logging
    seed: int = 42
    print_every: int = 1000
