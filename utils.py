from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from model import SkipGramNegativeSamplingModel

EPSILON = 1e-10


def ensure_output_dir(output_dir: Path | str = "outputs") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_vocab(output_dir: Path, id_to_word: list[str], counts: list[int]) -> None:
    vocab_path = output_dir / "vocab.txt"

    with open(vocab_path, "w", encoding="utf-8") as f:
        for word, count in zip(id_to_word, counts):
            f.write(f"{word}\t{count}\n")


def load_vocab(output_dir: Path | str) -> tuple[list[str], dict[str, int], list[int]]:
    output_dir = Path(output_dir)
    vocab_path = output_dir / "vocab.txt"

    id_to_word = []
    counts = []

    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            word, count = line.rstrip("\n").split("\t")
            id_to_word.append(word)
            counts.append(int(count))

    word_to_id = {word: i for i, word in enumerate(id_to_word)}
    return id_to_word, word_to_id, counts


def save_embeddings(output_dir: Path, model: SkipGramNegativeSamplingModel) -> None:
    np.save(output_dir / "input_embeddings.npy", model.input_embeddings)
    np.save(output_dir / "output_embeddings.npy", model.output_embeddings)


def load_embeddings(output_dir: Path | str) -> tuple[np.ndarray, np.ndarray]:
    output_dir = Path(output_dir)

    input_embeddings = np.load(output_dir / "input_embeddings.npy")
    output_embeddings = np.load(output_dir / "output_embeddings.npy")

    return input_embeddings, output_embeddings


def save_loss_history(output_dir: Path, steps: list[int], losses: list[float]) -> None:
    history_path = output_dir / "loss_history.csv"

    with open(history_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_loss"])

        for step, loss in zip(steps, losses):
            writer.writerow([step, loss])


def plot_loss(output_dir: Path, steps: list[int], losses: list[float]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses)
    plt.xlabel("Training step")
    plt.ylabel("Average loss")
    plt.title("Word2Vec SGNS Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()


def save_training_artifacts(
    output_dir: Path,
    id_to_word: list[str],
    counts: list[int],
    model: SkipGramNegativeSamplingModel,
    steps: list[int],
    losses: list[float],
) -> None:
    save_vocab(output_dir, id_to_word, counts)
    save_embeddings(output_dir, model)
    save_loss_history(output_dir, steps, losses)
    plot_loss(output_dir, steps, losses)


def load_training_artifacts(output_dir: Path | str = "outputs") -> dict:
    output_dir = Path(output_dir)

    id_to_word, word_to_id, counts = load_vocab(output_dir)
    input_embeddings, output_embeddings = load_embeddings(output_dir)

    return {
        "id_to_word": id_to_word,
        "word_to_id": word_to_id,
        "counts": counts,
        "input_embeddings": input_embeddings,
        "output_embeddings": output_embeddings,
    }


def get_embedding_matrix(
    artifacts: dict,
    source: str = "input",
) -> np.ndarray:
    input_embeddings = artifacts["input_embeddings"]
    output_embeddings = artifacts["output_embeddings"]

    if source == "input":
        return input_embeddings
    if source == "output":
        return output_embeddings

    raise ValueError("source must be one of: 'input', 'output'")


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector_norm = np.linalg.norm(vector) + EPSILON
    matrix_norms = np.linalg.norm(matrix, axis=1) + EPSILON

    similarities = matrix @ vector
    similarities = similarities / (matrix_norms * vector_norm)

    return similarities


def nearest_neighbors(
    word: str,
    artifacts: dict,
    top_k: int = 10,
    source: str = "input",
) -> list[tuple[str, float]]:
    word_to_id = artifacts["word_to_id"]
    id_to_word = artifacts["id_to_word"]

    if word not in word_to_id:
        raise ValueError(f"Word '{word}' not found in vocabulary")

    embeddings = get_embedding_matrix(artifacts, source=source)

    word_id = word_to_id[word]
    query_vector = embeddings[word_id]

    similarities = cosine_similarity(query_vector, embeddings)
    similarities[word_id] = -np.inf

    top_ids = np.argsort(-similarities)[:top_k]

    results = []
    for neighbor_id in top_ids:
        results.append((id_to_word[neighbor_id], float(similarities[neighbor_id])))

    return results
