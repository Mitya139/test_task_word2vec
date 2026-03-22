from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from model import SkipGramNegativeSamplingModel


def ensure_output_dir(output_dir: Path | str = "outputs") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_vocab(output_dir: Path, id_to_word: list[str], counts: list[int]) -> None:
    vocab_path = output_dir / "vocab.txt"

    with open(vocab_path, "w", encoding="utf-8") as f:
        for word, count in zip(id_to_word, counts):
            f.write(f"{word}\t{count}\n")


def save_embeddings(output_dir: Path, model: SkipGramNegativeSamplingModel) -> None:
    np.save(output_dir / "input_embeddings.npy", model.input_embeddings)
    np.save(output_dir / "output_embeddings.npy", model.output_embeddings)


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
