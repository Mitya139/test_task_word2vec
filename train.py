import numpy as np
from tqdm import tqdm

from config import Config
from preprocess import preprocess_corpus, generate_skipgram_pairs_for_review
from sampler import NegativeSampler
from model import SkipGramNegativeSamplingModel
from utils import ensure_output_dir, save_training_artifacts


def count_pairs_in_review(review_ids: list[int], window_size: int) -> int:
    total = 0
    n = len(review_ids)

    for center_pos in range(n):
        left = max(0, center_pos - window_size)
        right = min(n, center_pos + window_size + 1)
        total += (right - left - 1)

    return total


def count_total_pairs(encoded_reviews: list[list[int]], window_size: int) -> int:
    total = 0

    for review_ids in encoded_reviews:
        total += count_pairs_in_review(review_ids, window_size)

    return total


def train(config: Config):
    output_dir = ensure_output_dir("outputs")

    print("Preprocessing corpus...")
    data = preprocess_corpus(config)

    print(f"Number of reviews: {len(data.raw_reviews)}")
    print(f"Vocabulary size: {len(data.id_to_word)}")

    total_pairs = count_total_pairs(data.encoded_reviews, config.window_size)
    print(f"Total skip-gram pairs per epoch: {total_pairs}")

    sampler = NegativeSampler(
        counts=data.counts,
        power=config.negative_sampling_power,
        seed=config.seed,
    )

    model = SkipGramNegativeSamplingModel(
        vocab_size=len(data.id_to_word),
        embedding_dim=config.embedding_dim,
        seed=config.seed,
    )

    rng = np.random.default_rng(config.seed)

    logged_steps = []
    logged_losses = []

    global_step = 0

    total_steps = total_pairs * config.epochs
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        review_order = np.arange(len(data.encoded_reviews))
        rng.shuffle(review_order)

        progress_bar = tqdm(total=total_pairs, desc=f"Epoch {epoch + 1}", unit="pair")

        running_loss = 0.0
        running_steps = 0
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for review_index in review_order:
            review_ids = data.encoded_reviews[review_index]

            pairs = generate_skipgram_pairs_for_review(
                review_ids=review_ids,
                window_size=config.window_size,
            )

            for center_id, positive_id in pairs:
                progress = global_step / total_steps
                current_lr = max(
                    config.min_learning_rate,
                    config.learning_rate * (1.0 - progress)
                )
                negative_ids = sampler.sample(
                    num_negative=config.num_negative,
                    positive_id=positive_id,
                )

                loss = model.train_one_pair(
                    center_id=center_id,
                    positive_id=positive_id,
                    negative_ids=negative_ids,
                    learning_rate=current_lr,
                )

                global_step += 1
                running_steps += 1
                epoch_steps += 1

                running_loss += loss
                epoch_loss_sum += loss

                if running_steps % config.print_every == 0:
                    avg_loss = running_loss / running_steps
                    logged_steps.append(global_step)
                    logged_losses.append(avg_loss)

                    progress_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

                    running_loss = 0.0
                    running_steps = 0

            progress_bar.update(len(pairs))

        if running_steps > 0:
            avg_loss = running_loss / running_steps
            logged_steps.append(global_step)
            logged_losses.append(avg_loss)
            progress_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        epoch_avg_loss = epoch_loss_sum / epoch_steps
        progress_bar.close()

        print(f"Epoch {epoch + 1} average loss: {epoch_avg_loss:.4f}")

    print("\nSaving results...")
    save_training_artifacts(
        output_dir=output_dir,
        id_to_word=data.id_to_word,
        counts=data.counts,
        model=model,
        steps=logged_steps,
        losses=logged_losses,
    )

    print("Saved files:")
    print(output_dir / "vocab.txt")
    print(output_dir / "input_embeddings.npy")
    print(output_dir / "output_embeddings.npy")
    print(output_dir / "loss_history.csv")
    print(output_dir / "loss_curve.png")


if __name__ == "__main__":
    config = Config()
    train(config)
