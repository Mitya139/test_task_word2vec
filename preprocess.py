from pathlib import Path
import shutil
import kagglehub
import re
import pandas as pd
from collections import Counter
from dataclasses import dataclass

from config import Config

HTML_BREAK_RE = re.compile(r"<br\s*/?>")
HTML_TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass
class PreprocessData:
    raw_reviews: list[str]
    tokenized_reviews: list[list[str]]
    encoded_reviews: list[list[int]]
    word_to_id: dict[str, int]
    id_to_word: list[str]
    counts: list[int]


def ensure_dataset(config: Config) -> Path:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dataset_path = data_dir / config.dataset_filename
    if dataset_path.exists():
        return dataset_path

    downloaded_dir = Path(kagglehub.dataset_download(config.dataset_slug))

    for file_path in downloaded_dir.rglob(config.dataset_filename):
        shutil.copy2(file_path, dataset_path)
        return dataset_path

    raise FileNotFoundError(
        f"Could not find {config.dataset_filename} inside downloaded dataset"
    )


def load_reviews(config: Config) -> list[str]:
    dataset_path = ensure_dataset(config)

    df = pd.read_csv(dataset_path)
    reviews = df["review"].tolist()

    return reviews


def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    text = HTML_BREAK_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)

    tokens = TOKEN_RE.findall(text)
    return tokens


def tokenize_reviews(reviews: list[str]) -> list[list[str]]:
    tokenized_reviews = []

    for review in reviews:
        tokens = tokenize_text(review)
        tokenized_reviews.append(tokens)

    return tokenized_reviews


def build_vocab(
    tokenized_reviews: list[list[str]],
    min_count: int = 5,
    max_vocab_size: int | None = None,
) -> tuple[dict[str, int], list[str], list[int]]:
    counter = Counter()

    for tokens in tokenized_reviews:
        counter.update(tokens)

    vocab_items = []
    for word, count in counter.items():
        if count >= min_count:
            vocab_items.append((word, count))

    vocab_items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        vocab_items = vocab_items[:max_vocab_size]

    id_to_word = []
    counts = []

    for word, count in vocab_items:
        id_to_word.append(word)
        counts.append(count)

    word_to_id = {}
    for i, word in enumerate(id_to_word):
        word_to_id[word] = i

    return word_to_id, id_to_word, counts


def encode_reviews(
    tokenized_reviews: list[list[str]],
    word_to_id: dict[str, int],
) -> list[list[int]]:
    encoded_reviews = []

    for tokens in tokenized_reviews:
        ids = []

        for token in tokens:
            if token in word_to_id:
                ids.append(word_to_id[token])

        if len(ids) >= 2:
            encoded_reviews.append(ids)

    return encoded_reviews


def generate_skipgram_pairs_for_review(
    review_ids: list[int],
    window_size: int,
) -> list[tuple[int, int]]:
    pairs = []
    n = len(review_ids)

    for center_pos in range(n):
        center_id = review_ids[center_pos]

        left = max(0, center_pos - window_size)
        right = min(n, center_pos + window_size + 1)

        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue

            context_id = review_ids[context_pos]
            pairs.append((center_id, context_id))

    return pairs


def generate_skipgram_pairs(
    encoded_reviews: list[list[int]],
    window_size: int,
):
    for review_ids in encoded_reviews:
        for pair in generate_skipgram_pairs_for_review(review_ids, window_size):
            yield pair


def preprocess_corpus(config: Config) -> PreprocessData:
    raw_reviews = load_reviews(config)
    tokenized_reviews = tokenize_reviews(raw_reviews)

    min_count = getattr(config, "min_count", 5)
    max_vocab_size = getattr(config, "max_vocab_size", None)

    word_to_id, id_to_word, counts = build_vocab(
        tokenized_reviews=tokenized_reviews,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
    )

    encoded_reviews = encode_reviews(
        tokenized_reviews=tokenized_reviews,
        word_to_id=word_to_id,
    )

    return PreprocessData(
        raw_reviews=raw_reviews,
        tokenized_reviews=tokenized_reviews,
        encoded_reviews=encoded_reviews,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        counts=counts,
    )


if __name__ == "__main__":
    config = Config()
    data = preprocess_corpus(config)

    print("reviews:", len(data.raw_reviews))
    print("vocab size:", len(data.id_to_word))

    print("first tokenized review:")
    print(data.tokenized_reviews[0][:30])

    print("first encoded review:")
    print(data.encoded_reviews[0][:30])

    print("top 10 words:")
    for i in range(min(10, len(data.id_to_word))):
        print(i, data.id_to_word[i], data.counts[i])

    print("first 10 skip-gram pairs from first review:")
    pairs = generate_skipgram_pairs_for_review(
        data.encoded_reviews[0],
        config.window_size,
    )
    print(pairs[:10])
