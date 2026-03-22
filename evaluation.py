import argparse

from utils import load_training_artifacts, nearest_neighbors


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Word2Vec embeddings")
    parser.add_argument(
        "--word",
        type=str,
        required=True,
        help="Query word for nearest neighbors",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of nearest neighbors to show",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory with saved vocab and embeddings",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="input",
        choices=["input", "output"],
        help="Which embeddings to use",
    )

    args = parser.parse_args()

    artifacts = load_training_artifacts(args.output_dir)

    print(f"Vocabulary size: {len(artifacts['id_to_word'])}")
    print(f"Query word: {args.word}")
    print(f"Embedding source: {args.source}")
    print()

    try:
        neighbors = nearest_neighbors(
            word=args.word,
            artifacts=artifacts,
            top_k=args.top_k,
            source=args.source,
        )
    except ValueError as e:
        print(e)
        return

    print(f"Top {args.top_k} nearest neighbors for '{args.word}':")
    for rank, (neighbor_word, score) in enumerate(neighbors, start=1):
        print(f"{rank:2d}. {neighbor_word:<20} cosine={score:.4f}")


if __name__ == "__main__":
    main()
