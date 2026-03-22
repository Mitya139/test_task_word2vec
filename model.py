import numpy as np

EPSILON = 1e-10


class SkipGramNegativeSamplingModel:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        rng = np.random.default_rng(seed)

        self.input_embeddings = rng.uniform(
            low=-0.5 / embedding_dim,
            high=0.5 / embedding_dim,
            size=(vocab_size, embedding_dim),
        ).astype(np.float32)

        self.output_embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    def sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def train_one_pair(self, center_id: int, positive_id: int, negative_ids: np.ndarray, learning_rate: float) -> float:
        # 1. Forward
        center_vector = self.input_embeddings[center_id]
        positive_vector = self.output_embeddings[positive_id]
        negative_vectors = self.output_embeddings[negative_ids]

        pos_score = np.dot(center_vector, positive_vector)
        neg_scores = np.dot(negative_vectors, center_vector)

        pos_prob = self.sigmoid(pos_score)
        neg_probs = self.sigmoid(neg_scores)

        # 2. Loss
        loss = -np.log(pos_prob + EPSILON) - np.sum(np.log(1.0 - neg_probs + EPSILON))

        # 3. Gradients
        grad_pos_score = pos_prob - 1.0
        grad_neg_scores = neg_probs

        grad_center = grad_pos_score * positive_vector + np.dot(grad_neg_scores, negative_vectors)
        grad_positive_output = grad_pos_score * center_vector
        grad_negative_outputs = np.outer(grad_neg_scores, center_vector)

        # 4. Update
        self.input_embeddings[center_id] -= learning_rate * grad_center
        self.output_embeddings[positive_id] -= learning_rate * grad_positive_output
        np.add.at(self.output_embeddings, negative_ids, -learning_rate * grad_negative_outputs)

        return float(loss)

    def get_embedding(self, word_id: int) -> np.ndarray:
        return self.input_embeddings[word_id]

    def get_all_embeddings(self) -> np.ndarray:
        return self.input_embeddings
