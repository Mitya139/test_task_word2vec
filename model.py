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

    def forward(
            self,
            center_id: int,
            positive_id: int,
            negative_ids: list[int],
    ) -> dict:
        center_vector = self.input_embeddings[center_id]
        positive_vector = self.output_embeddings[positive_id]
        negative_vectors = self.output_embeddings[negative_ids]

        positive_score = np.dot(center_vector, positive_vector)
        negative_scores = np.dot(negative_vectors, center_vector)

        positive_prob = self.sigmoid(positive_score)
        negative_probs = self.sigmoid(negative_scores)

        return {
            "center_id": center_id,
            "positive_id": positive_id,
            "negative_ids": negative_ids,
            "center_vector": center_vector,
            "positive_vector": positive_vector,
            "negative_vectors": negative_vectors,
            "positive_score": positive_score,
            "negative_scores": negative_scores,
            "positive_prob": positive_prob,
            "negative_probs": negative_probs,
        }

    def compute_loss(self, forward_data: dict) -> float:
        positive_prob = forward_data["positive_prob"]
        negative_probs = forward_data["negative_probs"]

        positive_loss = -np.log(positive_prob + EPSILON)
        negative_loss = -np.sum(np.log(1.0 - negative_probs + EPSILON))

        total_loss = positive_loss + negative_loss
        return float(total_loss)

    def backward(self, forward_data: dict) -> dict:
        center_vector = forward_data["center_vector"]
        positive_vector = forward_data["positive_vector"]
        negative_vectors = forward_data["negative_vectors"]

        positive_prob = forward_data["positive_prob"]
        negative_probs = forward_data["negative_probs"]

        grad_positive_score = positive_prob - 1.0
        grad_negative_scores = negative_probs

        grad_center = grad_positive_score * positive_vector
        grad_center += np.sum(grad_negative_scores[:, None] * negative_vectors, axis=0)

        grad_positive_output = grad_positive_score * center_vector
        grad_negative_outputs = grad_negative_scores[:, None] * center_vector[None, :]

        return {
            "grad_center": grad_center,
            "grad_positive_output": grad_positive_output,
            "grad_negative_outputs": grad_negative_outputs,
        }

    def update(
            self,
            center_id: int,
            positive_id: int,
            negative_ids: np.ndarray,
            gradients: dict,
            learning_rate: float,
    ):
        self.input_embeddings[center_id] -= learning_rate * gradients["grad_center"]
        self.output_embeddings[positive_id] -= learning_rate * gradients["grad_positive_output"]

        np.add.at(
            self.output_embeddings,
            negative_ids,
            -learning_rate * gradients["grad_negative_outputs"],
        )

    def train_one_pair(
            self,
            center_id: int,
            positive_id: int,
            negative_ids: list[int],
            learning_rate: float,
    ) -> float:
        forward_data = self.forward(
            center_id=center_id,
            positive_id=positive_id,
            negative_ids=negative_ids,
        )

        loss = self.compute_loss(forward_data)
        gradients = self.backward(forward_data)

        self.update(
            center_id=center_id,
            positive_id=positive_id,
            negative_ids=negative_ids,
            gradients=gradients,
            learning_rate=learning_rate,
        )

        return loss

    def get_embedding(self, word_id: int) -> np.ndarray:
        return self.input_embeddings[word_id]

    def get_all_embeddings(self) -> np.ndarray:
        return self.input_embeddings
