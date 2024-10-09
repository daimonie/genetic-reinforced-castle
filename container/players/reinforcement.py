import numpy as np
from typing import Dict
from players.player import Player


class ReinforcedPlayer(Player):
    def __init__(self, config):
        super().__init__(config)
        self.qmatrix = np.ones(
            (self.config.num_castles, self.config.armies_per_player + 1)
        )

    def set_qmatrix(self, qmatrix):
        self.qmatrix = qmatrix

    def get_qmatrix(self):
        """
        Returns the current Q-matrix.

        Returns:
            numpy.ndarray: The current Q-matrix.
        """
        return self.qmatrix

    def update(self, reward: float, training_progress: float):
        """
        Update the Q-matrix based on the last action and received reward.

        Args:
            reward (float): The reward received for the last action.
            learning_rate (float): The learning rate for updating the Q-values.
        """
        learning_rate = self.config.learning_rate * (
            self.config.epsilon_decay
            ** (training_progress * self.config.reinforced_training_games)
        )

        if not hasattr(self, "last_distribution"):
            raise ValueError(
                "No cached distribution found. Make sure to call distribute_armies before update."
            )

        for castle, armies in self.last_distribution.items():
            castle_index = castle - 1  # Adjust for 0-based indexing
            current_q = self.qmatrix[castle_index, armies]

            # Calculate the update
            update = learning_rate * (reward - current_q)

            # Update the Q-value
            self.qmatrix[castle_index, armies] += update

        # Normalize Q-values to prevent unbounded growth
        self.qmatrix = (self.qmatrix - self.qmatrix.min()) / (
            self.qmatrix.max() - self.qmatrix.min() + 1e-8
        )

    def distribute_armies(self) -> Dict[int, int]:
        if self.qmatrix is None:
            raise ValueError(
                "Q-matrix has not been set. Call set_qmatrix before distributing armies."
            )

        distribution = {}
        total_armies = self.config.armies_per_player
        castles = list(self.config.points_per_castle.keys())

        # Epsilon-greedy strategy
        if self.config.random_generator.random() < self.config.epsilon:
            # Exploration: distribute armies randomly
            # Generate three random distributions
            distributions = [
                self.config.random_generator.multinomial(
                    total_armies, [1 / len(castles)] * len(castles)
                )
                for _ in range(3)
            ]

            # Calculate the weighted sum for each distribution
            # Higher numbered castles have more weight
            weighted_sums = [
                sum(armies * (i + 1) for i, armies in enumerate(dist))
                for dist in distributions
            ]

            # Choose the distribution with the highest weighted sum
            army_distribution = distributions[weighted_sums.index(max(weighted_sums))]
        else:
            # Exploitation: use Q-values
            q_values = self.qmatrix[: len(castles)]

            # Apply softmax to get probabilities for all castles simultaneously
            exp_q_values = np.exp(q_values)
            probabilities = exp_q_values / np.sum(exp_q_values, axis=1, keepdims=True)

            # Sample from the multinomial distribution
            army_distribution = self.config.random_generator.multinomial(
                total_armies, np.sum(probabilities, axis=1) / np.sum(probabilities)
            )

        # Assign armies to castles
        for castle, armies in zip(castles, army_distribution):
            distribution[castle] = int(armies)

        return distribution
