import numpy as np
from typing import Dict
from players.player import Player
from castle.game import Config


class ReinforcedPlayer(Player):
    def __init__(self, config: Config):
        super().__init__(config)
        # Initialize Q-matrix with small random values
        self.qmatrix = np.random.uniform(
            0, 0.1, (self.config.num_castles, self.config.armies_per_player + 1)
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
        """
        # Adjust learning rate decay
        learning_rate = self.config.learning_rate * (1 - training_progress)

        if not hasattr(self, "last_distribution"):
            raise ValueError(
                "No cached distribution found. Make sure to call distribute_armies before update."
            )

        # Normalize reward to be between -1 and 1
        if reward > 0:
            normalized_reward = reward / self.config.reinforced_win_reward
        else:
            normalized_reward = reward / self.config.reinforced_lose_penalty

        for castle, armies in self.last_distribution.items():
            castle_index = castle - 1  # Adjust for 0-based indexing
            current_q = self.qmatrix[castle_index, armies]

            # Calculate the update
            update = learning_rate * (normalized_reward - current_q)

            # Update the Q-value
            self.qmatrix[castle_index, armies] += update

        # Remove normalization step for entire Q-matrix

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
            # Improved exploration: distribute armies with preference for higher-value castles
            weights = np.array(
                [self.config.points_per_castle[castle] for castle in castles]
            )
            probabilities = weights / np.sum(weights)
            army_distribution = self.config.random_generator.multinomial(
                total_armies, probabilities
            )
        else:
            # Simplified exploitation: choose the best action for each castle
            army_distribution = np.zeros(len(castles), dtype=int)
            for i, castle in enumerate(castles):
                best_action = np.argmax(self.qmatrix[i, : total_armies + 1])
                army_distribution[i] = best_action

            # Adjust distribution if total exceeds available armies
            while np.sum(army_distribution) > total_armies:
                excess = np.sum(army_distribution) - total_armies
                for _ in range(excess):
                    non_zero_indices = np.where(army_distribution > 0)[0]
                    index_to_reduce = self.config.random_generator.choice(
                        non_zero_indices
                    )
                    army_distribution[index_to_reduce] -= 1

        # Assign armies to castles
        for castle, armies in zip(castles, army_distribution):
            distribution[castle] = int(armies)

        self.last_distribution = distribution
        return distribution
