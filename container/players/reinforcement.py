import numpy as np
from typing import Dict
from players.player import Player
from castle.game import Config


class ReinforcedPlayer(Player):
    def __init__(self, config: Config):
        super().__init__(config)
        self.num_castles = self.config.num_castles
        self.num_armies = self.config.armies_per_player
        # Q-matrix: [armies_left][castle] -> Q-value
        self.qmatrix = np.random.uniform(
            0, 0.1, (self.num_armies + 1, self.num_castles)
        )
        self.last_actions = []

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
        learning_rate = self.config.learning_rate * (1 - training_progress)
        discount_factor = 0.9  # You can adjust this

        # Normalize reward
        if reward > 0:
            normalized_reward = reward / self.config.reinforced_win_reward
        else:
            normalized_reward = reward / self.config.reinforced_lose_penalty

        # Reverse the actions to update from last to first
        for i, (armies_left, castle) in enumerate(reversed(self.last_actions)):
            castle_index = castle - 1
            current_q = self.qmatrix[armies_left, castle_index]

            if i == 0:  # Last action
                next_max_q = 0
            else:
                next_armies_left, next_castle = self.last_actions[-i]
                next_max_q = np.max(self.qmatrix[next_armies_left])

            # Q-learning update rule
            new_q = current_q + learning_rate * (
                normalized_reward + discount_factor * next_max_q - current_q
            )
            self.qmatrix[armies_left, castle_index] = max(
                0, new_q
            )  # Ensure non-negative

    def distribute_armies(self) -> Dict[int, int]:
        distribution = {castle: 0 for castle in range(1, self.num_castles + 1)}
        armies_left = self.num_armies
        self.last_actions = []
        # This approach creates a pseudo-state by considering the number of
        # armies left to distribute as part of the state. While there's no
        # traditional state progression in this single-decision game, this
        # method allows the agent to learn different strategies based on the
        # remaining resources. It's effective because:
        # 1. It captures the diminishing returns of placing armies.
        # 2. It allows for more nuanced decision-making as the distribution progresses.
        # 3. It can learn to prioritize certain castles early or late in the distribution.

        while armies_left > 0:
            if self.config.random_generator.random() < self.config.epsilon:
                # Exploration: choose a random castle
                castle = self.config.random_generator.choice(self.num_castles) + 1
            else:
                # Exploitation: choose the castle with the highest Q-value
                castle = np.argmax(self.qmatrix[armies_left]) + 1

            distribution[castle] += 1
            self.last_actions.append((armies_left, castle))
            armies_left -= 1

        return distribution
