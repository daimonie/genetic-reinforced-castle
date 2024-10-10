import numpy as np
from typing import Dict
from players.player import Player
from castle.game import Config
import gymnasium as gym
from gymnasium import spaces
from rl_zoo3 import DQN


class ReinforcedPlayer(Player):
    def __init__(self, config: Config):
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


class GymPlayer(Player):
    def __init__(self, config: Config):
        super().__init__(config)
        self.action_space = spaces.MultiDiscrete(
            [self.config.armies_per_player + 1] * self.config.num_castles
        )
        self.observation_space = spaces.Dict(
            {
                "castles": spaces.Box(
                    low=0,
                    high=self.config.armies_per_player,
                    shape=(self.config.num_castles,),
                    dtype=np.int32,
                ),
                "total_armies": spaces.Discrete(self.config.armies_per_player + 1),
            }
        )
        self.model = DQN("MultiInputPolicy", self, verbose=1)
        self.last_action = None
        self.last_state = None

    def reset(self, seed=None, options=None):
        super().reset()
        self.last_action = None
        self.last_state = None
        return self._get_observation(), {}

    def _get_observation(self):
        return {
            "castles": np.array(
                [
                    self.config.points_per_castle[castle]
                    for castle in range(1, self.config.num_castles + 1)
                ],
                dtype=np.int32,
            ),
            "total_armies": self.config.armies_per_player,
        }

    def step(self, action):
        self.last_action = action
        self.last_state = self._get_observation()
        # Placeholder values - update these based on your game logic
        return self._get_observation(), 0, False, False, {}

    def distribute_armies(self) -> Dict[int, int]:
        state = self._get_observation()
        action, _ = self.model.predict(state, deterministic=True)
        distribution = {i + 1: int(a) for i, a in enumerate(action)}
        self.last_action = distribution
        self.last_state = state
        return distribution

    def update(self, reward: float, training_progress: float):
        if self.last_action is None or self.last_state is None:
            return

        next_state = self._get_observation()
        done = False  # Update this based on your game logic
        self.model.learn(total_timesteps=1, reset_num_timesteps=False)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = DQN.load(path, env=self)

    # Implement these methods to make the class compatible with gym.Env
    def render(self):
        pass

    def close(self):
        pass
