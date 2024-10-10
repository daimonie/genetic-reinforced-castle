from abc import ABC, abstractmethod
from typing import Dict
from castle.game import Config


class Player(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def distribute_armies(self) -> Dict[int, int]:
        """
        Distribute armies among castles.

        Returns:
            Dict[int, int]: A dictionary where keys are castle numbers
            and values are the number of armies placed in each castle.
        """
        pass

    @abstractmethod
    def update(self, reward: float, training_progress: float):
        """
        Update the player's strategy based on the reward received.

        Args:
            reward (float): The reward received for the last action.
            training_progress (float): The current progress of training, typically between 0 and 1.
        """
        pass

    def sanitize_distribute_armies(self) -> Dict[int, int]:
        """
        Sanitize the army distribution to ensure it adheres to the total number of armies
        available to the player.

        Returns:
            Dict[int, int]: A sanitized distribution that matches the total number of armies.
        """
        distribution = self.distribute_armies()
        total_armies = self.config.armies_per_player
        sanitized = {castle: int(armies) for castle, armies in distribution.items()}

        # Calculate the difference between distributed and available armies
        distributed = sum(sanitized.values())
        difference = total_armies - distributed

        # Distribute the remainder evenly
        castles = list(sanitized.keys())
        for i in range(abs(difference)):
            castle = castles[i % len(castles)]
            sanitized[castle] += 1 if difference > 0 else -1

        self.last_distribution = sanitized
        return sanitized


class RandomPlayer(Player):
    def distribute_armies(self) -> Dict[int, int]:
        """
        Distribute armies randomly among castles using the config's random generator.

        Returns:
            Dict[int, int]: A dictionary where keys are castle numbers
            and values are the number of armies placed in each castle.
        """
        distribution = {}
        remaining_armies = self.config.armies_per_player
        castles = list(self.config.points_per_castle.keys())
        # Generate a random distribution for all castles simultaneously
        distribution_array = self.config.random_generator.multinomial(
            self.config.armies_per_player, [1 / len(castles)] * len(castles)
        )

        # Create the distribution dictionary
        distribution = {
            castle: int(armies) for castle, armies in zip(castles, distribution_array)
        }

        # No need to shuffle as the distribution is already random

        return distribution

    def update(self, reward: float, training_progress: float):
        """
        A dummy update method that does nothing.

        Args:
            reward (float): The reward received for the last action.
            training_progress (float): The current training progress.
        """
        pass
