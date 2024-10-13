from typing import Dict
from castle.game import Config
from .player import Player
from .chromosome import Chromosome


class GeneticPlayer(Player):
    def __init__(self, config: Config):
        super().__init__(config)
        self.chromosome = Chromosome(config)

    def distribute_armies(self) -> Dict[int, int]:
        """
        Distribute armies among castles based on the chromosome.

        Returns:
            Dict[int, int]: A dictionary where keys are castle numbers
            and values are the number of armies placed in each castle.
        """
        distribution_array = self.chromosome.get_distribution(
            self.config.armies_per_player
        )
        return {castle: int(armies) for castle, armies in enumerate(distribution_array)}

    def update(self, reward: float, training_progress: float):
        """
        Update the player's strategy based on the reward received.

        Args:
            reward (float): The reward received for the last action.
            training_progress (float): The current progress of training, typically between 0 and 1.
        """
        pass

    def mutate(self):
        self.chromosome.mutate()

    def __str__(self):
        return f"GeneticPlayer(chromosome={self.chromosome})"
