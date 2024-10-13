from typing import Dict
from castle.game import Config
from .player import FitnessPlayer
from .chromosome import Chromosome
import copy


class GeneticPlayer(FitnessPlayer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.chromosome = Chromosome(config)  # Pass the entire config object
        self.rewards = []  # Add a list to store rewards

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
        if reward > 100:
            adjusted_reward = reward - self.config.reinforced_win_reward
        elif reward < 0:
            adjusted_reward = reward + self.config.reinforced_lose_penalty
        else:
            adjusted_reward = reward
        self.rewards.append(adjusted_reward)  # Store the adjusted reward

    def mutate(self, mutation_rate=0.1, mutation_amount=0.1):
        self.chromosome.mutate(mutation_rate, mutation_amount)

    def __str__(self):
        return f"GeneticPlayer(chromosome={self.chromosome})"

    def copy(self):
        new_player = copy.deepcopy(self)
        new_player.rewards = []  # Reset rewards for the new copy
        return new_player

    def get_average_reward(self):
        """
        Calculate and return the average reward.

        Returns:
            float: The average reward, or 0 if no rewards have been received.
        """
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0

    def get_recent_performance(self) -> float:
        """
        Calculate and return the recent performance of the player.

        Returns:
            float: The average of the last 10 rewards, or the average reward if less than 10 games played.
        """
        return (
            sum(self.rewards[-10:]) / 10
            if len(self.rewards) >= 10
            else self.get_average_reward()
        )

    def fitness(self) -> float:
        """
        Calculate and return the fitness of the player.

        Returns:
            float: The calculated fitness score.
        """
        recent_performance = self.get_recent_performance()
        win_ratio = (
            sum(1 for r in self.rewards if r > 0) / len(self.rewards)
            if self.rewards
            else 0
        )

        fitness = (recent_performance * 0.7) + (win_ratio * 0.3)

        return fitness

    def crossover(self, other: "GeneticPlayer"):
        """
        Perform crossover with another GeneticPlayer.

        Args:
            other (GeneticPlayer): The other player to perform crossover with.

        This method modifies the current player's chromosome by combining it
        with the chromosome of the other player.
        """
        # Use the chromosome's crossover method
        self.chromosome = self.chromosome.crossover(other.chromosome)

        # Reset rewards after crossover
        self.rewards = []
