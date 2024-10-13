import numpy as np
from castle.game import Config


class Chromosome:
    def __init__(self, config: Config):
        self.config = config
        self.num_castles = config.num_castles
        self.genes = self.config.random_generator.random(self.num_castles)
        self.normalize_genes()

    def normalize_genes(self):
        """Normalize the genes to ensure they sum to 1."""
        self.genes = self.genes / np.sum(self.genes)

    def point_mutation(self, mutation_rate: float):
        """Perform point mutations on the chromosome."""
        for i in range(self.num_castles):
            if self.config.random_generator.random() < mutation_rate:
                self.genes[i] += self.config.random_generator.normal(
                    0, self.config.mutation_std_dev
                )
        self.normalize_genes()

    def swap_mutation(self, swap_probability: float):
        """Swap entire regions of the chromosome."""
        if self.config.random_generator.random() < swap_probability:
            idx1, idx2 = self.config.random_generator.choice(
                self.num_castles, size=2, replace=False
            )
            self.genes[idx1], self.genes[idx2] = self.genes[idx2], self.genes[idx1]

    def mutate(self):
        """Perform both point mutations and region swaps."""
        self.point_mutation(self.config.point_mutation_rate)
        self.swap_mutation(self.config.swap_probability)

    def get_distribution(self, total_armies: int) -> np.ndarray:
        """
        Get the distribution of armies based on the chromosome.

        Args:
            total_armies (int): Total number of armies to distribute.

        Returns:
            np.ndarray: Array of integers representing the army distribution.
        """
        return self.config.random_generator.multinomial(total_armies, self.genes)

    def __str__(self):
        return f"Chromosome(genes={self.genes})"
