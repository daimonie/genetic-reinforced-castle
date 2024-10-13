import numpy as np
import copy
from castle.game import Config


class Chromosome:
    def __init__(self, config: Config):
        self.config = config
        self.num_castles = config.num_castles
        self.genes = [
            self.config.random_generator.random()
            for _ in range(self.config.num_castles)
        ]
        self.normalize()

    def normalize(self):
        """Normalize the genes to ensure they sum to 1 and are valid probabilities."""
        self.genes = np.clip(self.genes, 0, None)  # Ensure all values are non-negative
        total = np.sum(self.genes)
        if total > 0:
            self.genes = self.genes / total
        else:
            # If all genes are zero, set them to equal probabilities
            self.genes = np.full_like(self.genes, 1.0 / len(self.genes))

    def point_mutation(self, mutation_rate: float):
        """Perform point mutations on the chromosome."""
        for i in range(self.num_castles):
            if self.config.random_generator.random() < mutation_rate:
                self.genes[i] += self.config.random_generator.normal(
                    0, self.config.mutation_std_dev
                )
        self.normalize()

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
        # Ensure genes are valid probabilities
        self.normalize()

        # Check for any remaining issues
        if (
            np.any(np.isnan(self.genes))
            or np.any(self.genes < 0)
            or np.any(self.genes > 1)
        ):
            raise ValueError("Invalid gene values detected after normalization")

        return self.config.random_generator.multinomial(total_armies, self.genes)

    def __str__(self):
        return f"Chromosome(genes={self.genes})"

    def crossover(self, other: "Chromosome") -> "Chromosome":
        """
        Perform crossover with another Chromosome.

        Args:
            other (Chromosome): The other chromosome to perform crossover with.

        Returns:
            Chromosome: A new chromosome resulting from the crossover.
        """
        # Ensure both chromosomes have the same length
        assert len(self.genes) == len(other.genes)

        # Choose a random crossover point
        crossover_point = self.config.random_generator.integers(1, len(self.genes))

        # Create a new chromosome
        new_chromosome = copy.deepcopy(self)

        # Perform crossover
        new_chromosome.genes = np.concatenate(
            [self.genes[:crossover_point], other.genes[crossover_point:]]
        )

        # Normalize the new chromosome
        new_chromosome.normalize()

        return new_chromosome

    def mutate(self, mutation_rate=0.1, mutation_amount=0.1):
        """Perform both point mutations and region swaps."""
        mask = np.random.random(self.genes.shape) < mutation_rate
        mutation = np.random.normal(0, mutation_amount, self.genes.shape)
        self.genes[mask] += mutation[mask]
        self.genes = np.clip(self.genes, 0, 1)
        self.genes /= self.genes.sum()

        # Ensure at least one gene is mutated
        if not np.any(mask):
            index = np.random.randint(0, len(self.genes))
            self.genes[index] += np.random.normal(0, mutation_amount)
            self.genes = np.clip(self.genes, 0, 1)
            self.genes /= self.genes.sum()
