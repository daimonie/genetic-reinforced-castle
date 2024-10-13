import unittest
import numpy as np
from castle.game import Config
from players.chromosome import Chromosome


class TestChromosome(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_castles=5, armies_per_player=100)
        self.chromosome = Chromosome(self.config)

    def test_initialization(self):
        self.assertEqual(len(self.chromosome.genes), self.config.num_castles)
        self.assertAlmostEqual(sum(self.chromosome.genes), 1.0, places=7)

    def test_normalize(self):
        self.chromosome.genes = np.array([1, 2, 3, 4, 5])
        self.chromosome.normalize()
        self.assertAlmostEqual(sum(self.chromosome.genes), 1.0, places=7)

    def test_point_mutation(self):
        original_genes = self.chromosome.genes.copy()
        self.chromosome.point_mutation(mutation_rate=1.0)  # Force mutation
        self.assertFalse(np.array_equal(original_genes, self.chromosome.genes))
        self.assertAlmostEqual(sum(self.chromosome.genes), 1.0, places=7)

    def test_swap_mutation(self):
        original_genes = self.chromosome.genes.copy()
        self.chromosome.swap_mutation(swap_probability=1.0)  # Force swap
        self.assertFalse(np.array_equal(original_genes, self.chromosome.genes))
        self.assertAlmostEqual(sum(self.chromosome.genes), 1.0, places=7)

    def test_mutate(self):
        original_genes = self.chromosome.genes.copy()
        self.chromosome.mutate(mutation_rate=0.5, mutation_amount=0.2)
        self.assertFalse(np.array_equal(original_genes, self.chromosome.genes))

    def test_get_distribution(self):
        total_armies = 100
        distribution = self.chromosome.get_distribution(total_armies)
        self.assertEqual(sum(distribution), total_armies)
        self.assertEqual(len(distribution), self.config.num_castles)

    def test_crossover(self):
        other_chromosome = Chromosome(self.config)
        new_chromosome = self.chromosome.crossover(other_chromosome)
        self.assertIsInstance(new_chromosome, Chromosome)
        self.assertEqual(len(new_chromosome.genes), self.config.num_castles)
        self.assertAlmostEqual(sum(new_chromosome.genes), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
