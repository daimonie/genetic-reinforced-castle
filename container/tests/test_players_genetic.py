import unittest
import numpy as np
from castle.game import Config
from players.genetic import GeneticPlayer
from players.chromosome import Chromosome


class TestGeneticPlayer(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_castles=5, armies_per_player=100)
        self.player = GeneticPlayer(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.player.chromosome, Chromosome)
        self.assertEqual(len(self.player.rewards), 0)

    def test_distribute_armies(self):
        distribution = self.player.distribute_armies()
        self.assertEqual(sum(distribution.values()), self.config.armies_per_player)
        self.assertEqual(len(distribution), self.config.num_castles)

    def test_update(self):
        initial_rewards = len(self.player.rewards)
        self.player.update(
            50, 0.5
        )  # Add a training progress value (0.5 in this example)
        self.assertEqual(len(self.player.rewards), initial_rewards + 1)
        self.assertAlmostEqual(
            self.player.rewards[-1], 50
        )  # Use assertAlmostEqual for float comparison

    def test_mutate(self):
        original_genes = self.player.chromosome.genes.copy()
        self.player.mutate(mutation_rate=0.5, mutation_amount=0.2)
        self.assertFalse(np.array_equal(original_genes, self.player.chromosome.genes))

    def test_copy(self):
        self.player.rewards = [1, 2, 3]
        copied_player = self.player.copy()
        self.assertIsNot(self.player, copied_player)
        self.assertEqual(len(copied_player.rewards), 0)

    def test_get_average_reward(self):
        self.player.rewards = [1, 2, 3, 4, 5]
        self.assertEqual(self.player.get_average_reward(), 3)

        self.player.rewards = []
        self.assertEqual(self.player.get_average_reward(), 0)

    def test_get_recent_performance(self):
        self.player.rewards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.assertEqual(self.player.get_recent_performance(), 7.5)

        self.player.rewards = [1, 2, 3]
        self.assertEqual(self.player.get_recent_performance(), 2)

    def test_fitness(self):
        self.player.rewards = [1, 2, 3, 4, 5]
        fitness = self.player.fitness()
        self.assertGreater(fitness, 0)
        self.assertLess(fitness, 5)

    def test_crossover(self):
        other_player = GeneticPlayer(self.config)
        original_chromosome = self.player.chromosome
        self.player.crossover(other_player)
        self.assertIsNot(self.player.chromosome, original_chromosome)
        self.assertEqual(len(self.player.rewards), 0)


if __name__ == "__main__":
    unittest.main()
