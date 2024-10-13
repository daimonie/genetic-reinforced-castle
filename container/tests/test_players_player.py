import unittest
from castle.game import Config
from players.player import Player, FitnessPlayer, RandomPlayer


class TestPlayer(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_castles=5, armies_per_player=100)

    def test_player_abstract_methods(self):
        with self.assertRaises(TypeError):
            Player(self.config)

    def test_fitness_player_abstract_methods(self):
        with self.assertRaises(TypeError):
            FitnessPlayer(self.config)


class TestRandomPlayer(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_castles=5, armies_per_player=100)
        self.player = RandomPlayer(self.config)

    def test_distribute_armies(self):
        distribution = self.player.distribute_armies()
        self.assertEqual(sum(distribution.values()), self.config.armies_per_player)
        self.assertEqual(len(distribution), self.config.num_castles)
        for castle, armies in distribution.items():
            self.assertIsInstance(castle, int)
            self.assertIsInstance(armies, int)
            self.assertGreaterEqual(armies, 0)

    def test_update(self):
        # Ensure update method doesn't raise an exception
        try:
            self.player.update(10.0, 0.5)
        except Exception as e:
            self.fail(f"update() raised {type(e).__name__} unexpectedly!")

    def test_sanitize_distribute_armies(self):
        distribution = self.player.sanitize_distribute_armies()
        self.assertEqual(sum(distribution.values()), self.config.armies_per_player)
        self.assertEqual(len(distribution), self.config.num_castles)
        for castle, armies in distribution.items():
            self.assertIsInstance(castle, int)
            self.assertIsInstance(armies, int)
            self.assertGreaterEqual(armies, 0)


if __name__ == "__main__":
    unittest.main()
