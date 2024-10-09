import unittest
from unittest.mock import Mock
import numpy as np
from players.player import Player, RandomPlayer
from players.reinforcement import ReinforcedPlayer


class TestPlayer(unittest.TestCase):
    def setUp(self):
        self.config = Mock()
        self.config.num_castles = 5
        self.config.armies_per_player = 100
        self.config.points_per_castle = {
            i + 1: i + 1 for i in range(self.config.num_castles)
        }
        self.config.random_generator = np.random.default_rng(42)

    def test_player_abstract_methods(self):
        with self.assertRaises(TypeError):
            Player(self.config)

    def test_sanitize_distribute_armies(self):
        class TestPlayer(Player):
            def distribute_armies(self):
                return {1: 30, 2: 30, 3: 30}  # Intentionally less than total

        player = TestPlayer(self.config)
        sanitized = player.sanitize_distribute_armies()
        self.assertEqual(sum(sanitized.values()), self.config.armies_per_player)
        self.assertEqual(set(sanitized.keys()), set(range(1, 4)))

    def test_random_player_distribute_armies(self):
        player = RandomPlayer(self.config)
        distribution = player.distribute_armies()
        self.assertEqual(sum(distribution.values()), self.config.armies_per_player)
        self.assertEqual(
            set(distribution.keys()), set(range(1, self.config.num_castles + 1))
        )


class TestReinforcedPlayer(unittest.TestCase):
    def setUp(self):
        self.config = Mock()
        self.config.num_castles = 5
        self.config.armies_per_player = 100
        self.config.points_per_castle = {
            i + 1: i + 1 for i in range(self.config.num_castles)
        }
        self.config.random_generator = np.random.default_rng(42)
        self.config.epsilon = 0.1
        self.config.learning_rate = 0.1
        self.config.epsilon_decay = 0.995
        self.config.reinforced_training_games = 1000

    def test_reinforced_player_initialization(self):
        player = ReinforcedPlayer(self.config)
        self.assertIsNotNone(player.qmatrix)
        self.assertEqual(
            player.qmatrix.shape,
            (self.config.num_castles, self.config.armies_per_player + 1),
        )

    def test_reinforced_player_distribute_armies(self):
        player = ReinforcedPlayer(self.config)
        distribution = player.distribute_armies()
        self.assertEqual(sum(distribution.values()), self.config.armies_per_player)
        self.assertEqual(
            set(distribution.keys()), set(range(1, self.config.num_castles + 1))
        )

    def test_reinforced_player_update(self):
        player = ReinforcedPlayer(self.config)
        player.last_distribution = {1: 50, 2: 50}
        initial_qvalue = player.qmatrix[0, 50]
        player.update(reward=10, training_progress=0.5)
        self.assertNotEqual(initial_qvalue, player.qmatrix[0, 50])


if __name__ == "__main__":
    unittest.main()
