import unittest
import numpy as np
from castle.game import Config
from players.reinforcement import ReinforcedPlayer


class TestReinforcedPlayer(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.player = ReinforcedPlayer(self.config)

    def test_initialization(self):
        self.assertIsInstance(self.player.qmatrix, np.ndarray)
        expected_shape = (self.config.armies_per_player + 1, self.config.num_castles)
        self.assertEqual(self.player.qmatrix.shape, expected_shape)

    def test_set_get_qmatrix(self):
        new_qmatrix = np.random.rand(
            self.config.num_castles, self.config.armies_per_player + 1
        )
        self.player.set_qmatrix(new_qmatrix)
        np.testing.assert_array_equal(self.player.get_qmatrix(), new_qmatrix)

    def test_distribute_armies(self):
        distribution = self.player.distribute_armies()
        self.assertEqual(sum(distribution.values()), self.config.armies_per_player)
        self.assertEqual(len(distribution), self.config.num_castles)
        for castle, armies in distribution.items():
            self.assertIsInstance(castle, int)
            self.assertIsInstance(armies, int)
            self.assertGreaterEqual(armies, 0)

    def test_update(self):
        initial_distribution = self.player.distribute_armies()
        initial_qmatrix = self.player.get_qmatrix().copy()

        # First update
        self.player.update(50, 0.5)  # Positive reward
        self.assertFalse(np.array_equal(initial_qmatrix, self.player.get_qmatrix()))

        # Second update (should not raise an error)
        second_qmatrix = self.player.get_qmatrix().copy()
        self.player.update(50, 0.5)
        self.assertFalse(np.array_equal(second_qmatrix, self.player.get_qmatrix()))

        # Check if a new distribution can be made after updates
        new_distribution = self.player.distribute_armies()
        self.assertIsInstance(new_distribution, dict)
        self.assertEqual(sum(new_distribution.values()), self.config.armies_per_player)

    def test_update_normalization(self):
        self.player.distribute_armies()

        self.player.update(self.config.reinforced_win_reward, 0.5)
        max_q_value = np.max(self.player.get_qmatrix())
        self.assertLessEqual(max_q_value, 1.0)

        self.player.distribute_armies()
        self.player.update(-self.config.reinforced_lose_penalty, 0.5)
        min_q_value = np.min(self.player.get_qmatrix())
        self.assertGreaterEqual(min_q_value, -1.0)

    def test_epsilon_greedy_strategy(self):
        # Test exploitation
        self.config.epsilon = 0
        exploited_distribution = self.player.distribute_armies()

        # Test exploration
        self.config.epsilon = 1
        explored_distribution = self.player.distribute_armies()

        self.assertNotEqual(exploited_distribution, explored_distribution)


if __name__ == "__main__":
    unittest.main()
