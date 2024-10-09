import unittest
from unittest.mock import Mock
import numpy as np
from main import Config, Game
from players.player import RandomPlayer
from players.reinforcement import ReinforcedPlayer


class TestConfig(unittest.TestCase):
    def test_config_initialization(self):
        config = Config(num_castles=5, armies_per_player=50, num_matches=20)
        self.assertEqual(config.num_castles, 5)
        self.assertEqual(config.armies_per_player, 50)
        self.assertEqual(config.num_matches, 20)
        self.assertEqual(config.points_per_castle, {1: 1, 2: 2, 3: 3, 4: 4, 5: 5})


class TestGame(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_castles=3, armies_per_player=30, num_matches=10)
        self.game = Game(self.config)

    def test_distribute_armies(self):
        self.game.distribute_armies(1, {1: 10, 2: 10, 3: 10})
        self.assertEqual(self.game.player1_distribution, {1: 10, 2: 10, 3: 10})

        self.game.distribute_armies(2, {1: 15, 2: 10, 3: 5})
        self.assertEqual(self.game.player2_distribution, {1: 15, 2: 10, 3: 5})

        with self.assertRaises(ValueError):
            self.game.distribute_armies(3, {1: 10, 2: 10, 3: 10})

    def test_calculate_score(self):
        self.game.player1_distribution = {1: 10, 2: 15, 3: 5}
        self.game.player2_distribution = {1: 15, 2: 10, 3: 5}
        player1_won, player1_score, player2_score = self.game.calculate_score()
        self.assertEqual(player1_score, 2)
        self.assertEqual(player2_score, 1)
        self.assertTrue(player1_won)

    def test_play_game(self):
        player1 = RandomPlayer(self.config)
        player2 = RandomPlayer(self.config)
        player1_won, player1_score, player2_score = self.game.play_game(
            player1, player2
        )
        self.assertIn(player1_won, [True, False])
        self.assertGreaterEqual(player1_score, 0)
        self.assertGreaterEqual(player2_score, 0)
        self.assertLess(player1_score, 7)  # sum of points for 3 castles
        self.assertLess(player2_score, 7)  # sum of points for 3 castles


class TestReinforcedPlayerIntegration(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_castles=3, armies_per_player=30, num_matches=10)
        self.config.reinforced_training_games = 100  # Reduce for faster testing
        self.game = Game(self.config)

    def test_reinforced_player_training(self):
        reinforced_player = ReinforcedPlayer(self.config)
        random_player = RandomPlayer(self.config)

        initial_q_matrix = reinforced_player.get_qmatrix().copy()

        # Train the reinforced player
        for _ in range(self.config.reinforced_training_games):
            reinforced_won, reinforced_score, opponent_score = self.game.play_game(
                reinforced_player, random_player
            )
            reward = (
                reinforced_score + self.config.reinforced_win_reward
                if reinforced_won
                else reinforced_score - self.config.reinforced_lose_penalty
            )
            reinforced_player.update(
                reward,
                training_progress=(_ + 1) / self.config.reinforced_training_games,
            )

        final_q_matrix = reinforced_player.get_qmatrix()

        # Check if Q-matrix has been updated
        self.assertFalse(np.array_equal(initial_q_matrix, final_q_matrix))


if __name__ == "__main__":
    unittest.main()
