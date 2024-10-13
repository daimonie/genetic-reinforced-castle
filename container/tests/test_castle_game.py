import unittest
from castle.game import Config, Game
from players.player import Player


class MockPlayer(Player):
    def __init__(self, config, distribution):
        super().__init__(config)
        self.distribution = distribution

    def distribute_armies(self):
        return self.distribution

    def update(self, reward, training_progress):
        pass


class TestCastleGame(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_castles=5, armies_per_player=50)
        self.game = Game(self.config)

    def test_distribute_armies(self):
        distribution = {1: 10, 2: 20, 3: 15, 4: 5}
        self.game.distribute_armies(1, distribution)
        self.assertEqual(self.game.player1_distribution, distribution)

        distribution2 = {1: 15, 2: 15, 3: 10, 4: 10}
        self.game.distribute_armies(2, distribution2)
        self.assertEqual(self.game.player2_distribution, distribution2)

        with self.assertRaises(ValueError):
            self.game.distribute_armies(3, distribution)

    def test_calculate_score(self):
        self.game.player1_distribution = {1: 10, 2: 20, 3: 15, 4: 5}
        self.game.player2_distribution = {1: 15, 2: 15, 3: 10, 4: 10}
        player1_won, player1_score, player2_score = self.game.calculate_score()

        self.assertFalse(player1_won)
        self.assertEqual(
            player1_score, 5
        )  # Player 1 wins castles 2 (2 points) and 3 (3 points)
        self.assertEqual(
            player2_score, 5
        )  # Player 2 wins castles 1 (1 point) and 4 (4 points)

    def test_play_game(self):
        player1 = MockPlayer(self.config, {1: 25, 2: 25})
        player2 = MockPlayer(self.config, {1: 20, 2: 30})

        player1_won, player1_score, player2_score = self.game.play_game(
            player1, player2
        )

        self.assertFalse(player1_won)
        self.assertEqual(player1_score, 1)  # Player 1 wins castle 1
        self.assertEqual(player2_score, 2)  # Player 2 wins castle 2

    def test_config_initialization(self):
        config = Config(
            num_castles=8,
            armies_per_player=80,
            num_matches=200,
            num_training_rounds=2000,
        )
        self.assertEqual(config.num_castles, 8)
        self.assertEqual(config.armies_per_player, 80)
        self.assertEqual(config.num_matches, 200)
        self.assertEqual(config.num_training_rounds, 2000)
        self.assertEqual(len(config.points_per_castle), 8)
        self.assertEqual(config.points_per_castle[8], 8)


if __name__ == "__main__":
    unittest.main()
