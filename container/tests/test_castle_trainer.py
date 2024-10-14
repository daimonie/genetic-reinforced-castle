import unittest
from unittest.mock import Mock, patch
import numpy as np
from castle.game import Config, Game
from castle.trainer import Trainer
from players.player import RandomPlayer
from players.reinforcement import ReinforcedPlayer
from players.genetic import GeneticPlayer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.config = Config(num_matches=10, num_training_rounds=100)
        self.game = Game(self.config)

    def test_trainer_initialization(self):
        trainer = Trainer(self.config, self.game, "random", "reinforced")
        self.assertEqual(trainer.player_left, "random")
        self.assertEqual(trainer.player_right, "reinforced")
        self.assertEqual(len(trainer.population_left), 1)
        self.assertEqual(len(trainer.population_right), 1)
        self.assertIsInstance(trainer.population_left[0], RandomPlayer)
        self.assertIsInstance(trainer.population_right[0], ReinforcedPlayer)

    def test_create_population(self):
        trainer = Trainer(self.config, self.game, "genetic", "genetic")
        self.assertEqual(len(trainer.population_left), self.config.population_size)
        self.assertEqual(len(trainer.population_right), self.config.population_size)
        self.assertIsInstance(trainer.population_left[0], GeneticPlayer)
        self.assertIsInstance(trainer.population_right[0], GeneticPlayer)

        # Add test for different player types
        trainer_mixed = Trainer(self.config, self.game, "random", "reinforced")
        self.assertEqual(len(trainer_mixed.population_left), 1)
        self.assertEqual(len(trainer_mixed.population_right), 1)
        self.assertIsInstance(trainer_mixed.population_left[0], RandomPlayer)
        self.assertIsInstance(trainer_mixed.population_right[0], ReinforcedPlayer)

    @patch("castle.game.Game.play_game")
    @patch("players.player.RandomPlayer.distribute_armies")
    @patch("players.reinforcement.ReinforcedPlayer.distribute_armies")
    @patch("players.reinforcement.ReinforcedPlayer.update")
    def test_play_round(
        self,
        mock_update,
        mock_reinforced_distribute,
        mock_random_distribute,
        mock_play_game,
    ):
        mock_play_game.return_value = (True, 10, 5)
        mock_random_distribute.return_value = [1, 1, 1]
        mock_reinforced_distribute.return_value = [1, 1, 1]
        trainer = Trainer(self.config, self.game, "random", "reinforced")

        # Manually set last_distribution for ReinforcedPlayer
        trainer.population_right[0].last_distribution = [1, 1, 1]

        left_results, right_results = trainer.play_round(0)
        self.assertEqual(len(left_results), 1)
        self.assertEqual(len(right_results), 1)
        self.assertGreater(left_results[0][1], right_results[0][1])

        # Check if update method was called for the reinforced player
        mock_update.assert_called_once()

        # Test with different game outcome
        mock_play_game.return_value = (False, 5, 10)
        left_results, right_results = trainer.play_round(1)
        self.assertLess(left_results[0][1], right_results[0][1])

    def test_evolve_population(self):
        trainer = Trainer(self.config, self.game, "genetic", "genetic")
        initial_population = trainer.population_left.copy()
        mock_results = [(player, np.random.random()) for player in initial_population]
        new_population = trainer.evolve_population(
            initial_population, mock_results, "left"
        )
        self.assertEqual(len(new_population), len(initial_population))
        self.assertNotEqual(new_population, initial_population)

    @patch("players.reinforcement.ReinforcedPlayer.distribute_armies")
    @patch("players.reinforcement.ReinforcedPlayer.update")
    def test_update_players(self, mock_update, mock_distribute):
        mock_distribute.return_value = [1, 1, 1]
        trainer = Trainer(self.config, self.game, "reinforced", "reinforced")
        left_player = trainer.population_left[0]
        right_player = trainer.population_right[0]

        # Manually set last_distribution for both players
        left_player.last_distribution = [1, 1, 1]
        right_player.last_distribution = [1, 1, 1]

        trainer.update_players(left_player, right_player, 10, -10, 0.5)

        # Check if update method was called for both players
        self.assertEqual(mock_update.call_count, 2)
        mock_update.assert_any_call(10, training_progress=0.5)
        mock_update.assert_any_call(-10, training_progress=0.5)

    @patch("castle.trainer.Trainer.play_round")
    @patch("castle.trainer.Trainer.evolve_population")
    def test_train(self, mock_evolve, mock_play_round):
        mock_play_round.return_value = ([(Mock(), 1)], [(Mock(), 0)])
        trainer = Trainer(self.config, self.game, "genetic", "genetic")

        # Update the num_training_rounds to match the actual implementation
        trainer.config.num_training_rounds = 1

        trainer.train()

        self.assertEqual(mock_play_round.call_count, trainer.config.num_training_rounds)
        self.assertEqual(mock_evolve.call_count, trainer.config.num_training_rounds * 2)


if __name__ == "__main__":
    unittest.main()
