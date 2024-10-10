import numpy as np
from typing import Dict


class Config:
    def __init__(
        self,
        num_castles=10,
        armies_per_player=100,
        num_matches=100,
        num_training_rounds=1000,
    ):
        self.num_castles = num_castles
        self.points_per_castle = {i + 1: i + 1 for i in range(self.num_castles)}
        self.armies_per_player = armies_per_player
        self.random_generator = np.random.default_rng()
        self.num_matches = num_matches

        self.learning_rate = 0.05  # Lower learning rate for stability
        self.discount_factor = (
            0.95  # Higher discount factor to value future rewards more
        )
        self.epsilon = 0.3  # Higher initial epsilon for more exploration
        self.epsilon_decay = 0.9995  # Slower decay to maintain exploration longer
        self.reinforced_training_games = num_training_rounds
        self.reinforced_win_reward = 100
        self.reinforced_lose_penalty = 50


class Game:
    def __init__(self, config):
        self.config = config
        self.player1_distribution = {}
        self.player2_distribution = {}

    def distribute_armies(self, player, distribution):
        if player == 1:
            self.player1_distribution = distribution
        elif player == 2:
            self.player2_distribution = distribution
        else:
            raise ValueError("Player must be 1 or 2")

    def calculate_score(self):
        player1_score = 0
        player2_score = 0

        for castle in self.config.points_per_castle:
            armies1 = self.player1_distribution.get(castle, 0)
            armies2 = self.player2_distribution.get(castle, 0)

            if armies1 > armies2:
                player1_score += self.config.points_per_castle[castle]
            elif armies2 > armies1:
                player2_score += self.config.points_per_castle[castle]
            # If armies are equal, no points are awarded

        player1_won = player1_score > player2_score
        return player1_won, player1_score, player2_score

    def play_game(self, player1, player2):
        distribution1 = player1.sanitize_distribute_armies()
        distribution2 = player2.sanitize_distribute_armies()

        # Distribute armies for both players
        self.distribute_armies(1, distribution1)
        self.distribute_armies(2, distribution2)

        # Calculate the game result
        player1_won, player1_score, player2_score = self.calculate_score()

        return player1_won, player1_score, player2_score
