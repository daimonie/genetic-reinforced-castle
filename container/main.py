import numpy as np
from players.player import Player, RandomPlayer
from players.reinforcement import ReinforcedPlayer


class Config:
    def __init__(self, num_castles=10, armies_per_player=100, num_matches=100):
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
        self.reinforced_training_games = 1000000
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


if __name__ == "__main__":
    config = Config(num_matches=100)
    print(f"Number of castles: {config.num_castles}")
    print(f"Points per castle: {config.points_per_castle}")
    print(f"Armies per player: {config.armies_per_player}")

    player1 = RandomPlayer(config)
    player2 = RandomPlayer(config)
    game = Game(config)
    player1_wins = 0

    for game_number in range(config.num_matches):
        player1_won, _, _ = game.play_game(player1, player2)
        player1_wins += 1 if player1_won else 0

    player1_win_percentage = (player1_wins / config.num_matches) * 100

    print(
        f"\nRandom vs Random ({config.num_matches} matches): win percentage: {player1_win_percentage:.2f}%"
    )

    # Create a ReinforcedPlayer instance
    reinforced_player = ReinforcedPlayer(config)

    # Train the reinforced player
    reinforced_wins = 0
    for game_number in range(config.reinforced_training_games):
        reinforced_won, reinforced_score, opponent_score = game.play_game(
            reinforced_player, player2
        )

        # Calculate reward based on game outcome
        if reinforced_won:
            reward = reinforced_score + config.reinforced_win_reward
        else:
            reward = reinforced_score - config.reinforced_lose_penalty

        # Update the reinforced player's Q-matrix
        training_progress = (game_number + 1) / config.reinforced_training_games
        reinforced_player.update(reward, training_progress=training_progress)

        # printing logic
        # Update wins and scores for the last 1000 games
        if game_number == 0:
            reinforced_wins = []
            reinforced_scores = []
        reinforced_wins.append(1 if reinforced_won else 0)
        reinforced_scores.append(reinforced_score)
        current_wins = sum(reinforced_wins[-1000:])

        # Calculate current win percentage and average score for the last 1000 games
        current_win_percentage = (current_wins / min(1000, len(reinforced_wins))) * 100
        average_score = sum(reinforced_scores[-1000:]) / min(
            1000, len(reinforced_scores)
        )

        if (game_number + 1) % 1000 == 0:
            print(
                f"\rGame {game_number + 1}/{config.reinforced_training_games} - Reinforced Player wins percentage {current_win_percentage:.2f}% last 1000 games, Average score: {average_score:.2f}",
                end="",
                flush=True,
            )

    print(
        f"\nReinforced player training completed after {config.reinforced_training_games} games."
    )

    # Play games with the trained reinforced player
    reinforced_wins = 0
    for game_number in range(config.num_matches):
        reinforced_won, _, _ = game.play_game(reinforced_player, player2)
        reinforced_wins += 1 if reinforced_won else 0

        reinforced_win_percentage = (reinforced_wins / config.num_matches) * 100
        print(
            f"\rGame {game_number + 1}/{config.num_matches} - Reinforced Player wins: {reinforced_wins}",
            end="",
            flush=True,
        )

    reinforced_win_percentage = (reinforced_wins / config.num_matches) * 100

    print(
        f"\nReinforced vs Random ({config.num_matches} matches): win percentage: {reinforced_win_percentage:.2f}%"
    )
