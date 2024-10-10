import click
import numpy as np
from players.player import Player, RandomPlayer
from players.reinforcement import ReinforcedPlayer, GymPlayer
from castle.game import Game, Config


def create_player(player_type, config):
    if player_type == "random":
        return RandomPlayer(config)
    elif player_type == "reinforced":
        return ReinforcedPlayer(config)
    elif player_type == "gym":
        return GymPlayer(config)
    else:
        raise ValueError(f"Invalid player type: {player_type}")


def train_players(player1, player2, config, game):
    print(
        f"Training {player1.__class__.__name__} against {player2.__class__.__name__}..."
    )
    player1_wins = []
    player1_scores = []

    for game_number in range(config.reinforced_training_games):
        player1_won, player1_score, player2_score = game.play_game(player1, player2)

        # Calculate rewards
        if player1_won:
            player1_reward = player1_score + config.reinforced_win_reward
            player2_reward = player2_score - config.reinforced_lose_penalty
        else:
            player1_reward = player1_score - config.reinforced_lose_penalty
            player2_reward = player2_score + config.reinforced_win_reward

        # Update players
        training_progress = (game_number + 1) / config.reinforced_training_games
        player1.update(player1_reward, training_progress=training_progress)
        player2.update(player2_reward, training_progress=training_progress)

        # Update statistics
        player1_wins.append(1 if player1_won else 0)
        player1_scores.append(player1_score)

        # Print progress
        if (game_number + 1) % 1000 == 0:
            current_wins = sum(player1_wins[-1000:])
            current_win_percentage = (current_wins / min(1000, len(player1_wins))) * 100
            average_score = sum(player1_scores[-1000:]) / min(1000, len(player1_scores))
            print(
                f"\rGame {game_number + 1}/{config.reinforced_training_games} - {player1.__class__.__name__} wins percentage {current_win_percentage:.2f}% last 1000 games, Average score: {average_score:.2f}",
                end="",
                flush=True,
            )

    print(f"\nTraining completed after {config.reinforced_training_games} games.")


@click.command()
@click.option(
    "--left-player",
    type=click.Choice(["random", "reinforced", "gym"]),
    default="random",
    help="Type of left player",
)
@click.option(
    "--right-player",
    type=click.Choice(["random", "reinforced", "gym"]),
    default="random",
    help="Type of right player",
)
@click.option("--num-matches", default=100, help="Number of matches to play")
@click.option("--num-training-rounds", default=10000, help="Number of training rounds")
@click.option(
    "--train/--no-train",
    default=False,
    help="Whether to train the players before matches",
)
def main(left_player, right_player, num_matches, num_training_rounds, train):
    config = Config(num_matches=num_matches, num_training_rounds=num_training_rounds)
    print(f"Number of castles: {config.num_castles}")
    print(f"Points per castle: {config.points_per_castle}")
    print(f"Armies per player: {config.armies_per_player}")
    print(f"Number of training rounds: {num_training_rounds}")

    player1 = create_player(left_player, config)
    player2 = create_player(right_player, config)
    game = Game(config)

    if train:
        train_players(player1, player2, config, game)

    player1_wins = 0

    for game_number in range(config.num_matches):
        player1_won, _, _ = game.play_game(player1, player2)
        player1_wins += 1 if player1_won else 0

        player1_win_percentage = (player1_wins / (game_number + 1)) * 100
        print(
            f"\rGame {game_number + 1}/{config.num_matches} - {left_player.capitalize()} Player wins: {player1_wins}",
            end="",
            flush=True,
        )

    player1_win_percentage = (player1_wins / config.num_matches) * 100

    print(
        f"\n{left_player.capitalize()} vs {right_player.capitalize()} ({config.num_matches} matches): {left_player.capitalize()} win percentage: {player1_win_percentage:.2f}%"
    )


if __name__ == "__main__":
    main()
