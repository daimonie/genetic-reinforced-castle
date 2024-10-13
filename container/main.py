import math
from players.player import RandomPlayer
from players.reinforcement import ReinforcedPlayer
from players.genetic import GeneticPlayer
from castle.game import Game, Config
from castle.trainer import Trainer
import click
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


@click.command()
@click.option(
    "--left-player",
    type=click.Choice(["random", "reinforced", "genetic"]),
    default="random",
    help="Type of left player",
)
@click.option(
    "--right-player",
    type=click.Choice(["random", "reinforced", "genetic"]),
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

    game = Game(config)
    trainer = Trainer(config, game, left_player, right_player)

    if train:
        training_data = trainer.train()

    # After training, get the best players from each population
    player1 = trainer.best_player("left")
    player2 = trainer.best_player("right")

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
    plot_training_results(
        training_data, left_player, right_player, player1_win_percentage
    )


def plot_training_results(
    training_data, left_player, right_player, final_left_win_percentage
):
    left_wins, right_wins = training_data
    rounds = range(1, len(left_wins) + 1)
    # Randomly select 100 data points
    num_samples = min(100, len(left_wins))
    sample_indices = np.random.choice(len(left_wins), num_samples, replace=False)
    sample_indices.sort()  # Sort indices to maintain chronological order

    if len(left_wins) > 250:
        # Create binned data
        num_bins = min(100, len(left_wins))
        bin_size = len(left_wins) // num_bins
        binned_left_wins = []
        binned_right_wins = []
        binned_rounds = []

        for i in range(0, len(left_wins), bin_size):
            bin_end = min(i + bin_size, len(left_wins))
            binned_left_wins.append(sum(left_wins[i:bin_end]))
            binned_right_wins.append(sum(right_wins[i:bin_end]))
            binned_rounds.append(rounds[i])

        plt.figure(figsize=(12, 6))
        plt.scatter(
            binned_rounds,
            binned_left_wins,
            label=f"{left_player.capitalize()} Wins",
            alpha=0.7,
        )
        plt.scatter(
            binned_rounds,
            binned_right_wins,
            label=f"{right_player.capitalize()} Wins",
            alpha=0.7,
        )
    else:
        plt.figure(figsize=(12, 6))
        plt.scatter(
            rounds, left_wins, label=f"{left_player.capitalize()} Wins", alpha=0.7
        )
        plt.scatter(
            rounds, right_wins, label=f"{right_player.capitalize()} Wins", alpha=0.7
        )

    plt.title(
        f"Training Progress: {left_player.capitalize()} vs {right_player.capitalize()} (Final {left_player.capitalize()} Win %: {final_left_win_percentage:.2f}%)"
    )
    plt.xlabel("Training Rounds")
    plt.ylabel("Wins")
    plt.legend()
    plt.savefig(f"output/training_progress_{left_player}_vs_{right_player}.png")
    plt.close()

    # When running in a Docker container, we can't show the plot directly
    # Instead, we'll save it to a file and print a message
    print(
        f"Training progress plot saved as 'output/training_progress_{left_player}_vs_{right_player}.png'"
    )
    print(
        "To view the plot, you'll need to copy the file from the Docker container to your local machine."
    )


if __name__ == "__main__":
    main()
