import math
import itertools
import numpy as np
from players.player import RandomPlayer
from players.reinforcement import ReinforcedPlayer
from players.genetic import GeneticPlayer


def create_player(player_type, config):
    if player_type == "random":
        return RandomPlayer(config)
    elif player_type == "reinforced":
        return ReinforcedPlayer(config)
    elif player_type == "genetic":
        return GeneticPlayer(config)
    else:
        raise ValueError(f"Invalid player type: {player_type}")


class Trainer:
    def __init__(self, config, game, player_left, player_right):
        self.config = config
        self.game = game
        self.player_left = player_left
        self.player_right = player_right
        self.initialize()

    def initialize(self):
        self.population_left = self.create_population(self.player_left)
        self.population_right = self.create_population(self.player_right)
        self.pop_size_left = len(self.population_left)
        self.pop_size_right = len(self.population_right)
        self.num_rounds = math.ceil(
            self.config.num_training_rounds
            / max(self.pop_size_left, self.pop_size_right)
        )

    def create_population(self, player_type):
        if player_type in self.config.population_players:
            print(
                f"Creating a population of {self.config.population_size} {player_type} players"
            )
            return [
                create_player(player_type, self.config)
                for _ in range(self.config.population_size)
            ]
        print(f"Creating a single {player_type} player")
        return [create_player(player_type, self.config)]

    def train(self):
        print(
            f"Training {self.player_left.capitalize()} against {self.player_right.capitalize()}..."
        )

        left_wins = []
        right_wins = []

        for round_number in range(self.num_rounds):
            left_results, right_results = self.play_round(round_number)
            self.evolve_populations(left_results, right_results)
            self.print_progress(round_number, left_results, right_results)

            # Count wins for each side in this round
            left_round_wins = sum(1 for _, score in left_results if score > 0)
            right_round_wins = sum(1 for _, score in right_results if score > 0)

            left_wins.append(left_round_wins)
            right_wins.append(right_round_wins)

        print(f"\nTraining completed after {self.num_rounds} rounds.")
        return [left_wins, right_wins]

    def play_round(self, round_number):
        left_results = []
        right_results = []
        training_progress = (round_number + 1) / self.num_rounds
        # Ensure everyone has a match by shuffling and pairing
        shuffled_left = self.config.random_generator.permutation(self.population_left)
        shuffled_right = self.config.random_generator.permutation(self.population_right)

        # Pair players, repeating the smaller population if necessary
        for left_player, right_player in zip(
            shuffled_left,
            itertools.cycle(shuffled_right)
            if len(shuffled_right) < len(shuffled_left)
            else shuffled_right,
        ):
            player1_reward, player2_reward = self.play_game(left_player, right_player)
            self.update_players(
                left_player,
                right_player,
                player1_reward,
                player2_reward,
                training_progress,
            )
            left_results.append((left_player, player1_reward))
            right_results.append((right_player, player2_reward))
        # Calculate the percentage of positive scores for the left population
        positive_left_scores = sum(1 for _, score in left_results if score > 0)
        total_left_scores = len(left_results)
        positive_percentage = (
            (positive_left_scores / total_left_scores) * 100
            if total_left_scores > 0
            else 0
        )

        return left_results, right_results

    def play_game(self, left_player, right_player):
        player1_won, player1_score, player2_score = self.game.play_game(
            left_player, right_player
        )

        if player1_won:
            player1_reward = player1_score + self.config.reinforced_win_reward
            player2_reward = player2_score - self.config.reinforced_lose_penalty
        else:
            player1_reward = player1_score - self.config.reinforced_lose_penalty
            player2_reward = player2_score + self.config.reinforced_win_reward

        return player1_reward, player2_reward

    def update_players(
        self,
        left_player,
        right_player,
        player1_reward,
        player2_reward,
        training_progress,
    ):
        left_player.update(player1_reward, training_progress=training_progress)
        right_player.update(player2_reward, training_progress=training_progress)

    def evolve_populations(self, left_results, right_results):
        if self.player_left in self.config.population_players:
            self.population_left = self.evolve_population(
                self.population_left, left_results, "left"
            )

        if self.player_right in self.config.population_players:
            self.population_right = self.evolve_population(
                self.population_right, right_results, "right"
            )

    def print_progress(self, round_number, left_results, right_results):
        avg_left_score = np.mean([r[1] for r in left_results])
        avg_right_score = np.mean([r[1] for r in right_results])
        print(
            f"\rRound {round_number + 1}/{self.num_rounds} - {self.player_left} avg score: {avg_left_score:.2f}, {self.player_right} avg score: {avg_right_score:.2f}",
            end="",
            flush=True,
        )

    def evolve_population(self, population, results, left_or_right):
        # Sort players by their fitness
        sorted_players = sorted(results, key=lambda x: x[0].fitness(), reverse=True)

        # Save the best player
        best_player = sorted_players[0][0]
        if left_or_right == "left":
            self.best_left_player = best_player
        else:
            self.best_right_player = best_player

        new_population = []

        # Elitism: Keep top 10% unchanged and use them for reproduction
        elitism_count = max(1, int(0.1 * len(population)))
        elite_players = [player for player, _ in sorted_players[:elitism_count]]
        new_population.extend(elite_players)

        # Fill the rest of the population
        while len(new_population) < len(population):
            # Tournament selection
            tournament_size = 5
            tournament_indices = self.config.random_generator.choice(
                len(sorted_players), tournament_size, replace=False
            )
            tournament = [sorted_players[i] for i in tournament_indices]
            parent1 = max(tournament, key=lambda x: x[0].fitness())[0]

            # Select second parent from elite players
            parent2 = self.config.random_generator.choice(elite_players)

            # Create offspring
            offspring = parent1.copy()
            offspring.crossover(parent2)
            offspring.mutate()
            new_population.append(offspring)

        return new_population

    def best_player(self, left_or_right):
        """
        Returns the best player from either the left or right population.

        Args:
            left_or_right (str): Either "left" or "right" to specify which population to choose from.

        Returns:
            The best player instance from the specified population.

        Raises:
            ValueError: If an invalid value for left_or_right is provided.
        """
        if left_or_right not in ["left", "right"]:
            raise ValueError("Invalid argument. Must be either 'left' or 'right'.")

        player_type = self.player_left if left_or_right == "left" else self.player_right

        if player_type in self.config.population_players:
            return (
                self.best_left_player
                if left_or_right == "left"
                else self.best_right_player
            )
        else:
            return (
                self.population_left[0]
                if left_or_right == "left"
                else self.population_right[0]
            )
