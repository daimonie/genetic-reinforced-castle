import math
from players.player import RandomPlayer
from players.reinforcement import ReinforcedPlayer
from players.genetic import GeneticPlayer
from castle.game import Game, Config


def create_player(player_type, config):
    if player_type == "random":
        return RandomPlayer(config)
    elif player_type == "reinforced":
        if player_type == "reinforced":
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
        if player_type.lower() in self.config.population_players:
            return [
                create_player(player_type, self.config)
                for _ in range(self.config.population_size)
            ]
        return [create_player(player_type, self.config)]

    def train(self):
        print(
            f"Training {self.player_left.capitalize()} against {self.player_right.capitalize()}..."
        )

        for round_number in range(self.num_rounds):
            left_results, right_results = self.play_round(round_number)
            self.evolve_populations(left_results, right_results)
            self.print_progress(round_number, left_results, right_results)

        print(f"\nTraining completed after {self.num_rounds} rounds.")

    def play_round(self, round_number):
        left_results = []
        right_results = []
        training_progress = (round_number + 1) / self.num_rounds

        for left_player in self.population_left:
            for right_player in self.population_right:
                player1_reward, player2_reward = self.play_game(
                    left_player, right_player
                )
                self.update_players(
                    left_player,
                    right_player,
                    player1_reward,
                    player2_reward,
                    training_progress,
                )
                left_results.append((left_player, player1_reward))
                right_results.append((right_player, player2_reward))

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
        if self.population_left[0].__class__ in self.config.population_players:
            player_class = create_player(
                self.population_left[0].__class__, self.config
            ).__class__
            self.population_left = self.evolve_population(
                self.population_left, left_results, player_class
            )

        if self.population_right[0].__class__ in self.config.population_players:
            player_class = create_player(
                self.population_right[0].__class__, self.config
            ).__class__
            self.population_right = self.evolve_population(
                self.population_right, right_results, player_class
            )

    def print_progress(self, round_number, left_results, right_results):
        avg_left_score = np.mean([r[1] for r in left_results])
        avg_right_score = np.mean([r[1] for r in right_results])
        print(
            f"\rRound {round_number + 1}/{self.num_rounds} - Left avg score: {avg_left_score:.2f}, Right avg score: {avg_right_score:.2f}",
            end="",
            flush=True,
        )

    def evolve_population(self, population, results, player_class):
        # Sort players by their rewards
        sorted_players = sorted(results, key=lambda x: x[1], reverse=True)

        new_population = []

        # Top performers reproduce twice
        top_count = self.config.genetic_top_reproduction
        for player, reward in sorted_players[:top_count]:
            new_population.append(player)
            new_player = player.copy()
            new_player.mutate()

            new_population.append(new_player)
            another_new_player = player.copy()
            another_new_player.mutate()

            new_population.append(another_new_player)

        # Middle performers stay in the pool, possibly updating
        middle_count = self.config.genetic_middle_reproduction
        for player, reward in sorted_players[top_count : top_count + middle_count]:
            new_population.append(player)
            if np.random.random() < 0.5:  # 50% chance to update
                new_player = player.copy()
                new_player.mutate()
                new_population.append(new_player)
        # Fill the rest of the population with new players of the same type
        while len(new_population) < len(population):
            if player_class.__name__ == self.population_left[0].__class__.__name__:
                new_population.append(create_player(self.player_left, self.config))
            elif player_class.__name__ == self.population_right[0].__class__.__name__:
                new_population.append(create_player(self.player_right, self.config))
            else:
                raise ValueError(f"Unexpected player class: {player_class.__name__}")

        # Randomly shuffle the population
        self.config.random_generator.shuffle(new_population)

        # Select the first population_size players
        return new_population[: self.config.population_size]


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

    player1 = create_player(left_player, config)
    player2 = create_player(right_player, config)
    game = Game(config)

    if train:
        trainer = Trainer(player1, player2, config, game, left_player, right_player)
        trainer.train()

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
