# Castle Game AI

This project implements an AI-driven game where different types of players compete in a strategic castle conquest game. The main script, `main.py`, allows you to run simulations and compare different AI strategies.

## Player Types

1. **Random Player**: Makes random moves without any strategy.
2. **Reinforced Player**: Uses reinforcement learning to improve its strategy over time.
3. **Genetic Player**: Employs genetic algorithms to evolve and improve its strategy across generations.

## Running the Game

To run the game, use the following command:

    poetry run python main.py [OPTIONS]

### Command-line Arguments

- `--left-player`: Type of left player (default: "random")
- `--right-player`: Type of right player (default: "random")
- `--num-matches`: Number of matches to play (default: 100)
- `--num-training-rounds`: Number of training rounds (default: 10000)
- `--train/--no-train`: Whether to train the players before matches (default: False)

Example:

    poetry run python main.py --left-player reinforced --right-player random --num-matches 1000 --num-training-rounds 1000 --train

## Player Classes

### RandomPlayer

The `RandomPlayer` class implements a simple strategy that makes random moves. It serves as a baseline for comparing other AI strategies.

### ReinforcedPlayer

The `ReinforcedPlayer` class uses reinforcement learning to improve its strategy over time. It employs a pseudostate representation of the game environment to make decisions. This pseudostate captures essential information about the game state without revealing complete information, allowing the player to learn and adapt its strategy based on partial observations. The player learns from the outcomes of its actions and adjusts its behavior accordingly, updating its policy to maximize long-term rewards. Through repeated interactions with the environment, the `ReinforcedPlayer` gradually refines its decision-making process, becoming more effective at conquering castles and outperforming opponents.

### GeneticPlayer

The `GeneticPlayer` class employs genetic algorithms to evolve and improve its strategy across generations. It uses a `Chromosome` class to represent its strategy, which encodes the player's decision-making process. The `Chromosome` class includes methods for crossover and mutation:

- Crossover: Combines genetic information from two parent chromosomes to create offspring, potentially inheriting beneficial traits from both parents.
- Mutation: Introduces small random changes to a chromosome, allowing for exploration of new strategies.

These genetic operations allow the `GeneticPlayer` to adapt and refine its strategy over time, creating new and potentially more effective approaches based on successful ones from previous generations.

## Genetic vs Reinforced Battle

To compare the performance of the Genetic and Reinforced players, you can run a battle between them using the following command:

    poetry run python main.py --left-player genetic --right-player reinforced --num-matches 100 --num-training-rounds 20000 --train

This will train both players for 10,000 rounds and then play 1,000 matches between them. The results will show the win percentage for each player, allowing you to compare their effectiveness.

The training progress will be saved as a plot in the `output` directory, which you can analyze to see how each player's performance evolved during training.
