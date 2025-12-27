import random
from functools import partial
from typing import Callable, Optional

import numpy as np


class GA:
    def __init__(
        self,
        fitness_func: Callable,
        population_size: int,
        num_genes: int,
        mutation_rate: float,
        max_iter: int,
        epsilon: float,
        seed: Optional[int] = None,
    ):
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.num_genes = num_genes
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        if not seed:
            self.seed = random.seed()
        else:
            self.seed = seed

    def initialize_population(self) -> np.ndarray:
        self.population = np.random.randint(
            low=0, high=2, size=(self.population_size, self.num_genes)
        )
        return self.population

    def crossover(self, x1: np.ndarray, x2: np.ndarray):
        pass

    def mutate(self, x: np.ndarray):
        pass

    def evolve(self):
        # Create initial population and evaluate it
        population = self.initialize_population()
        scores = self.fitness_func(population)

        indices = np.argsort(scores)[::-1]
        scores = scores[indices]
        population = population[indices]

        for iter in range(self.max_iter):
            pass


def fitness_knapsack(
    population: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    max_capacity: int,
) -> np.ndarray:
    total_weights = np.sum(population * weights, axis=1)
    total_values = np.sum(population * values, axis=1)

    total_values[total_weights > max_capacity] = 0.0

    return total_values


def main():
    x = np.array([[1, 0, 0, 1, 1, 0, 1]])
    weights = np.random.rand(*x.shape)
    values = np.random.rand(*x.shape)
    max_capacity = 3

    print(f"Candidate solution: {x}\n")
    print(f"Weight of every object: {weights}")
    print(f"Total weight of all objects: {np.sum(weights)}\n")
    print(f"Value of every object: {values}")
    print(f"Total value of all objects: {np.sum(values)}\n")

    fitness_score = fitness_knapsack(x, weights, values, max_capacity)
    print(f"Fitness score: {fitness_score}")

    fitness_func = partial(
        fitness_knapsack, weights=weights, values=values, max_capacity=max_capacity
    )

    optimizer = GA(
        fitness_func=fitness_func,
        population_size=30,
        num_genes=x.shape[1],
        mutation_rate=0.05,
        max_iter=100,
        epsilon=0.2,
    )
    optimizer.evolve()


if __name__ == "__main__":
    main()
