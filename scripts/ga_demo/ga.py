import random
from functools import partial
from typing import Callable, Optional, Tuple

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
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def initialize_population(self) -> np.ndarray:
        self.population = np.random.randint(
            low=0, high=2, size=(self.population_size, self.num_genes)
        )
        return self.population

    def crossover(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        crossover_point = np.random.randint(1, x1.size)

        x1_flat = x1.ravel()
        x2_flat = x2.ravel()

        child1_flat = np.concatenate(
            [x1_flat[:crossover_point], x2_flat[crossover_point:]]
        )

        child2_flat = np.concatenate(
            [x2_flat[:crossover_point], x1_flat[crossover_point:]]
        )

        child1 = child1_flat.reshape(x1.shape)
        child2 = child2_flat.reshape(x1.shape)

        return child1, child2

    def mutate(self, x: np.ndarray) -> np.ndarray:
        for i in range(x.size):
            if np.random.rand() < self.mutation_rate:
                x.flat[i] = 1 - x.flat[i]
        return x

    def roulette_wheel(self, scores: np.ndarray) -> Tuple[int, int]:
        if np.sum(scores) == 0 or np.count_nonzero(scores) < 2:
            indices = np.arange(len(scores))
            i1, i2 = np.random.choice(indices, size=2, replace=False)
            return i1, i2

        probabilities = scores / np.sum(scores)
        flat_probs = probabilities.ravel()
        indices = np.arange(flat_probs.size)
        i1_flat, i2_flat = np.random.choice(
            indices, size=2, replace=False, p=flat_probs
        )

        i1 = np.unravel_index(i1_flat, probabilities.shape)
        i2 = np.unravel_index(i2_flat, probabilities.shape)
        return i1, i2

    def evolve(self) -> np.ndarray:
        population = self.initialize_population()
        scores = self.fitness_func(population)

        indices = np.argsort(scores)[::-1]
        scores = scores[indices]
        population = population[indices]

        best_fitness = scores[0]
        prev_best_fitness = best_fitness

        for iter in range(self.max_iter):
            i1, i2 = self.roulette_wheel(scores)

            child1, child2 = self.crossover(population[i1], population[i2])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            child1_score = self.fitness_func(child1.reshape(1, -1))[0]
            child2_score = self.fitness_func(child2.reshape(1, -1))[0]

            if child1_score > scores[-1]:
                population[-1] = child1
                scores[-1] = child1_score

            if child2_score > scores[-2]:
                population[-2] = child2
                scores[-2] = child2_score

            indices = np.argsort(scores)[::-1]
            scores = scores[indices]
            population = population[indices]

            best_fitness = scores[0]

            improvement = best_fitness - prev_best_fitness
            if improvement > 0 and improvement < self.epsilon:
                print(
                    f"Early stopping at iteration {iter + 1}: improvement ({improvement:.6f}) < epsilon ({self.epsilon})"
                )
                break

            prev_best_fitness = best_fitness

        return population[0]


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
    items_number = 30
    weights = np.random.rand(items_number)
    values = np.random.rand(items_number)
    max_capacity = 3

    print(f"Weight of every object: {weights}")
    print(f"Total weight of all objects: {np.sum(weights)}\n")
    print(f"Value of every object: {values}")
    print(f"Total value of all objects: {np.sum(values)}\n")

    fitness_func = partial(
        fitness_knapsack, weights=weights, values=values, max_capacity=max_capacity
    )

    optimizer = GA(
        fitness_func=fitness_func,
        population_size=30,
        num_genes=items_number,
        mutation_rate=0.02,
        max_iter=1000,
        epsilon=0.1,
    )

    print("Running genetic algorithm...\n")
    best_solution = optimizer.evolve()

    print(f"\nBest solution found: {best_solution}")
    best_fitness = fitness_knapsack(
        best_solution.reshape(1, -1), weights, values, max_capacity
    )[0]
    print(f"Best fitness score: {best_fitness}")

    best_weight = np.sum(best_solution * weights.ravel())
    best_value = np.sum(best_solution * values.ravel())
    print(f"Total weight: {best_weight}")
    print(f"Total value: {best_value}")
    print(f"Max capacity: {max_capacity}")


if __name__ == "__main__":
    main()
