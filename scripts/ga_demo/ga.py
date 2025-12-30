import random
from functools import partial
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Greedy:
    def __init__(
        self,
        weights: np.ndarray,
        values: np.ndarray,
        max_capacity: float,
        heuristic: str,
        fitness_func: Callable,
    ):
        self.weights = weights
        self.values = values
        self.max_capacity = max_capacity
        self.fitness_func = fitness_func
        self.heuristic = heuristic
        self.best_fitness_history = []

    def optimize(self) -> np.ndarray:
        n = len(self.weights)
        solution = np.zeros(n, dtype=int)

        if self.heuristic == "h1":
            ratios = self.values  
            order = np.argsort(-ratios)
        elif self.heuristic == "h2":
            ratios = self.weights  
            order = np.argsort(ratios)  
        else:
            ratios = self.values / self.weights 
            order = np.argsort(-ratios)

        remaining_capacity = self.max_capacity
        best_fitness = float("-inf")

        for i in order:
            if self.weights[i] <= remaining_capacity:
                solution[i] = 1
                remaining_capacity -= self.weights[i]

                current_fitness = self.fitness_func(solution.reshape(1, -1))[0]
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    self.best_fitness_history.append(best_fitness)

        if not self.best_fitness_history:
            self.best_fitness_history.append(
                self.fitness_func(solution.reshape(1, -1))[0]
            )

        return solution


class GA:
    def __init__(
        self,
        fitness_func: Callable,
        population_size: int,
        num_genes: int,
        mutation_rate: float,
        max_iter: int,
        epsilon: float,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.num_genes = num_genes
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.best_fitness_history = []
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
        if np.random.rand() > self.crossover_rate:
            return x1.copy(), x2.copy()
        point = np.random.randint(1, x1.size)
        c1 = np.concatenate([x1[:point], x2[point:]])
        c2 = np.concatenate([x2[:point], x1[point:]])
        return c1, c2

    def mutate(self, x: np.ndarray) -> np.ndarray:
        for i in range(x.size):
            if np.random.rand() < self.mutation_rate:
                x[i] = 1 - x[i]
        return x

    def roulette_wheel(self, scores: np.ndarray) -> np.ndarray:
        if np.sum(scores) == 0 or np.count_nonzero(scores) < 2:
            return np.random.choice(len(scores), 2, replace=False)
        probs = scores / np.sum(scores)
        return np.random.choice(len(scores), 2, replace=False, p=probs)

    def optimize(self) -> np.ndarray:
        population = self.initialize_population()
        scores = self.fitness_func(population)

        idx = np.argsort(scores)[::-1]
        population = population[idx]
        scores = scores[idx]

        prev_best = scores[0]
        elite_size = int(self.elitism_rate * self.population_size)

        for it in range(self.max_iter):
            new_population = []

            elites = population[:elite_size]
            new_population.extend(elites)

            while len(new_population) < self.population_size:
                i1, i2 = self.roulette_wheel(scores)
                p1, p2 = population[i1], population[i2]
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)

            population = np.array(new_population)
            scores = self.fitness_func(population)

            idx = np.argsort(scores)[::-1]
            population = population[idx]
            scores = scores[idx]

            best = scores[0]
            self.best_fitness_history.append(best)

            improvement = best - prev_best
            if 0 < improvement < self.epsilon:
                print(
                    f"Early stopping at iteration {it + 1}: improvement ({improvement:.6f}) < epsilon ({self.epsilon})"
                )
                break

            prev_best = best

        return population[0]


def fitness_knapsack_hard(
    population: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    max_capacity: int,
) -> np.ndarray:
    total_weights = np.sum(population * weights, axis=1)
    total_values = np.sum(population * values, axis=1)
    total_values[total_weights > max_capacity] = 0.0

    return total_values


def fitness_knapsack_soft(
    population: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    max_capacity: float,
    alpha: float = 0.2,
) -> np.ndarray:
    total_weights = np.sum(population * weights, axis=1)
    total_values = np.sum(population * values, axis=1)

    excess = np.maximum(0.0, total_weights - max_capacity)
    penalty = np.exp(-alpha * excess)

    return total_values * penalty


def plot_fitness_comparison(
    histories: dict[str, List[float]],
    title: str = "Fitness Comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Best Fitness",
    figsize: tuple = (8, 5),
    save_path: str | None = "scripts/ga_demo/fitness_comparison.png",
):
    sns.set_theme(style="whitegrid")

    min_len = max(len(h) for h in histories.values())

    data = []
    for label, history in histories.items():
        for i, v in enumerate(history[:min_len]):
            data.append(
                {
                    "Iteration": i,
                    "Best Fitness": v,
                    "Method": label,
                }
            )

    df = pd.DataFrame(data)

    plt.figure(figsize=figsize)

    sns.lineplot(
        data=df,
        x="Iteration",
        y="Best Fitness",
        hue="Method",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()


def main():
    items_number = 200
    weights = np.random.rand(items_number)
    values = np.random.rand(items_number)
    max_capacity = int(0.1 * items_number)

    print(f"Weight of every object: {weights}")
    print(f"Total weight of all objects: {np.sum(weights)}\n")
    print(f"Value of every object: {values}")
    print(f"Total value of all objects: {np.sum(values)}\n")

    fitness_soft = partial(
        fitness_knapsack_soft, weights=weights, values=values, max_capacity=max_capacity
    )

    fitness_hard = partial(
        fitness_knapsack_hard, weights=weights, values=values, max_capacity=max_capacity
    )

    configs = {
        "population_size": 30,
        "num_genes": items_number,
        "mutation_rate": 0.02,
        "max_iter": 1000,
        "epsilon": 0,
        "crossover_rate": 0.8,
        "elitism_rate": 0.2,
    }

    optimizer_soft = GA(fitness_func=fitness_soft, **configs)

    optimizer_hard = GA(fitness_func=fitness_hard, **configs)

    print("Running genetic algorithm...\n")
    best_solution_soft = optimizer_soft.optimize()
    best_solution_hard = optimizer_hard.optimize()

    print("\n=== SOFT CONSTRAINTS ===")

    best_fitness_soft = fitness_knapsack_soft(
        best_solution_soft.reshape(1, -1),
        weights,
        values,
        max_capacity,
    )[0]

    best_weight_soft = np.sum(best_solution_soft * weights.ravel())
    best_value_soft = np.sum(best_solution_soft * values.ravel())

    print(f"Best fitness score: {best_fitness_soft}")
    print(f"Total weight: {best_weight_soft}")
    print(f"Total value: {best_value_soft}")
    print(f"Max capacity: {max_capacity}")

    print("\n=== HARD CONSTRAINTS ===")

    best_fitness_hard = fitness_knapsack_hard(
        best_solution_hard.reshape(1, -1),
        weights,
        values,
        max_capacity,
    )[0]

    best_weight_hard = np.sum(best_solution_hard * weights.ravel())
    best_value_hard = np.sum(best_solution_hard * values.ravel())

    print(f"Best fitness score: {best_fitness_hard}")
    print(f"Total weight: {best_weight_hard}")
    print(f"Total value: {best_value_hard}")
    print(f"Max capacity: {max_capacity}")

    greedy = Greedy(
        weights=weights,
        values=values,
        max_capacity=max_capacity,
        heuristic="h3",
        fitness_func=fitness_hard,
    )

    best_solution_greedy = greedy.optimize()

    print("\n=== GREEDY ===")

    best_fitness_greedy = fitness_knapsack_hard(
        best_solution_greedy.reshape(1, -1),
        weights,
        values,
        max_capacity,
    )[0]

    best_weight_greedy = np.sum(best_solution_greedy * weights)
    best_value_greedy = np.sum(best_solution_greedy * values)

    print(f"Best fitness score: {best_fitness_greedy}")
    print(f"Total weight: {best_weight_greedy}")
    print(f"Total value: {best_value_greedy}")
    print(f"Max capacity: {max_capacity}")

    plot_fitness_comparison(
        {
            "GA Soft": optimizer_soft.best_fitness_history,
            "GA Hard": optimizer_hard.best_fitness_history,
            "Greedy": greedy.best_fitness_history,
        }
    )


if __name__ == "__main__":
    main()
