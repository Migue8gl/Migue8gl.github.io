import random
import time
from functools import partial
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
                break

            prev_best = best

        return population[0]


class GAVectorized:
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

    def crossover_vectorized(
        self, population: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        n_pairs = len(indices) // 2
        offspring = []

        for i in range(n_pairs):
            i1, i2 = indices[i * 2], indices[i * 2 + 1]
            p1, p2 = population[i1], population[i2]

            if np.random.rand() <= self.crossover_rate:
                point = np.random.randint(1, p1.size)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            offspring.extend([c1, c2])

        return np.array(offspring[: self.population_size])

    def mutate_vectorized(self, population: np.ndarray) -> np.ndarray:
        mask = (
            np.random.rand(population.shape[0], population.shape[1])
            < self.mutation_rate
        )
        population[mask] = 1 - population[mask]
        return population

    def optimize(self) -> np.ndarray:
        population = self.initialize_population()
        scores = self.fitness_func(population)

        idx = np.argsort(scores)[::-1]
        population = population[idx]
        scores = scores[idx]

        prev_best = scores[0]
        elite_size = int(self.elitism_rate * self.population_size)

        for it in range(self.max_iter):
            if np.sum(scores) == 0 or np.count_nonzero(scores) < 2:
                parent_indices = np.random.choice(
                    len(scores), self.population_size, replace=True
                )
            else:
                probs = scores / np.sum(scores)
                parent_indices = np.random.choice(
                    len(scores), self.population_size, replace=True, p=probs
                )

            offspring = self.crossover_vectorized(population, parent_indices)
            offspring = self.mutate_vectorized(offspring)

            elites = population[:elite_size]
            offspring = offspring[: self.population_size - elite_size]

            population = np.vstack([elites, offspring])
            scores = self.fitness_func(population)

            idx = np.argsort(scores)[::-1]
            population = population[idx]
            scores = scores[idx]

            best = scores[0]
            self.best_fitness_history.append(best)

            improvement = best - prev_best
            if 0 < improvement < self.epsilon:
                break

            prev_best = best

        return population[0]


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


def run_experiment(problem_size: int, algorithm: str, seed: int) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    weights = np.random.rand(problem_size)
    values = np.random.rand(problem_size)
    max_capacity = int(0.1 * problem_size)

    fitness_soft = partial(
        fitness_knapsack_soft, weights=weights, values=values, max_capacity=max_capacity
    )

    configs = {
        "population_size": 50,
        "num_genes": problem_size,
        "mutation_rate": 0.02,
        "max_iter": 1000,
        "epsilon": 0,
        "crossover_rate": 0.8,
        "elitism_rate": 0.2,
        "seed": seed,
    }

    start_time = time.time()

    if algorithm == "Sequential":
        optimizer = GA(fitness_func=fitness_soft, **configs)
    else:
        optimizer = GAVectorized(fitness_func=fitness_soft, **configs)

    _ = optimizer.optimize()
    elapsed_time = time.time() - start_time

    best_fitness = (
        optimizer.best_fitness_history[-1] if optimizer.best_fitness_history else 0
    )

    return {
        "problem_size": problem_size,
        "algorithm": algorithm,
        "time": elapsed_time,
        "fitness": best_fitness,
        "seed": seed,
    }


def main():
    problem_sizes = [50, 100, 200, 400, 800, 1600]
    n_experiments = 5
    results = []

    print("Comparing Sequential vs Vectorized GA implementations")
    print("=" * 80)

    for problem_size in problem_sizes:
        print(f"\nProblem size: {problem_size} items")
        print("-" * 80)

        for seed in range(n_experiments):
            for algorithm in ["Sequential", "Vectorized"]:
                result = run_experiment(problem_size, algorithm, seed)
                results.append(result)
                print(
                    f"  {algorithm:12s} | Seed {seed} | Time: {result['time']:.4f}s | Fitness: {result['fitness']:.4f}"
                )

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = (
        df.groupby(["problem_size", "algorithm"])
        .agg({"time": ["mean", "std"], "fitness": ["mean", "std"]})
        .round(4)
    )
    print(summary)

    speedup = (
        df.groupby(["problem_size", "seed"], group_keys=False)
        .apply(
            lambda x: x[x["algorithm"] == "Sequential"]["time"].values[0]
            / x[x["algorithm"] == "Vectorized"]["time"].values[0],
            include_groups=False,
        )
        .reset_index(name="speedup")
    )
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS (Vectorized vs Sequential)")
    print("=" * 80)
    speedup_summary = (
        speedup.groupby("problem_size")["speedup"].agg(["mean", "std"]).round(4)
    )
    print(speedup_summary)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(
        data=df,
        x="problem_size",
        y="time",
        hue="algorithm",
        ax=axes[0, 0],
        errorbar="sd",
    )
    axes[0, 0].set_title(
        "Execution Time by Problem Size", fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Problem Size (number of items)")
    axes[0, 0].set_ylabel("Time (seconds)")

    sns.barplot(
        data=df,
        x="problem_size",
        y="fitness",
        hue="algorithm",
        ax=axes[0, 1],
        errorbar="sd",
    )
    axes[0, 1].set_title("Fitness by Problem Size", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Problem Size (number of items)")
    axes[0, 1].set_ylabel("Fitness Value")

    speedup_mean = speedup.groupby("problem_size")["speedup"].mean().reset_index()
    sns.barplot(
        data=speedup_mean,
        x="problem_size",
        y="speedup",
        ax=axes[1, 0],
        color="steelblue",
    )
    axes[1, 0].axhline(y=1, color="red", linestyle="--", label="No speedup")
    axes[1, 0].set_title(
        "Average Speedup (Sequential / Vectorized)", fontsize=14, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Problem Size (number of items)")
    axes[1, 0].set_ylabel("Speedup Factor")
    axes[1, 0].legend()

    sns.lineplot(
        data=df, x="problem_size", y="time", hue="algorithm", marker="o", ax=axes[1, 1]
    )
    axes[1, 1].set_title(
        "Time Scaling with Problem Size", fontsize=14, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Problem Size (number of items)")
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("scripts/ga_demo/ga_comparison.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
