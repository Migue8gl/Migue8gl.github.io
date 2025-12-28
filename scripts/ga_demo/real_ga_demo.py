from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class ContinuousGA:
    def __init__(
        self,
        fitness_func: Callable,
        population_size: int,
        num_genes: int,
        bounds: Tuple[float, float],
        mutation_rate: float,
        mutation_scale: float,
        max_iter: int,
        epsilon: float,
        seed: Optional[int] = None,
    ):
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.num_genes = num_genes
        self.bounds = bounds
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.max_iter = max_iter
        self.epsilon = epsilon

        if seed is not None:
            np.random.seed(seed)

        self.history = []

    def initialize_population(self) -> np.ndarray:
        self.population = np.random.uniform(
            low=self.bounds[0],
            high=self.bounds[1],
            size=(self.population_size, self.num_genes),
        )
        return self.population

    def crossover(
        self, x1: np.ndarray, x2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = np.random.rand()
        child1 = alpha * x1 + (1 - alpha) * x2
        child2 = (1 - alpha) * x1 + alpha * x2
        return child1, child2

    def mutate(self, x: np.ndarray) -> np.ndarray:
        for i in range(x.size):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.normal(0, self.mutation_scale)
                x.flat[i] += mutation
                x.flat[i] = np.clip(x.flat[i], self.bounds[0], self.bounds[1])
        return x

    def roulette_wheel(self, scores: np.ndarray) -> Tuple[int, int]:
        min_score = np.min(scores)
        adjusted_scores = scores - min_score + 1e-6

        if np.sum(adjusted_scores) == 0:
            indices = np.arange(len(adjusted_scores))
            i1, i2 = np.random.choice(indices, size=2, replace=False)
            return i1, i2

        probabilities = adjusted_scores / np.sum(adjusted_scores)
        indices = np.arange(len(probabilities))
        i1, i2 = np.random.choice(indices, size=2, replace=False, p=probabilities)
        return i1, i2

    def evolve(self) -> np.ndarray:
        population = self.initialize_population()
        scores = self.fitness_func(population)

        indices = np.argsort(scores)[::-1]
        scores = scores[indices]
        population = population[indices]

        self.history.append((population.copy(), scores.copy()))

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

            self.history.append((population.copy(), scores.copy()))

            best_fitness = scores[0]

            improvement = best_fitness - prev_best_fitness
            if improvement > 0 and improvement < self.epsilon:
                print(f"Convergencia en iteración {iter + 1}")
                break

            prev_best_fitness = best_fitness

        return population[0]


def sphere_function(population: np.ndarray) -> np.ndarray:
    x = population[:, 0]
    y = population[:, 1]
    return -(x**2 + y**2)


def rastrigin_function(population: np.ndarray) -> np.ndarray:
    x = population[:, 0]
    y = population[:, 1]
    A = 10
    return -(
        A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    )


def ackley_function(population: np.ndarray) -> np.ndarray:
    x = population[:, 0]
    y = population[:, 1]
    return -(
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )


def booth_function(population: np.ndarray) -> np.ndarray:
    x = population[:, 0]
    y = population[:, 1]
    return -((x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2)


def create_3d_animation(
    ga: ContinuousGA,
    fitness_func: Callable,
    bounds: Tuple[float, float],
    func_name: str,
    global_optimum: Tuple[float, float],
):
    resolution = 100
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([[X[i, j], Y[i, j]]])
            Z[i, j] = fitness_func(point)[0]

    plt.style.use("default")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        Z,
        color="lightgray",
        alpha=0.15,
        edgecolor="gray",
        linewidth=0.3,
        antialiased=True,
    )

    population, scores = ga.history[0]

    scatter_population = ax.scatter(
        population[:, 0],
        population[:, 1],
        scores,
        c="blue",
        s=80,
        marker="o",
        alpha=0.7,
        label="Población",
    )

    global_opt_z = fitness_func(np.array([global_optimum]).reshape(1, -1))[0]

    scatter_best = ax.scatter(
        [global_optimum[0]],
        [global_optimum[1]],
        [global_opt_z],
        c="red",
        s=300,
        marker="*",
        edgecolors="darkred",
        linewidths=2,
        label="Óptimo Global",
    )

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Fitness", fontsize=12)
    ax.set_title(f"Algoritmo Genético - {func_name}", fontsize=14, pad=15)
    ax.legend(loc="upper right", fontsize=10)

    iteration_text = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.8
        ),
        fontweight="bold",
    )

    total_frames = len(ga.history)
    rotation_speed = 120 / total_frames

    def update(frame):
        if frame < len(ga.history):
            population, scores = ga.history[frame]

            scatter_population._offsets3d = (
                population[:, 0],
                population[:, 1],
                scores,
            )

            iteration_text.set_text(
                f"Generación: {frame}\nMejor Fitness: {scores[0]:.4f}"
            )

            azim = 45 + (frame * rotation_speed)
            ax.view_init(elev=35, azim=azim)

        return scatter_population, scatter_best, iteration_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(ga.history),
        interval=200,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    return fig, anim


def run_optimization(fitness_func, bounds, func_name, output_file, global_optimum):
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZANDO: {func_name}")
    print("=" * 60)

    optimizer = ContinuousGA(
        fitness_func=fitness_func,
        population_size=60,
        num_genes=2,
        bounds=bounds,
        mutation_rate=0.05,
        mutation_scale=0.8,
        max_iter=150,
        epsilon=0.0001,
    )

    print("\nEjecutando algoritmo genético...")
    best_solution = optimizer.evolve()

    print("\nRESULTADOS:")
    print(f"Mejor solución: [{best_solution[0]:.6f}, {best_solution[1]:.6f}]")
    best_fitness = fitness_func(best_solution.reshape(1, -1))[0]
    print(f"Mejor fitness: {best_fitness:.6f}")
    print(f"Generaciones: {len(optimizer.history)}")

    print("\nCreando animación 3D...")
    fig, anim = create_3d_animation(
        optimizer, fitness_func, bounds, func_name, global_optimum
    )

    print(f"Guardando animación en {output_file}...")
    anim.save(output_file, writer="pillow", fps=7, dpi=80)
    print(f"✓ Animación guardada: {output_file}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("ALGORITMO GENÉTICO - OPTIMIZACIÓN CONTINUA")
    print("PROBANDO 4 FUNCIONES")
    print("=" * 60)

    run_optimization(
        fitness_func=sphere_function,
        bounds=(-5, 5),
        func_name="Sphere",
        output_file="scripts/ga_demo/ga_sphere.gif",
        global_optimum=(0.0, 0.0),
    )

    run_optimization(
        fitness_func=rastrigin_function,
        bounds=(-5.12, 5.12),
        func_name="Rastrigin",
        output_file="scripts/ga_demo/ga_rastrigin.gif",
        global_optimum=(0.0, 0.0),
    )

    run_optimization(
        fitness_func=ackley_function,
        bounds=(-5, 5),
        func_name="Ackley",
        output_file="scripts/ga_demo/ga_ackley.gif",
        global_optimum=(0.0, 0.0),
    )

    run_optimization(
        fitness_func=booth_function,
        bounds=(-10, 10),
        func_name="Booth",
        output_file="scripts/ga_demo/ga_booth.gif",
        global_optimum=(1.0, 3.0),
    )

    print(f"\n{'=' * 60}")
    print("✓ TODAS LAS ANIMACIONES COMPLETADAS")
    print("=" * 60)


if __name__ == "__main__":
    main()
