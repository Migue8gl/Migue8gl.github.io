import time
from itertools import permutations
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TSP_GRASP:
    def __init__(self, max_iterations: int = 20, alpha: float = 0.2):
        self.max_iterations = max_iterations
        self.alpha = alpha

    def build_solution(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Construct a feasible TSP solution using the GRASP construction phase.

        Starting from city 0, the solution is built incrementally by selecting
        the next city from a Restricted Candidate List (RCL). The RCL contains
        the best candidates according to the greedy criterion, and one of them
        is chosen uniformly at random.

        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix.

        Returns:
            np.ndarray: A feasible TSP tour represented as an ordered array of
            city indices.
        """

        n = distance_matrix.shape[1]

        unvisited = np.array([i for i in range(1, n)])
        candidates = []
        solution = np.zeros(1, dtype=np.int64)

        for _ in range(n - 1):
            for city in unvisited:
                cost = distance_matrix[solution[-1], city]

                candidates.append((city, cost))

            costs = [cost for _, cost in candidates]
            min_cost = min(costs)
            max_cost = max(costs)

            threshold = min_cost + self.alpha * (max_cost - min_cost)
            rcl = [city for city, cost in candidates if cost <= threshold]
            next_city = np.random.choice(rcl)
            solution = np.append(solution, next_city)

            unvisited = np.setdiff1d(unvisited, [next_city])
            candidates = []
        return np.array(solution)

    def local_search(
        self, solution: np.ndarray, distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Improve a TSP solution using a swap-based local search.

        The algorithm iteratively swaps pairs of cities and accepts the first
        improving move. The process stops when no improving swap is found.

        Args:
            solution (np.ndarray): Initial TSP tour.
            distance_matrix (np.ndarray): Pairwise distance matrix.

        Returns:
            np.ndarray: Locally improved TSP tour.
        """

        best_solution = solution.copy()
        best_cost = tsp_cost(best_solution, distance_matrix)

        improved = True

        while improved:
            improved = False

            for i in range(1, len(solution) - 1):
                for j in range(i + 1, len(solution)):
                    candidate = best_solution.copy()
                    candidate[i], candidate[j] = candidate[j], candidate[i]

                    candidate_cost = tsp_cost(candidate, distance_matrix)

                    if candidate_cost < best_cost:
                        best_solution = candidate
                        best_cost = candidate_cost
                        improved = True
                        break

                if improved:
                    break

        return best_solution

    def optimize(self, distance_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        best_solution = None
        best_cost = np.inf

        for _ in range(self.max_iterations):
            solution = self.build_solution(distance_matrix)

            solution = self.local_search(solution, distance_matrix)

            cost = tsp_cost(solution, distance_matrix)

            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        return best_solution, best_cost


def tsp_cost(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Compute the total cost of a TSP tour.

    The tour is treated as a closed cycle, including the return from the
    last city to the starting city.

    Args:
        tour (np.ndarray): Ordered array of city indices.
        distance_matrix (np.ndarray): Pairwise distance matrix.

    Returns:
        float: Total cost of the tour.
    """

    cost = 0.0

    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]

    cost += distance_matrix[tour[-1], tour[0]]

    return cost


def solve_tsp_brute_force(distance_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the Traveling Salesman Problem (TSP)

    Args:
        distance_matrix (np.ndarray): A square symmetric matrix of shape
            (n, n), where distance_matrix[i, j] represents the distance
            between cities i and j.

    Returns:
        tuple[np.ndarray, float]:
            - The optimal tour as a NumPy array of city indices, starting
              at city 0.
            - The total length (cost) of the optimal tour, including the
              return from the last city back to the starting city.
    """
    n = distance_matrix.shape[1]
    best_tour = None
    best_cost = np.inf

    for rest in permutations(range(1, n)):
        tour = (0,) + rest
        tour_array = np.array(tour)
        tour_cost = tsp_cost(tour_array, distance_matrix)

        if best_cost > tour_cost:
            best_cost = tour_cost
            best_tour = tour_array

    return best_tour, best_cost


def solve_tsp_greedy(distance_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the TSP using a simple greedy nearest-neighbor heuristic.

    Starting from city 0, the algorithm always selects the closest
    unvisited city. Unlike GRASP, there is no randomness or local search.

    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix.

    Returns:
        tuple[np.ndarray, float]:
            - The constructed TSP tour.
            - The total cost of the tour.
    """

    n = distance_matrix.shape[0]

    tour = [0]
    unvisited = set(range(1, n))

    while unvisited:
        current_city = tour[-1]

        next_city = min(
            unvisited,
            key=lambda city: distance_matrix[current_city, city]
        )

        tour.append(next_city)
        unvisited.remove(next_city)

    tour = np.array(tour, dtype=np.int64)
    cost = tsp_cost(tour, distance_matrix)

    return tour, cost


def solve_tsp(
    distance_matrix: np.ndarray, version: Literal["v1", "v2"] = "v2"
) -> tuple[np.ndarray, float]:
    """Router to different implementations of TSP solvers.

    Args:
        distance_matrix (np.ndarray): A square symmetric matrix of shape
            (n, n), where distance_matrix[i, j] represents the distance
            between cities i and j.

    Returns:
        tuple[np.ndarray, float]:
            - The optimal tour as a NumPy array of city indices, starting
              at city 0.
            - The total length (cost) of the optimal tour, including the
              return from the last city back to the starting city.
    """

    if version == "v1":
        return solve_tsp_brute_force(distance_matrix)
    else:
        return solve_tsp_greedy(distance_matrix)



def create_cities(num_cities: int) -> np.ndarray:
    """Create random coordinates representing a set of cities.

    Args:
        num_cities (int): Number of cities.

    Returns:
        np.ndarray: A matrix containing the coordinates of each city.
    """

    return np.random.rand(num_cities, 2)


def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Create a symmetric distance matrix from a set of city coordinates.

    Args:
        cities (np.ndarray): Matrix containing the coordinates of each city.

    Returns:
        np.ndarray: A symmetric matrix containing the distances between cities.
    """

    diff = cities[:, None, :] - cities[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def visualize_cities(cities: np.ndarray) -> None:
    """Visualize a set of interconnected cities in a 2D plane.

    Args:
        cities (np.ndarray): Matrix containing the coordinates of each city.
    """

    plt.figure(figsize=(6, 6))

    n = len(cities)

    for i in range(n):
        for j in range(i + 1, n):
            plt.plot(
                [cities[i, 0], cities[j, 0]],
                [cities[i, 1], cities[j, 1]],
                color="lightgray",
                linewidth=0.8,
                zorder=1,
            )

    plt.scatter(cities[:, 0], cities[:, 1], color="red", zorder=2)

    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i))

    plt.axis("equal")
    plt.savefig("scripts/grasp/cities.png")


def visualize_tour(cities: np.ndarray, tour: np.ndarray, best_cost: float) -> None:
    """Visualize the optimal TSP tour.

    Args:
        cities (np.ndarray): Array of city coordinates with shape (n, 2).
        tour (np.ndarray): Optimal tour as an ordered array of city indices.
        best_cost (float): Total cost of the optimal tour.
    """

    plt.figure(figsize=(6, 6))

    n = len(cities)

    for i in range(n):
        for j in range(i + 1, n):
            plt.plot(
                [cities[i, 0], cities[j, 0]],
                [cities[i, 1], cities[j, 1]],
                color="lightgray",
                linewidth=0.8,
                zorder=1,
            )

    for i in range(len(tour) - 1):
        a = cities[tour[i]]
        b = cities[tour[i + 1]]
        plt.plot([a[0], b[0]], [a[1], b[1]], "b-", linewidth=2)

    a = cities[tour[-1]]
    b = cities[tour[0]]
    plt.plot([a[0], b[0]], [a[1], b[1]], "b-", linewidth=2)

    plt.scatter(cities[:, 0], cities[:, 1], color="red", zorder=2)

    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i))

    plt.title(f"TSP Tour (Cost = {best_cost:.3f}, N = {n})")
    plt.axis("equal")
    plt.savefig("scripts/grasp/best_tour.png")


if __name__ == "__main__":
    n_cities = 12
    cities = create_cities(n_cities)
    distance_matrix = create_distance_matrix(cities)

    visualize_cities(cities)

    results = pd.DataFrame(
        columns=[
            "algorithm",
            "n_cities",
            "execution_time_s",
            "tour_cost",
        ]
    )

    start = time.perf_counter()
    best_tour, best_cost = solve_tsp(distance_matrix, "v1")
    end = time.perf_counter()

    results.loc[len(results)] = {
        "algorithm": "Brute Force",
        "n_cities": distance_matrix.shape[0],
        "execution_time_s": f"{end - start:6f}",
        "tour_cost": best_cost,
    }
    
    start = time.perf_counter()
    best_tour, best_cost = solve_tsp(distance_matrix, "v2")
    end = time.perf_counter()

    results.loc[len(results)] = {
        "algorithm": "Greedy",
        "n_cities": distance_matrix.shape[0],
        "execution_time_s": f"{end - start:6f}",
        "tour_cost": best_cost,
    }

    grasp = TSP_GRASP()

    start = time.perf_counter()
    grasp_solution, best_cost = grasp.optimize(distance_matrix)
    end = time.perf_counter()

    results.loc[len(results)] = {
        "algorithm": "GRASP",
        "n_cities": distance_matrix.shape[0],
        "execution_time_s": f"{end - start:6f}",
        "tour_cost": best_cost,
    }

    results.to_csv("scripts/grasp/result.csv")

    visualize_tour(cities, best_tour, best_cost)
