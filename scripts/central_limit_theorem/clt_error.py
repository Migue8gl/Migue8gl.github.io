import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    np.random.seed(42)
    n_samples = 1000
    sample_sizes = [1, 2, 5, 15, 30]

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(sample_sizes))

    for i, sample_size in enumerate(sample_sizes):
        sample_means = [
            np.mean(np.random.randint(0, 10, sample_size)) for _ in range(n_samples)
        ]

        sem = np.std(sample_means, ddof=1)
        mean_of_means = np.mean(sample_means)
        label = f"n={sample_size} (SEM={sem:.3f})"

        sns.kdeplot(
            sample_means,
            color=colors[i],
            label=label,
            linewidth=2,
            fill=True,
            alpha=0.2,
        )

        plt.axvline(mean_of_means, color=colors[i], linestyle="--", alpha=0.5)

    plt.title("Central Limit Theorem: Distribution of Sample Means", fontsize=14)
    plt.xlabel("Sample Mean", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Sample size", loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig("scripts/central_limit_theorem/clt_densities_overlay.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
