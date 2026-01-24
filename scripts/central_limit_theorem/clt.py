import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    n_samples = 10000  
    sample_size = 600

    sample_means = [
        np.mean(np.random.randint(0, 10, sample_size))
        for _ in range(n_samples)
    ]

    sns.histplot(sample_means, bins=50, kde=True)
    plt.title("Central Limit Theorem: Distribution of Sample Means")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.savefig(f"scripts/central_limit_theorem/clt_{sample_size}.png")
    plt.close()

if __name__ == "__main__":
    main()
