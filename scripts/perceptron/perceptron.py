import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_data(n_samples: int, n_features: int, W_true: np.ndarray, b_true: float):
    while True:
        X = np.random.randn(n_samples, n_features)
        y = np.sign(X @ W_true + b_true)
        y = y.flatten()
        if len(np.unique(y)) > 1:
            break
    return X, y


def visualize_data(
    X: np.ndarray, y: np.ndarray, title: str, W: np.ndarray = None, show_weights_vector: bool = False
) -> None:
    plt.figure()
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette="coolwarm",
    )

    if W is not None:
        b, w1, w2 = W

        x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
        y_vals = -(w1 * x_vals + b) / w2
        plt.plot(x_vals, y_vals, color="green")

        if show_weights_vector:
            x0 = np.mean(x_vals)
            y0 = -(w1 * x0 + b) / w2

            k = 1
            x1 = x0 + k*w1
            y1 = y0 + k*w2

            plt.plot([x0, x1], [y0, y1], color="purple", linestyle="--", linewidth=2)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(title)


def perceptron(X: np.ndarray, y: np.ndarray):
    W_bias = np.random.random(X.shape[1] + 1)
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

    W_new = W_bias.copy()

    while True:
        y_hat = X_bias @ W_new
        c = np.sign(y_hat)

        all_correct = True
        for i in range(len(y)):
            if y[i] != c[i]:
                index = i
                all_correct = False
                break
        if all_correct:
            break
        W_new = W_new + y[index] * X_bias[index]

    return W_new


def main():
    n_samples = 30
    n_features = 2
    W_true = np.random.random(2)
    b_true = np.random.random()

    X, y = create_data(n_samples, n_features, W_true, b_true)
    visualize_data(X, y, title="./scripts/perceptron/linear_dataset.png")

    W = perceptron(X, y)
    visualize_data(X, y, W=W, title="./scripts/perceptron/linear_dataset_solved.png", show_weights_vector=True)


if __name__ == "__main__":
    main()
