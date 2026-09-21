import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        noise=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        output_directory="./scripts/symbolic_regression/outputs",
        model_selection="best",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7,
        label="Predicciones",
    )

    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Predicción perfecta",
    )

    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(f"Regresión simbólica — MSE = {mse:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("./scripts/symbolic_regression/x.png")

    print("Best equation:")
    print(model.sympy())


if __name__ == "__main__":
    main()
