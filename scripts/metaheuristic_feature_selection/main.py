import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mealpy import GA, BinaryVar
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def load_sklearn(loader):
    data = loader()
    return data.data, data.target


def load_toxicity():
    toxicity = fetch_ucirepo(id=728)

    X = toxicity.data.features.to_numpy()
    y = toxicity.data.targets.squeeze().to_numpy()

    return X, y


DATASETS = {
    "Toxicity": load_toxicity,
    "Digits": lambda: load_sklearn(load_digits),
    "Wine": lambda: load_sklearn(load_wine),
    "Breast Cancer": lambda: load_sklearn(load_breast_cancer),
}


ALPHA = 0.1
RANDOM_STATE = 42


def accuracy(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return accuracy_score(
        y_test,
        model.predict(X_test),
    )


def run_experiment(name, loader):
    X, y = loader()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    def objective(solution):
        selected = solution > 0.5

        if not selected.any():
            return 1.0

        model = RandomForestClassifier(
            n_estimators=30,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

        model.fit(
            X_train[:, selected],
            y_train,
        )

        accuracy = accuracy_score(
            y_val,
            model.predict(X_val[:, selected]),
        )

        return ALPHA * selected.mean() + (1 - ALPHA) * (1 - accuracy)

    problem = {
        "bounds": BinaryVar(n_vars=X.shape[1]),
        "minmax": "min",
        "obj_func": objective,
    }

    optimizer = GA.EliteSingleGA(
        epoch=20,
        pop_size=50,
    )

    best = optimizer.solve(
        problem,
        seed=RANDOM_STATE,
    )

    selected = best.solution > 0.5
    k = selected.sum()

    ga_accuracy = accuracy(
        X_train[:, selected],
        X_test[:, selected],
        y_train,
        y_test,
    )

    all_accuracy = accuracy(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    kbest = SelectKBest(
        f_classif,
        k=k,
    )

    X_train_kbest = kbest.fit_transform(
        X_train,
        y_train,
    )

    X_test_kbest = kbest.transform(X_test)

    kbest_accuracy = accuracy(
        X_train_kbest,
        X_test_kbest,
        y_train,
        y_test,
    )

    return {
        "Dataset": name,
        "Features": X.shape[1],
        "GA Features": k,
        "GA %": 100 * k / X.shape[1],
        "All": all_accuracy,
        "GA": ga_accuracy,
        "SelectKBest": kbest_accuracy,
        "History": optimizer.history.list_global_best_fit,
    }


def plot_results(results):
    df = pd.DataFrame(results)

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width,
        df["All"],
        width,
        label="All Features",
    )

    plt.bar(
        x,
        df["GA"],
        width,
        label="GA",
    )

    plt.bar(
        x + width,
        df["SelectKBest"],
        width,
        label="SelectKBest",
    )

    plt.xticks(x, df["Dataset"])
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Classification Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "./scripts/metaheuristic_feature_selection/accuracy_comparison.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    plt.bar(
        df["Dataset"],
        df["GA %"],
    )

    plt.ylabel("Selected Features (%)")
    plt.ylim(0, 100)
    plt.title("Features Selected by GA")
    plt.tight_layout()
    plt.savefig(
        "./scripts/metaheuristic_feature_selection/selected_features.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 5))

    for result in results:
        plt.plot(
            result["History"],
            label=result["Dataset"],
        )

    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("GA Optimization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "./scripts/metaheuristic_feature_selection/ga_evolution.png",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    results = [run_experiment(name, loader) for name, loader in DATASETS.items()]

    df = pd.DataFrame(results)

    print("\nRESULTS")
    print("=" * 70)

    print(
        df.drop(columns="History").to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    df.drop(columns="History").to_csv(
        "./scripts/metaheuristic_feature_selection/feature_selection_results.csv",
        index=False,
    )

    plot_results(results)
