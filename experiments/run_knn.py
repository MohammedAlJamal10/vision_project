from dataset.loader import load_dataset
from dataset.loader import CLASSES
from dataset.split import stratified_split
from evaluation.plots import plot_confusion_matrix
from evaluation.plots import plot_k_sweep
from experiments.feature_utils import enhanced_fer_features
from experiments.feature_utils import fit_pca
from experiments.feature_utils import pca_from_train
from experiments.feature_utils import standardize_from_train
from experiments.feature_utils import transform_pca
from models.knn import KNNClassifier, sweep_k


ANNOTATIONS_FILE = "data/annotations.csv"
PCA_COMPONENTS = [64, 96, 128, 192, 256]
PLOTS_DIR = "logs/plots"


def choose_best_pca_and_k(X_train, y_train, X_val, y_val):

    pca_mean, pca_components = fit_pca(
        X_train,
        max(PCA_COMPONENTS)
    )

    best = {
        "val_acc": -1.0,
        "n_components": None,
        "k": None,
    }

    for n_components in PCA_COMPONENTS:

        components = pca_components[:, :n_components]
        X_train_pca = transform_pca(
            X_train,
            pca_mean,
            components
        )
        X_val_pca = transform_pca(
            X_val,
            pca_mean,
            components
        )

        X_train_pca, X_val_pca, _ = standardize_from_train(
            X_train_pca,
            X_val_pca,
            X_val_pca
        )

        best_k, results = sweep_k(
            X_train_pca,
            y_train,
            X_val_pca,
            y_val,
            k_values=[3, 5, 7, 9, 11, 15, 21, 31, 41, 61],
            weighted=True
        )

        val_acc = results[best_k]

        print(
            f"PCA={n_components:3d} | "
            f"best_k={best_k:2d} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best["val_acc"]:
            best = {
                "val_acc": val_acc,
                "n_components": n_components,
                "k": best_k,
                "k_results": results,
            }

    return best


def main():

    X, y = load_dataset(ANNOTATIONS_FILE)
    features = enhanced_fer_features(X, verbose=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(
        features,
        y
    )

    X_train, X_val, X_test = standardize_from_train(
        X_train,
        X_val,
        X_test
    )

    best = choose_best_pca_and_k(
        X_train,
        y_train,
        X_val,
        y_val
    )

    plot_k_sweep(
        best["k_results"],
        title=f"KNN Validation Accuracy vs K (PCA={best['n_components']})",
        save_path=f"{PLOTS_DIR}/knn_k_sweep.png"
    )

    X_train, X_val, X_test = pca_from_train(
        X_train,
        X_val,
        X_test,
        best["n_components"]
    )

    X_train, X_val, X_test = standardize_from_train(
        X_train,
        X_val,
        X_test
    )

    model = KNNClassifier(
        k=best["k"],
        weighted=True
    ).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_acc = float((y_pred == y_test).mean())

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=CLASSES,
        title="KNN Confusion Matrix",
        save_path=f"{PLOTS_DIR}/knn_confusion_matrix.png"
    )

    print(
        f"\nSelected PCA={best['n_components']} | "
        f"k={best['k']} | "
        f"Val Accuracy: {best['val_acc']:.4f}"
    )
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
