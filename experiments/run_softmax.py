import numpy as np

from dataset.loader import load_dataset
from dataset.loader import CLASSES
from dataset.split import stratified_split
from evaluation.plots import plot_confusion_matrix
from evaluation.plots import plot_training_curves
from experiments.feature_utils import add_quadratic_features
from experiments.feature_utils import enhanced_fer_features
from experiments.feature_utils import pca_from_train
from experiments.feature_utils import standardize_from_train
from models.cnn.optim import Adam
from models.softmax import SoftmaxRegression
from utils.earlystopping import EarlyStopping


ANNOTATIONS_FILE = "data/annotations.csv"
PCA_COMPONENTS = 128
PLOTS_DIR = "logs/plots"


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

    X_train, X_val, X_test = pca_from_train(
        X_train,
        X_val,
        X_test,
        PCA_COMPONENTS
    )

    X_train, X_val, X_test = standardize_from_train(
        X_train,
        X_val,
        X_test
    )

    X_train, X_val, X_test = add_quadratic_features(
        X_train,
        X_val,
        X_test
    )

    X_train, X_val, X_test = standardize_from_train(
        X_train,
        X_val,
        X_test
    )

    model = SoftmaxRegression(
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y)),
        lr=1e-3,
        lambda_l2=1e-3,
        batch_size=128,
        max_epochs=120
    )

    early = EarlyStopping(patience=25)

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        optimizer=Adam(lr=model.lr),
        early_stopping=early,
        early_stopping_metric="val_acc",
        grad_clip_norm=10.0
    )

    plot_training_curves(
        model.history,
        title="Softmax Regression Training Curves",
        save_path=f"{PLOTS_DIR}/softmax_training_curves.png"
    )

    y_pred = model.predict(X_test)
    test_acc = float((y_pred == y_test).mean())

    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=CLASSES,
        title="Softmax Regression Confusion Matrix",
        save_path=f"{PLOTS_DIR}/softmax_confusion_matrix.png"
    )

    print(f"\nBest Validation Accuracy: {max(model.history['val_acc']):.4f}")
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
