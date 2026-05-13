from evaluation.plots import plot_model_comparison


PLOTS_DIR = "logs/plots"


MODEL_TEST_ACCURACIES = {
    "KNN": 0.0,
    "Softmax": 0.0,
    "CNN": 0.0,
    "MobileNetV2": 0.0,
}


def main():
    model_names = list(MODEL_TEST_ACCURACIES.keys())
    accuracies = list(MODEL_TEST_ACCURACIES.values())

    plot_model_comparison(
        model_names,
        accuracies,
        title="Model Test Accuracy Comparison",
        save_path=f"{PLOTS_DIR}/model_comparison.png"
    )

    print(f"Saved model comparison plot to {PLOTS_DIR}/model_comparison.png")


if __name__ == "__main__":
    main()
