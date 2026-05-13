import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):

    cm = np.zeros((n_classes, n_classes), dtype=np.int32)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def precision_recall_f1(cm):

    n_classes = cm.shape[0]

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    for i in range(n_classes):

        tp = cm[i, i]

        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp + 1e-8)

        recall[i] = tp / (tp + fn + 1e-8)

        f1[i] = (
            2 * precision[i] * recall[i]
            / (precision[i] + recall[i] + 1e-8)
        )

    return precision, recall, f1


def macro_f1(f1_scores):

    return np.mean(f1_scores)


def weighted_f1(f1_scores, y_true, n_classes):

    weights = np.zeros(n_classes)

    for i in range(n_classes):
        weights[i] = np.sum(y_true == i)

    weights /= np.sum(weights)

    return np.sum(weights * f1_scores)


def classification_report(
    y_true,
    y_pred,
    class_names
):

    n_classes = len(class_names)

    cm = confusion_matrix(
        y_true,
        y_pred,
        n_classes
    )

    precision, recall, f1 = precision_recall_f1(cm)

    macro = macro_f1(f1)

    weighted = weighted_f1(
        f1,
        y_true,
        n_classes
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        f"{'Class':<15}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 60)

    for i, name in enumerate(class_names):

        print(
            f"{name:<15}"
            f"{precision[i]:<12.4f}"
            f"{recall[i]:<12.4f}"
            f"{f1[i]:<12.4f}"
        )

    print("-" * 60)

    print(f"Macro F1:    {macro:.4f}")
    print(f"Weighted F1: {weighted:.4f}")

    print("=" * 60)

    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro,
        "weighted_f1": weighted
    }