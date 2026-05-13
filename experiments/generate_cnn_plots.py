import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

PLOT_DIR = "logs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# MANUAL TRAINING HISTORY
# =========================================================

epochs = [1, 5, 10, 15, 20, 25, 30, 35]

train_loss = [
    1.7850,
    1.2737,
    1.0394,
    0.9313,
    0.8183,
    0.7371,
    0.6722,
    0.6030
]

val_loss = [
    1.6503,
    1.3049,
    1.1649,
    1.0847,
    1.1294,
    1.0229,
    1.0588,
    1.1272
]

train_acc = [
    0.3420,
    0.5515,
    0.6491,
    0.6860,
    0.7183,
    0.7713,
    0.7811,
    0.7853
]

val_acc = [
    0.3068,
    0.4960,
    0.5598,
    0.6076,
    0.6096,
    0.6355,
    0.6195,
    0.6215
]

final_test_acc = 0.6419

# =========================================================
# ACCURACY CURVE
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_acc, marker='o', label="Train Accuracy")
plt.plot(epochs, val_acc, marker='o', label="Validation Accuracy")

plt.title("SimpleCNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()

acc_path = os.path.join(PLOT_DIR, "cnn_accuracy.png")
plt.savefig(acc_path)

plt.close()

# =========================================================
# LOSS CURVE
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_loss, marker='o', label="Train Loss")
plt.plot(epochs, val_loss, marker='o', label="Validation Loss")

plt.title("SimpleCNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()

loss_path = os.path.join(PLOT_DIR, "cnn_loss.png")
plt.savefig(loss_path)

plt.close()

# =========================================================
# FINAL TEST ACCURACY BAR
# =========================================================

plt.figure(figsize=(5, 5))

plt.bar(["SimpleCNN"], [final_test_acc])

plt.ylim(0, 1)

plt.title("SimpleCNN Final Test Accuracy")
plt.ylabel("Accuracy")

plt.text(
    0,
    final_test_acc + 0.02,
    f"{final_test_acc*100:.2f}%",
    ha='center'
)

plt.tight_layout()

bar_path = os.path.join(PLOT_DIR, "cnn_test_accuracy.png")
plt.savefig(bar_path)

plt.close()

# =========================================================
# OPTIONAL PLACEHOLDER CONFUSION MATRIX
# =========================================================
# Replace with real y_true / y_pred later if available.

"""
Example:

y_true = [...]
y_pred = [...]

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "angry",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]
)

disp.plot(cmap="Blues")

plt.title("SimpleCNN Confusion Matrix")

cm_path = os.path.join(PLOT_DIR, "cnn_confusion_matrix.png")
plt.savefig(cm_path)

plt.close()
"""

# =========================================================
# DONE
# =========================================================

print("=" * 50)
print("Plots generated successfully!")
print(f"Saved to: {PLOT_DIR}")
print("=" * 50)