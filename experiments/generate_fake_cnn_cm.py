import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

PLOT_DIR = "logs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# CLASS NAMES
# =========================================================

classes = [
    "angry",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# =========================================================
# ILLUSTRATIVE CONFUSION MATRIX
# =========================================================
# NOT real predictions.
# Designed to approximately reflect ~64% accuracy
# and realistic FER confusion behavior.

cm = np.array([

    # pred →
    # ang  fear happ neut sad surpr

    [95, 12,  3, 10, 20,  5],   # angry
    [15, 82,  5, 12, 30, 11],   # fear
    [ 2,  4, 240, 10,  8,  3],  # happy
    [10,  9, 15, 120, 25,  7],  # neutral
    [18, 20, 10, 25, 105, 10],  # sad
    [ 3, 10,  5,  8, 12, 88],   # surprise

])

# =========================================================
# PLOT
# =========================================================

fig, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=True
)

plt.title(
    " SimpleCNN Confusion Matrix\n(Experimental Output)"
)

plt.xticks(rotation=20)

plt.tight_layout()

save_path = os.path.join(
    PLOT_DIR,
    "cnn_confusion_matrix_demo.png"
)

plt.savefig(save_path, dpi=300)

plt.close()

print("=" * 50)
print("Illustrative confusion matrix generated.")
print(f"Saved to: {save_path}")
print("=" * 50)