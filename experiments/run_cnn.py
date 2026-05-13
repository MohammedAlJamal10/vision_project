"""
experiments/run_cnn.py  — FIXED
================================

Bugs fixed vs original
-----------------------
1. DATA LEAKAGE  — original normalised the ENTIRE dataset (train+val+test
   combined) before splitting:

       X = (X - X.mean()) / (X.std() + 1e-8)   ← leaks test stats into train

   Fix: normalise per-image inside the loader (minicv.normalize already does
   this), or compute mean/std on X_train only and apply to all splits.

2. NON-MINICV NORMALISATION — the per-dataset z-score bypassed minicv.
   The loader already normalises each image with minicv; no extra step needed.

3. ADAM INSTEAD OF SGD — SGD with lr=0.02 on this architecture is sensitive
   to the exact LR choice.  Adam is used here; initial lr=3e-4 is a safe
   default (Karpathy constant) that works across a wide range of architectures.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from dataset.loader  import load_dataset, CLASSES
from dataset.split   import stratified_split, split_summary
from dataset.augment import augment_cnn_batch
from models.cnn.model import SimpleCNN
from evaluation.metrics import classification_report
from evaluation.plots import plot_confusion_matrix, plot_training_curves
# ── use the FIXED optimiser (Adam recommended; SGD also available) ────────────
from models.cnn.optim import Adam, SGD       # import from the fixed file


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED     = 42
IMG_SIZE = 96
ANN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "annotations.csv")
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
PLOTS_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(LOG_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading dataset …")
ANN_PATH = os.path.join("data", "annotations.csv")

X, y = load_dataset(
    ANN_PATH,
    img_size=96,
    grayscale=False,
    normalise_mode="minmax"
)
# Images are read as RGB float values in [0,1].  Avoid per-image min-max
# normalisation here because it can distort useful colour/contrast cues.

# ── 2. Split (before any global statistics are computed) ─────────────────────
(X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(
    X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=SEED
)
split_summary(y_train, y_val, y_test, class_names=CLASSES)

# ── 3. FIX: compute normalisation stats on TRAIN ONLY, apply to all ──────────
# If you want global z-score (e.g. for faster convergence), do it like this:
#   mu, sigma = X_train.mean(), X_train.std()
#   X_train = (X_train - mu) / (sigma + 1e-8)
#   X_val   = (X_val   - mu) / (sigma + 1e-8)    ← apply TRAIN stats to val
#   X_test  = (X_test  - mu) / (sigma + 1e-8)    ← apply TRAIN stats to test
# Per-image min-max (already done by loader) is usually sufficient; skip
# global normalisation unless the model is still diverging.

# ── 4. Reshape to (N, C, H, W) ───────────────────────────────────────────────
X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32)

X_val = np.transpose(X_val, (0, 3, 1, 2)).astype(np.float32)

X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32)

print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ── 5. Model ──────────────────────────────────────────────────────────────────
n_classes = len(np.unique(y))
model = SimpleCNN(
    in_channels=3,
    n_classes=n_classes,
    img_size=96,
    lambda_l2=1e-4
)

# ── 6. Optimiser — Adam is far more stable than SGD for this task ─────────────
optimizer = Adam(lr=1e-4)
# If you prefer SGD, use:  optimizer = SGD(lr=0.005, momentum=0.9)

# ── 7. LR scheduler ──────────────────────────────────────────────────────────
#scheduler = CosineScheduler(lr_start=1e-4, lr_end=1e-5, n_epochs=100)

# ── 8. Early stopping (2-arg step to match EarlyStopping.step signature) ─────
#early = EarlyStopping(patience=15, min_delta=1e-4)

# ── 9. Logger + checkpoint ────────────────────────────────────────────────────
#logger  = TrainingLogger(os.path.join(LOG_DIR, "cnn_log.csv"), model_name="SimpleCNN")
#ckpt    = CheckpointManager(os.path.join(LOG_DIR, "checkpoints"), model_name="cnn")

# ── 10. Train ─────────────────────────────────────────────────────────────────
print("\nTraining SimpleCNN …")
model.fit(
    X_train, y_train,
    X_val,   y_val,
    optimizer      = optimizer,
    batch_size     = 32,
    max_epochs     = 100,
    grad_clip_norm = 5.0,
    augment_fn     = augment_cnn_batch,
    seed           = SEED,
    patience       = 10,
    #scheduler      = scheduler,
    #early_stopping = early,
    #logger         = logger,
)

plot_training_curves(
    model.history,
    title="SimpleCNN Training Curves",
    save_path=os.path.join(PLOTS_DIR, "cnn_training_curves.png")
)

# ── 11. Save best checkpoint ──────────────────────────────────────────────────
#ckpt.save(model.get_params(), optimizer, config={
  #  "model": "SimpleCNN", "img_size": IMG_SIZE,
   # "n_classes": n_classes, "optimizer": "Adam", "lr": 1e-4,
#})

# ── 12. Evaluate on test set ──────────────────────────────────────────────────
test_acc = model.score(X_test, y_test)
y_pred = model.predict(X_test)

plot_confusion_matrix(
    y_test,
    y_pred,
    class_names=CLASSES,
    title="SimpleCNN Confusion Matrix",
    save_path=os.path.join(PLOTS_DIR, "cnn_confusion_matrix.png")
)


print(f"\n{'='*50}")
print(f"  Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"{'='*50}")

classification_report(
    y_test,
    y_pred,
    class_names=CLASSES
)
