import os
import csv
import numpy as np
import minicv as cv


CLASSES = [
    "angry",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}


def load_dataset(
    annotations_file: str,
    img_size: int = 96,
    grayscale: bool = False,
    normalise_mode: str = "minmax",
):

    images = []
    labels = []

    if not os.path.isfile(annotations_file):
        raise FileNotFoundError(
            f"Annotation file not found: {annotations_file}"
        )

    with open(annotations_file, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            img_path = row["image_path"]
            label_name = row["label"]

            if label_name not in LABEL2IDX:
                print(f"[SKIPPED] Unknown label: {label_name}")
                continue

            if not os.path.isfile(img_path):
                print(f"[SKIPPED] Missing file: {img_path}")
                continue

            try:

                img = cv.read_image(img_path)

                if img is None:
                    print(f"[SKIPPED] Could not read: {img_path}")
                    continue

                # ─────────────────────────────
                # Force RGB consistency
                # ─────────────────────────────
                if grayscale:

                    if img.ndim == 3:
                        img = cv.rgb_to_gray(img)

                else:

                    if img.ndim == 2:
                        img = cv.gray_to_rgb(img)

                    # Remove alpha channel safely
                    if img.ndim == 3 and img.shape[-1] == 4:
                        img = img[:, :, :3]

                # ─────────────────────────────
                # Resize
                # ─────────────────────────────
                img = cv.resize(img, img_size, img_size)

                # ─────────────────────────────
                # Normalize
                # ─────────────────────────────
                img = cv.normalize(
                    img,
                    mode=normalise_mode
                )

                # Final shape safety
                if grayscale:

                    if img.shape != (img_size, img_size):
                        print(f"[SKIPPED] Bad grayscale shape: {img.shape}")
                        continue

                else:

                    if img.shape != (img_size, img_size, 3):
                        print(f"[SKIPPED] Bad RGB shape: {img.shape}")
                        continue

                images.append(img.astype(np.float32))
                labels.append(LABEL2IDX[label_name])

            except Exception as e:

                print(f"[SKIPPED] {img_path} -> {e}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"\n[INFO] Loaded {len(X)} images")
    print(f"[INFO] X shape: {X.shape}")
    print(f"[INFO] y shape: {y.shape}")

    return X, y