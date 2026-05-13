import os
import csv

DATA_DIR = "data"

CLASSES = [
    "angry",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

OUTPUT_CSV = os.path.join(DATA_DIR, "annotations.csv")


def main():

    rows = []

    for label in CLASSES:

        class_dir = os.path.join(DATA_DIR, label)

        if not os.path.isdir(class_dir):
            print(f"[WARNING] Missing folder: {class_dir}")
            continue

        for file in os.listdir(class_dir):

            file_lower = file.lower()

            if not file_lower.endswith((
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )):
                continue

            rel_path = os.path.join(DATA_DIR, label, file)

            rows.append([rel_path, label])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["image_path", "label"])

        writer.writerows(rows)

    print(f"\n[INFO] Saved: {OUTPUT_CSV}")
    print(f"[INFO] Total samples: {len(rows)}")


if __name__ == "__main__":
    main()