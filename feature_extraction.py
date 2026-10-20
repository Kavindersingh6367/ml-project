
from pathlib import Path
import json

import cv2
import numpy as np

# Dataset and output paths
ROOT = Path(r"D:\mlproject\clean_classification_dataset")
OUTPUT_DIR = ROOT / "features"

IMAGE_SIZE = (64, 64)
SPLITS = ("train", "valid", "test")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def extract_features(image_path: Path) -> np.ndarray:
    """Convert one image into a normalized RGB pixel feature vector."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to the range [0, 1].
    image = image.astype(np.float32) / 255.0

    return image.flatten()


def main():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {ROOT}")

    train_dir = ROOT / "train"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    # Establish one consistent class-to-label mapping.
    class_names = sorted(
        folder.name for folder in train_dir.iterdir()
        if folder.is_dir()
    )

    if not class_names:
        raise ValueError("No class folders found in the training directory.")

    class_to_id = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Class mapping:")
    for name, label in class_to_id.items():
        print(f"  {name}: {label}")

    results = {}

    for split in SPLITS:
        split_dir = ROOT / split

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        features = []
        labels = []
        image_paths = []
        skipped = []

        for class_name in class_names:
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}"
                )

            for image_path in sorted(class_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    continue

                try:
                    vector = extract_features(image_path)
                except (ValueError, cv2.error) as error:
                    skipped.append(str(image_path))
                    print(f"WARNING: {error}")
                    continue

                features.append(vector)
                labels.append(class_to_id[class_name])
                image_paths.append(str(image_path.resolve()))

        if not features:
            raise ValueError(f"No readable images found in {split_dir}")

        X = np.stack(features).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)

        results[f"X_{split}"] = X
        results[f"y_{split}"] = y

        np.savez_compressed(
            OUTPUT_DIR / f"{split}_features.npz",
            X=X,
            y=y,
            class_names=np.asarray(class_names),
            image_paths=np.asarray(image_paths),
        )

        print(f"\n{'=' * 40}")
        print(f"Split: {split.upper()}")
        print(f"Images processed: {len(X)}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label array shape: {y.shape}")
        print(f"Skipped unreadable images: {len(skipped)}")

        for class_name, label in class_to_id.items():
            count = int(np.sum(y == label))
            print(f"  {class_name}: {count}")

    # Save the mapping for use when training and predicting.
    mapping_path = OUTPUT_DIR / "class_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(class_to_id, file, indent=4)

    print(f"\n{'=' * 40}")
    print("FEATURE EXTRACTION COMPLETED!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Files created:")
    print("  train_features.npz")
    print("  valid_features.npz")
    print("  test_features.npz")
    print("  class_mapping.json")


if __name__ == "__main__":
    main()
