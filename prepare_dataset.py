
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ==================================================
# CONFIGURATION
# ==================================================

SOURCE_ROOT = Path(r"D:\mlproject\original_images")
OUTPUT_ROOT = Path(r"D:\mlproject\clean_classification_dataset")

TARGET_TRAIN = 350
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42
random.seed(SEED)

CLASSES = [
    "bottel", "buket", "comb", "cup", "cycle",
    "earbuds", "notebook", "pen", "phone", "shirt"
]


def read_valid_images(folder):
    """Return readable image paths from a class folder."""
    images = []

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        images.append(path)

    return images


def augment_image(source_path, output_path):
    """Create one augmented image for training."""
    image = cv2.imread(str(source_path))

    if image is None:
        raise ValueError(f"Cannot read image: {source_path}")

    height, width = image.shape[:2]

    # Horizontal flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # Small rotation
    angle = random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle, 1.0
    )
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Brightness adjustment
    brightness = random.uniform(0.85, 1.15)
    image = np.clip(
        image.astype(np.float32) * brightness, 0, 255
    ).astype(np.uint8)

    # Mild random crop
    height, width = image.shape[:2]
    crop_fraction = random.uniform(0.0, 0.12)
    crop_width = max(1, int(width * (1 - crop_fraction)))
    crop_height = max(1, int(height * (1 - crop_fraction)))

    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)

    image = image[y:y + crop_height, x:x + crop_width]
    image = cv2.resize(image, (width, height))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Could not save image: {output_path}")


def main():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Original dataset folder not found: {SOURCE_ROOT}"
        )

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT_ROOT}\n"
            "Rename or remove the output folder before running again."
        )

    # Read and validate the original 150 images.
    all_paths = []
    all_labels = []

    for class_name in CLASSES:
        class_dir = SOURCE_ROOT / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        images = read_valid_images(class_dir)

        if len(images) != 15:
            raise ValueError(
                f"{class_name}: found {len(images)} readable images; "
                "expected 15."
            )

        all_paths.extend(images)
        all_labels.extend([class_name] * len(images))

    # 70% training, 15% validation, 15% testing.
    train_paths, holdout_paths, train_labels, holdout_labels = (
        train_test_split(
            all_paths,
            all_labels,
            test_size=0.30,
            stratify=all_labels,
            random_state=SEED,
        )
    )

    valid_paths, test_paths, valid_labels, test_labels = (
        train_test_split(
            holdout_paths,
            holdout_labels,
            test_size=0.50,
            stratify=holdout_labels,
            random_state=SEED,
        )
    )

    splits = {
        "train": (train_paths, train_labels),
        "valid": (valid_paths, valid_labels),
        "test": (test_paths, test_labels),
    }

    # Copy originals into their respective splits.
    for split, (paths, labels) in splits.items():
        for path, label in zip(paths, labels):
            destination = OUTPUT_ROOT / split / label
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / path.name)

    # Augment training data only, evenly across classes.
    original_train_count = len(train_paths)
    needed = TARGET_TRAIN - original_train_count

    if needed < 0:
        raise ValueError(
            f"Training contains {original_train_count} images, "
            f"more than target {TARGET_TRAIN}."
        )

    train_by_class = {
        class_name: list(
            (OUTPUT_ROOT / "train" / class_name).glob("*")
        )
        for class_name in CLASSES
    }

    for class_name in CLASSES:
        train_by_class[class_name] = [
            p for p in train_by_class[class_name]
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    for index in range(needed):
        class_name = CLASSES[index % len(CLASSES)]
        source = random.choice(train_by_class[class_name])

        output_path = (
            OUTPUT_ROOT / "train" / class_name
            / f"aug_{index + 1:04d}.jpg"
        )
        augment_image(source, output_path)

    # Report final split counts.
    total = 0

    print("\n========== FINAL DATASET ==========")

    for split in ["train", "valid", "test"]:
        split_total = 0
        print(f"\n{split.upper()}")

        for class_name in CLASSES:
            folder = OUTPUT_ROOT / split / class_name
            count = len(read_valid_images(folder))
            split_total += count
            print(f"{class_name}: {count}")

        total += split_total
        print(f"Split total: {split_total}")

    print(f"\nTOTAL IMAGES: {total}")
    print(f"Output: {OUTPUT_ROOT}")
    print("Augmentation was applied only to training images.")


if __name__ == "__main__":
    main()
