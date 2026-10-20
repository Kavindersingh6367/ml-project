
from pathlib import Path

# Dataset root directory
ROOT = Path(r"D:\mlproject\clean_classification_dataset")

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

SPLITS = ["train", "valid", "test"]


def rename_images():
    if not ROOT.exists():
        print(f"Dataset not found: {ROOT}")
        return

    total_renamed = 0

    for split in SPLITS:
        split_path = ROOT / split

        if not split_path.is_dir():
            print(f"Skipping missing folder: {split_path}")
            continue

        print(f"\n========== {split.upper()} ==========")

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            images = sorted(
                file for file in class_dir.iterdir()
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )

            # Temporary names prevent filename collisions
            temp_files = []

            for index, image_path in enumerate(images, start=1):
                temp_path = image_path.with_name(
                    f"__temp_rename_{index:05d}{image_path.suffix.lower()}"
                )

                counter = 1
                while temp_path.exists():
                    temp_path = image_path.with_name(
                        f"__temp_rename_{index:05d}_{counter}{image_path.suffix.lower()}"
                    )
                    counter += 1

                image_path.rename(temp_path)
                temp_files.append((temp_path, image_path.suffix.lower()))

            for index, (temp_path, extension) in enumerate(
                temp_files, start=1
            ):
                new_path = class_dir / f"{class_dir.name}_{index:03d}{extension}"

                # Avoid overwriting an existing file
                if new_path.exists():
                    raise FileExistsError(
                        f"Cannot rename safely: {new_path} already exists"
                    )

                temp_path.rename(new_path)
                total_renamed += 1

            print(f"{class_dir.name}: {len(images)} images renamed")

    print("\n========== COMPLETED ==========")
    print(f"Total images renamed: {total_renamed}")
    print(f"Dataset: {ROOT}")


if __name__ == "__main__":
    rename_images()
