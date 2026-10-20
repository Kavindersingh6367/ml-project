
from pathlib import Path
import csv

ROOT = Path(r"D:\mlproject\clean_classification_dataset")

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ROOT}")

    # Fixed mapping so labels stay consistent across all splits.
    classes = sorted(
        folder.name
        for folder in (ROOT / "train").iterdir()
        if folder.is_dir()
    )

    if not classes:
        raise ValueError("No class folders found inside the train directory.")

    class_to_id = {name: idx for idx, name in enumerate(classes)}
    records = []

    for split in SPLITS:
        split_dir = ROOT / split

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for class_name in classes:
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder: {class_dir}"
                )

            for image_path in sorted(class_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    records.append({
                        "split": split,
                        "image_path": str(image_path.resolve()),
                        "filename": image_path.name,
                        "class_name": class_name,
                        "label_id": class_to_id[class_name],
                    })

    output_csv = ROOT / "labels.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "image_path",
                "filename",
                "class_name",
                "label_id",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print("\n========== CLASS LABEL MAPPING ==========")
    for class_name, label_id in class_to_id.items():
        print(f"{class_name}: {label_id}")

    print("\n========== ANNOTATION SUMMARY ==========")
    for split in SPLITS:
        count = sum(row["split"] == split for row in records)
        print(f"{split}: {count} images")

    print(f"\nTotal labeled images: {len(records)}")
    print(f"CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
