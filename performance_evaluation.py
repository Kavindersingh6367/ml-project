
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path(
    r"D:\mlproject\clean_classification_dataset\features"
)

MODEL_DIR = FEATURE_DIR / "models"
OUTPUT_DIR = MODEL_DIR / "evaluation_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_metadata(metadata_path):
    """Load metadata and support both previous JSON formats."""
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Older script used "best_model"; newer script uses
    # "selected_model".
    model_name = metadata.get(
        "selected_model",
        metadata.get("best_model", "Unknown")
    )

    # Support both formats for the test accuracy.
    test_accuracy = metadata.get("test_accuracy")

    if test_accuracy is None:
        test_metrics = metadata.get("test_metrics", {})
        test_accuracy = test_metrics.get("Test Accuracy (%)")

        if test_accuracy is not None:
            test_accuracy = test_accuracy / 100.0

    return metadata, model_name, test_accuracy


# ============================================================
# PLOT MODEL METRICS
# ============================================================

def plot_model_comparison(results):
    """Plot validation metrics for all trained models."""
    metrics = [
        "Validation Accuracy (%)",
        "Validation Precision (%)",
        "Validation Recall (%)",
        "Validation F1-score (%)",
    ]

    available = [column for column in metrics if column in results.columns]

    if not available:
        print("No validation metric columns found; skipping comparison graph.")
        return

    plot_data = results.set_index("Model")[available]

    ax = plot_data.plot(
        kind="bar",
        figsize=(13, 7),
        width=0.8
    )

    ax.set_title("Classification Model Performance Comparison")
    ax.set_xlabel("Classification Algorithm")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    path = OUTPUT_DIR / "model_performance_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


# ============================================================
# PLOT TRAINING TIMES
# ============================================================

def plot_training_time(results):
    """Compare the training time of all models."""
    column = "Training Time (seconds)"

    if column not in results.columns:
        print("Training-time column missing; skipping training-time graph.")
        return

    plot_data = results.sort_values(column)

    ax = plot_data.plot(
        x="Model",
        y=column,
        kind="bar",
        legend=False,
        figsize=(11, 6)
    )

    ax.set_title("Training Time Comparison")
    ax.set_xlabel("Classification Algorithm")
    ax.set_ylabel("Training Time (seconds)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    path = OUTPUT_DIR / "training_time_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(matrix_path, model_name):
    """Visualize the selected model's saved confusion matrix."""
    matrix_df = pd.read_csv(matrix_path, index_col=0)

    labels = matrix_df.index.astype(str).tolist()
    matrix = matrix_df.to_numpy(dtype=int)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    if matrix.shape[0] != len(labels):
        raise ValueError("Confusion matrix labels do not match its shape.")

    fig, ax = plt.subplots(figsize=(11, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
        colorbar=True
    )

    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    plt.tight_layout()

    path = OUTPUT_DIR / "best_model_confusion_matrix.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


# ============================================================
# MAIN
# ============================================================

def main():
    comparison_path = MODEL_DIR / "model_comparison.csv"
    matrix_path = MODEL_DIR / "confusion_matrix.csv"
    metadata_path = MODEL_DIR / "model_metadata.json"
    final_results_path = MODEL_DIR / "final_test_results.csv"

    required_files = [comparison_path, matrix_path, metadata_path]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run train_models.py first."
            )

    results = pd.read_csv(comparison_path)

    if results.empty:
        raise ValueError("Model comparison file is empty.")

    metadata, model_name, test_accuracy = load_metadata(metadata_path)

    print("\n" + "=" * 65)
    print("         MODEL PERFORMANCE EVALUATION")
    print("=" * 65)

    print("\n========== VALIDATION COMPARISON ==========")
    print(results.to_string(index=False))

    print("\n========== SELECTED MODEL ==========")
    print("Model:", model_name)

    if test_accuracy is not None:
        print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
    else:
        print("Test accuracy was not found in the metadata.")

    # Display the remaining final test metrics if available.
    if final_results_path.is_file():
        final_df = pd.read_csv(final_results_path)

        print("\n========== FINAL TEST METRICS ==========")
        print(final_df.to_string(index=False))

    # Generate evaluation plots.
    plot_model_comparison(results)
    plot_training_time(results)
    plot_confusion_matrix(matrix_path, model_name)

    print("\n" + "=" * 65)
    print("PERFORMANCE EVALUATION COMPLETED")
    print("=" * 65)
    print("All graphs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
