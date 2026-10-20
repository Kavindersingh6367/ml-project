
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path(
    r"D:\mlproject\clean_classification_dataset\features"
)

OUTPUT_DIR = FEATURE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# LOAD EXTRACTED FEATURES
# ============================================================

def load_split(split_name):
    file_path = FEATURE_DIR / f"{split_name}_features.npz"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing feature file: {file_path}\n"
            "Run feature_extraction.py first."
        )

    with np.load(file_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        classes = data["class_names"].tolist()

    if X.ndim != 2:
        raise ValueError(f"{split_name}: expected a 2D feature matrix.")

    if len(X) != len(y):
        raise ValueError(f"{split_name}: feature and label counts differ.")

    if len(X) == 0:
        raise ValueError(f"{split_name}: no images found.")

    if not np.isfinite(X).all():
        raise ValueError(f"{split_name}: features contain NaN or infinity.")

    return X, y, classes


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}Accuracy (%)": round(
            accuracy_score(y_true, y_pred) * 100, 2
        ),
        f"{prefix}Precision (%)": round(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ) * 100, 2
        ),
        f"{prefix}Recall (%)": round(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ) * 100, 2
        ),
        f"{prefix}F1-score (%)": round(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ) * 100, 2
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("        IMAGE CLASSIFICATION - MODEL TRAINING")
    print("=" * 65)

    X_train, y_train, classes = load_split("train")
    X_valid, y_valid, valid_classes = load_split("valid")
    X_test, y_test, test_classes = load_split("test")

    # Verify identical class mappings across splits.
    if classes != valid_classes or classes != test_classes:
        raise ValueError(
            "Class names/order differ between train, valid and test."
        )

    expected_labels = set(range(len(classes)))

    if set(np.unique(y_train)) != expected_labels:
        raise ValueError("Training set does not contain all classes.")

    if not set(np.unique(y_valid)).issubset(expected_labels):
        raise ValueError("Validation labels contain unknown classes.")

    if not set(np.unique(y_test)).issubset(expected_labels):
        raise ValueError("Test labels contain unknown classes.")

    print("\nDATASET INFORMATION")
    print("-" * 65)
    print(f"Training images   : {len(X_train)}")
    print(f"Validation images : {len(X_valid)}")
    print(f"Test images       : {len(X_test)}")
    print(f"Features per image: {X_train.shape[1]}")
    print(f"Number of classes : {len(classes)}")
    print(f"Classes           : {classes}")

    # ========================================================
    # DEFINE SIX MODELS
    # ========================================================

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                C=1.0,
                random_state=RANDOM_STATE
            )
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            random_state=RANDOM_STATE
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=3)
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                random_state=RANDOM_STATE
            )
        ),

        "Gaussian Naive Bayes": GaussianNB(),
    }

    validation_results = []
    test_results = []
    fitted_models = {}

    # ========================================================
    # TRAIN AND VALIDATE EACH MODEL
    # ========================================================

    for name, model in models.items():
        print(f"\n{'=' * 65}")
        print(f"Training: {name}")
        print("=" * 65)

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        # Validation is used for model comparison.
        valid_predictions = model.predict(X_valid)

        valid_metrics = calculate_metrics(
            y_valid, valid_predictions, "Validation "
        )

        validation_row = {
            "Model": name,
            **valid_metrics,
            "Training Time (seconds)": round(training_time, 3),
        }

        validation_results.append(validation_row)
        fitted_models[name] = model

        print(
            f"Validation accuracy: "
            f"{valid_metrics['Validation Accuracy (%)']:.2f}%"
        )
        print(
            f"Validation macro F1: "
            f"{valid_metrics['Validation F1-score (%)']:.2f}%"
        )
        print(f"Training time: {training_time:.3f} seconds")

    # ========================================================
    # SAVE VALIDATION COMPARISON
    # ========================================================

    validation_df = pd.DataFrame(validation_results)

    validation_df = validation_df.sort_values(
        by=["Validation Accuracy (%)", "Validation F1-score (%)"],
        ascending=False
    ).reset_index(drop=True)

    validation_path = OUTPUT_DIR / "model_comparison.csv"
    validation_df.to_csv(validation_path, index=False)

    print(f"\n{'=' * 65}")
    print("VALIDATION PERFORMANCE COMPARISON")
    print("=" * 65)
    print(validation_df.to_string(index=False))

    # Select best model without looking at test performance.
    best_name = validation_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    print(f"\nSelected using validation data: {best_name}")

    # ========================================================
    # FINAL TEST EVALUATION
    # Evaluate the selected model only on the test set.
    # ========================================================

    print(f"\n{'=' * 65}")
    print("FINAL TEST EVALUATION")
    print("=" * 65)

    test_predictions = best_model.predict(X_test)

    final_metrics = calculate_metrics(
        y_test, test_predictions, "Test "
    )

    print(f"Selected model: {best_name}")
    print(f"Accuracy : {final_metrics['Test Accuracy (%)']:.2f}%")
    print(f"Precision: {final_metrics['Test Precision (%)']:.2f}%")
    print(f"Recall   : {final_metrics['Test Recall (%)']:.2f}%")
    print(f"F1-score : {final_metrics['Test F1-score (%)']:.2f}%")

    # Save selected model's classification report.
    report = classification_report(
        y_test,
        test_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_path = OUTPUT_DIR / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0
        )
    )

    # Save the selected model's confusion matrix.
    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=list(range(len(classes)))
    )

    matrix_path = OUTPUT_DIR / "confusion_matrix.csv"

    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    ).to_csv(matrix_path, index_label="Actual class")

    # Save final selected-model metrics.
    final_results = {
        "Model": best_name,
        **final_metrics,
        "Test images": int(len(y_test)),
        "Training images": int(len(y_train)),
        "Validation images": int(len(y_valid)),
        "Feature count": int(X_train.shape[1]),
    }

    final_results_path = OUTPUT_DIR / "final_test_results.csv"
    pd.DataFrame([final_results]).to_csv(
        final_results_path, index=False
    )

    # ========================================================
    # SAVE TRAINED MODELS
    # ========================================================

    # Save each fitted model for future comparison or inference.
    for name, model in fitted_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(
            model,
            OUTPUT_DIR / f"{safe_name}.joblib"
        )

    # Save the selected best model separately.
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

    metadata = {
        "selected_model": best_name,
        "class_names": classes,
        "image_size": [64, 64],
        "feature_count": int(X_train.shape[1]),
        "validation_selection_metric": "accuracy",
        "test_metrics": {
            key: float(value)
            for key, value in final_metrics.items()
        },
    }

    metadata_path = OUTPUT_DIR / "model_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # ========================================================
    # SUMMARY
    # ========================================================

    print(f"\n{'=' * 65}")
    print("TRAINING AND EVALUATION COMPLETED")
    print("=" * 65)
    print(f"Best model              : {best_name}")
    print(f"Validation comparison   : {validation_path}")
    print(f"Test results            : {final_results_path}")
    print(f"Classification report   : {report_path}")
    print(f"Confusion matrix        : {matrix_path}")
    print(f"Saved model              : {OUTPUT_DIR / 'best_model.joblib'}")
    print(f"All output files         : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
