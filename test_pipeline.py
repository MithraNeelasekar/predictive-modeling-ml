"""
test_pipeline.py
Automated end-to-end verification script for the machine learning pipeline.
Tests data preprocessing, model training, evaluation metrics, and single-sample inference
for both Classification and Regression tasks.
"""

import sys
import os
import pandas as pd
import numpy as np

# Configure UTF-8 encoding for standard output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure root dir is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import inspect_dataframe, detect_problem_type, prepare_dataset, transform_single_input
from models import get_available_models, train_models
from evaluation import build_comparison_summary, plot_confusion_matrix, plot_roc_curves, plot_actual_vs_predicted


def test_classification_pipeline():
    print("========================================")
    print("Testing Classification Pipeline (Breast Cancer)")
    print("========================================")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "breast_cancer.csv")
    assert os.path.exists(data_path), f"Sample data file not found: {data_path}"

    df = pd.read_csv(data_path)
    target_col = "target"

    # 1. Inspection
    info = inspect_dataframe(df)
    assert info["num_rows"] == 569, f"Expected 569 rows, got {info['num_rows']}"
    assert target_col in info["columns"]
    print("[OK] inspect_dataframe passed.")

    # 2. Problem Type Detection
    prob_type = detect_problem_type(df, target_col)
    assert prob_type == "Classification", f"Expected Classification, got {prob_type}"
    print("[OK] detect_problem_type passed.")

    # 3. Preprocessing (No data leakage test)
    prep = prepare_dataset(df, target_col=target_col, problem_type="Classification", test_size=0.2, random_state=42)
    assert prep["X_train_proc"].shape[0] + prep["X_test_proc"].shape[0] == 569
    assert len(prep["classes"]) == 2
    print(f"[OK] prepare_dataset passed: Train={prep['X_train_proc'].shape}, Test={prep['X_test_proc'].shape}, Classes={prep['classes']}")

    # 4. Model Training
    models_to_train = get_available_models("Classification")
    assert len(models_to_train) == 3
    results = train_models(
        models_to_train,
        "Classification",
        prep["X_train_proc"],
        prep["y_train"],
        prep["X_test_proc"],
        random_state=42
    )
    assert len(results) == 3
    print("[OK] train_models (Classification) passed.")

    # 5. Evaluation & Comparison
    comp_df, best_model, eval_cache = build_comparison_summary(results, "Classification", prep["y_test"])
    print(f"[OK] Comparison Table:\n{comp_df.to_string(index=False)}")
    print(f"[OK] Best Model Identified: {best_model}")
    assert best_model in models_to_train
    assert "Accuracy" in comp_df.columns
    assert "F1-Score" in comp_df.columns

    # 6. Plotting Functions
    cm_fig = plot_confusion_matrix(eval_cache[best_model]["confusion_matrix"], prep["classes"], best_model)
    assert cm_fig is not None
    roc_fig = plot_roc_curves(results, prep["y_test"], prep["classes"])
    assert roc_fig is not None
    print("[OK] Confusion matrix and ROC curve figures generated cleanly.")

    # 7. Single Sample Inference
    sample_raw = df.drop(columns=[target_col]).iloc[0].to_dict()
    raw_cols = df.drop(columns=[target_col]).columns.tolist()
    vec = transform_single_input(sample_raw, prep["preprocessor"], raw_cols)
    best_estimator = results[best_model]["model"]
    pred_class_idx = best_estimator.predict(vec)[0]
    pred_label = prep["label_encoder"].inverse_transform([pred_class_idx])[0]
    print(f"[OK] Single sample inference passed: Predicted Class = {pred_label}")


def test_regression_pipeline():
    print("\n========================================")
    print("Testing Regression Pipeline (Diabetes)")
    print("========================================")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diabetes_regression.csv")
    assert os.path.exists(data_path), f"Sample data file not found: {data_path}"

    df = pd.read_csv(data_path)
    target_col = "target"

    # 1. Inspection
    info = inspect_dataframe(df)
    assert info["num_rows"] == 442
    print("[OK] inspect_dataframe passed.")

    # 2. Problem Type Detection
    prob_type = detect_problem_type(df, target_col)
    assert prob_type == "Regression", f"Expected Regression, got {prob_type}"
    print("[OK] detect_problem_type passed.")

    # 3. Preprocessing
    prep = prepare_dataset(df, target_col=target_col, problem_type="Regression", test_size=0.2, random_state=42)
    assert prep["X_train_proc"].shape[0] + prep["X_test_proc"].shape[0] == 442
    print(f"[OK] prepare_dataset passed: Train={prep['X_train_proc'].shape}, Test={prep['X_test_proc'].shape}")

    # 4. Model Training
    models_to_train = get_available_models("Regression")
    assert len(models_to_train) == 3
    results = train_models(
        models_to_train,
        "Regression",
        prep["X_train_proc"],
        prep["y_train"],
        prep["X_test_proc"],
        random_state=42
    )
    assert len(results) == 3
    print("[OK] train_models (Regression) passed.")

    # 5. Evaluation & Comparison
    comp_df, best_model, eval_cache = build_comparison_summary(results, "Regression", prep["y_test"])
    print(f"[OK] Comparison Table:\n{comp_df.to_string(index=False)}")
    print(f"[OK] Best Model Identified: {best_model}")
    assert best_model in models_to_train
    assert "R2 Score" in comp_df.columns
    assert "RMSE" in comp_df.columns

    # 6. Plotting Functions
    reg_fig = plot_actual_vs_predicted(prep["y_test"], results[best_model]["y_pred"], best_model)
    assert reg_fig is not None
    print("[OK] Actual vs Predicted regression plot generated cleanly.")

    # 7. Single Sample Inference
    sample_raw = df.drop(columns=[target_col]).iloc[0].to_dict()
    raw_cols = df.drop(columns=[target_col]).columns.tolist()
    vec = transform_single_input(sample_raw, prep["preprocessor"], raw_cols)
    best_estimator = results[best_model]["model"]
    pred_val = best_estimator.predict(vec)[0]
    print(f"[OK] Single sample inference passed: Predicted Target Value = {pred_val:.2f}")


if __name__ == "__main__":
    test_classification_pipeline()
    test_regression_pipeline()
    print("\nALL PIPELINE TESTS PASSED SUCCESSFULLY!")
