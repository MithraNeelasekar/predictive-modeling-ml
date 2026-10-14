"""
evaluation.py
Calculates performance metrics, generates comparison tables, and creates
clean Matplotlib/Seaborn visualization plots for both classification and regression.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)


# Set clean modern style for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


METRIC_EXPLANATIONS = {
    # Classification
    "Accuracy": "The proportion of total predictions that were correct (both true positives and true negatives). Be cautious with imbalanced datasets where accuracy can be deceptively high.",
    "Precision": "Out of all instances the model predicted as positive, how many were actually positive? (Minimizes false alarms / False Positives).",
    "Recall (Sensitivity)": "Out of all actual positive instances, how many did the model correctly identify? (Minimizes missed detections / False Negatives).",
    "F1-Score": "The harmonic mean of Precision and Recall. It provides a balanced assessment and is the gold standard when classes are unevenly distributed.",
    "ROC-AUC": "Area Under the Receiver Operating Characteristic Curve. Measures how effectively the model distinguishes between positive and negative classes across all possible decision thresholds (1.0 = perfect, 0.5 = random guess).",
    "Confusion Matrix": "A cross-tabulation table comparing the true labels against the model's predicted labels, revealing exact counts of True Positives, False Positives, True Negatives, and False Negatives.",
    # Regression
    "MAE (Mean Absolute Error)": "The average magnitude of the absolute errors between predictions and actual targets. Expressed in the exact same unit as the target variable.",
    "MSE (Mean Squared Error)": "The average of squared differences between predictions and actual targets. Strongly penalizes large errors or outliers.",
    "RMSE (Root Mean Squared Error)": "The square root of MSE. Like MAE, it is expressed in the target variable's unit, but penalizes large errors more severely than MAE.",
    "R² Score (Coefficient of Determination)": "The proportion of variance in the dependent variable predictable from the independent variables (1.0 = perfect explanation, 0.0 = baseline predicting the mean)."
}


def evaluate_classification_model(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes all standard classification metrics for a single model.
    """
    num_classes = len(np.unique(y_test))
    is_binary = (num_classes == 2)
    average_mode = "binary" if is_binary else "weighted"

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)

    # ROC AUC calculation
    auc_val = None
    if y_proba is not None:
        try:
            if is_binary:
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    auc_val = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc_val = roc_auc_score(y_test, y_proba)
            else:
                if y_proba.ndim == 2 and y_proba.shape[1] == num_classes:
                    auc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc_val = None

    cm = confusion_matrix(y_test, y_pred)

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc_val) if auc_val is not None else None,
        "confusion_matrix": cm
    }


def evaluate_regression_model(
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes all standard regression metrics for a single model.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2 Score": float(r2)
    }


def build_comparison_summary(
    trained_results: Dict[str, Dict[str, Any]],
    problem_type: str,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, str, Dict[str, Dict[str, Any]]]:
    """
    Evaluates all trained models, builds a comparison table, and determines the best model.
    Returns (comparison_df, best_model_name, evaluation_cache).
    """
    records = []
    eval_cache = {}

    for model_name, res in trained_results.items():
        y_pred = res["y_pred"]
        y_proba = res.get("y_proba")
        training_time = res["training_time"]

        if problem_type == "Classification":
            metrics = evaluate_classification_model(y_test, y_pred, y_proba)
            eval_cache[model_name] = metrics
            records.append({
                "Model": model_name,
                "Accuracy": round(metrics["Accuracy"], 4),
                "Precision": round(metrics["Precision"], 4),
                "Recall": round(metrics["Recall"], 4),
                "F1-Score": round(metrics["F1-Score"], 4),
                "ROC-AUC": round(metrics["ROC-AUC"], 4) if metrics["ROC-AUC"] is not None else "N/A",
                "Training Time (s)": round(training_time, 4)
            })
        else:
            metrics = evaluate_regression_model(y_test, y_pred)
            eval_cache[model_name] = metrics
            records.append({
                "Model": model_name,
                "MAE": round(metrics["MAE"], 4),
                "MSE": round(metrics["MSE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "R2 Score": round(metrics["R2 Score"], 4),
                "Training Time (s)": round(training_time, 4)
            })

    comparison_df = pd.DataFrame(records)

    # Determine best model
    if problem_type == "Classification":
        # Sort by F1-Score (preferred for balanced assessment), then Accuracy
        sorted_df = comparison_df.sort_values(by=["F1-Score", "Accuracy"], ascending=[False, False])
        best_model_name = sorted_df.iloc[0]["Model"]
    else:
        # Sort by R2 Score (higher is better), then RMSE (lower is better)
        sorted_df = comparison_df.sort_values(by=["R2 Score", "RMSE"], ascending=[False, True])
        best_model_name = sorted_df.iloc[0]["Model"]

    return comparison_df, best_model_name, eval_cache


# Visualizations

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], model_name: str) -> plt.Figure:
    """
    Renders an annotated Seaborn confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    return fig


def plot_roc_curves(
    trained_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    classes: List[str]
) -> Optional[plt.Figure]:
    """
    Plots ROC curves for all models that provide prediction probabilities.
    Supports binary classification.
    """
    num_classes = len(np.unique(y_test))
    if num_classes != 2:
        return None  # Multi-class handled via metric reporting

    fig, ax = plt.subplots(figsize=(7, 5))
    has_any_roc = False

    for model_name, res in trained_results.items():
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue

        try:
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                scores = y_proba[:, 1]
            else:
                scores = y_proba

            fpr, tpr, _ = roc_curve(y_test, scores)
            auc_val = roc_auc_score(y_test, scores)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", lw=2)
            has_any_roc = True
        except Exception:
            continue

    if not has_any_roc:
        plt.close(fig)
        return None

    # Reference diagonal (random classifier baseline)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Random Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curves Comparison", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    return fig


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> plt.Figure:
    """
    Renders an Actual vs. Predicted scatter plot with reference 45-degree line.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", s=45, color="#1f77b4")

    # 45-degree reference line (ideal prediction)
    min_val = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_test)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", lw=2, label="Perfect Fit (y = x)")

    ax.set_xlabel("Actual Values", fontsize=11)
    ax.set_ylabel("Predicted Values", fontsize=11)
    ax.set_title(f"Actual vs. Predicted: {model_name}", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="upper left")
    return fig


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> plt.Figure:
    """
    Renders a residual distribution plot (errors = Actual - Predicted).
    """
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", ax=ax, bins=25)
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    ax.set_xlabel("Residual Error (Actual - Predicted)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Residual Distribution: {model_name}", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="upper right")
    return fig
