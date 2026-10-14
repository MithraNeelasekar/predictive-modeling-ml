"""
models.py
Machine Learning model definitions, descriptions, and training pipeline.
Supports Logistic Regression, Decision Trees, and Random Forests for Classification,
and Linear Regression, Decision Trees, and Random Forests for Regression.
"""

import time
from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# Beginner-friendly descriptions of each algorithm
MODEL_DESCRIPTIONS = {
    # Classification
    "Logistic Regression": {
        "type": "Classification",
        "description": "A linear model that uses the sigmoid function to estimate class probabilities. Fast, interpretable, and effective for linearly separable patterns.",
        "pros": "Simple, computationally efficient, provides probability estimates.",
        "cons": "Assumes a linear relationship between features and log-odds of target."
    },
    "Decision Tree Classifier": {
        "type": "Classification",
        "description": "A non-parametric tree-based model that splits features using Gini impurity or entropy. Mirrors human decision-making rules.",
        "pros": "Highly interpretable, handles non-linear relationships, requires minimal data preparation.",
        "cons": "Prone to overfitting if tree depth is not constrained."
    },
    "Random Forest Classifier": {
        "type": "Classification",
        "description": "An ensemble of multiple decision trees trained on random subsets of data and features (bagging). Aggregates tree predictions via majority vote.",
        "pros": "Robust against overfitting, handles complex interactions, high predictive accuracy.",
        "cons": "Less interpretable than a single tree, computationally more demanding."
    },
    # Regression
    "Linear Regression": {
        "type": "Regression",
        "description": "A foundational algorithm that models the relationship between dependent and independent variables by fitting a linear equation (Ordinary Least Squares).",
        "pros": "Extremely fast, straightforward coefficients interpretation.",
        "cons": "Sensitive to outliers, assumes linear relationship and constant variance (homoscedasticity)."
    },
    "Decision Tree Regressor": {
        "type": "Regression",
        "description": "A tree model that partitions feature space into distinct rectangular regions and predicts the mean target value of each region.",
        "pros": "Captures complex non-linear patterns, no assumption of linearity.",
        "cons": "Stepwise predictions rather than continuous, prone to high variance/overfitting."
    },
    "Random Forest Regressor": {
        "type": "Regression",
        "description": "An ensemble of decision tree regressors combining predictions by averaging. Smooths out extreme errors from individual trees.",
        "pros": "High accuracy, effectively minimizes variance and overfitting.",
        "cons": "Slower training time on very large datasets, larger model size in memory."
    }
}


def get_available_models(problem_type: str) -> List[str]:
    """
    Returns the list of available model names based on problem type.
    """
    if problem_type == "Classification":
        return [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier"
        ]
    elif problem_type == "Regression":
        return [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor"
        ]
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


def instantiate_model(model_name: str, random_state: int = 42):
    """
    Initializes a Scikit-learn estimator instance with reproducible seeds.
    """
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == "Decision Tree Classifier":
        return DecisionTreeClassifier(random_state=random_state, max_depth=8)
    elif model_name == "Random Forest Classifier":
        return RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
    elif model_name == "Linear Regression":
        return LinearRegression()
    elif model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(random_state=random_state, max_depth=8)
    elif model_name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_models(
    selected_models: List[str],
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Trains each selected model, measures training time, and generates predictions on the test set.
    """
    results = {}

    for model_name in selected_models:
        model = instantiate_model(model_name, random_state=random_state)

        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        # Predictions on test split
        y_pred = model.predict(X_test)

        # Probabilities for classification (if supported)
        y_proba = None
        if problem_type == "Classification":
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    y_proba = None
            elif hasattr(model, "decision_function"):
                try:
                    y_proba = model.decision_function(X_test)
                except Exception:
                    y_proba = None

        results[model_name] = {
            "model": model,
            "training_time": training_time,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "info": MODEL_DESCRIPTIONS.get(model_name, {})
        }

    return results
