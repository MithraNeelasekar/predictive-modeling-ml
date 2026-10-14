"""
preprocessing.py
Data exploration, validation, and leakage-free preprocessing pipeline.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def inspect_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive summary of the dataset for the Dataset Explorer.
    """
    num_rows, num_cols = df.shape

    # Missing values summary
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / num_rows) * 100
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values": missing_series.values,
        "Missing %": missing_pct.round(2).values,
        "Unique Values": [df[col].nunique() for col in df.columns]
    })

    # Summary statistics
    numeric_df = df.select_dtypes(include=[np.number])
    stats_df = numeric_df.describe().T if not numeric_df.empty else pd.DataFrame()

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": df.columns.tolist(),
        "missing_summary": missing_df,
        "summary_statistics": stats_df,
        "numeric_columns": numeric_df.columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist()
    }


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Heuristically infers whether the target variable represents a Classification
    or Regression problem.
    """
    target_series = df[target_col].dropna()
    num_unique = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series.dtype)

    if not is_numeric:
        return "Classification"

    # If numeric but only 2 to 10 unique values (e.g. 0/1 binary or few integer classes)
    if num_unique <= 10 and pd.api.types.is_integer_dtype(target_series.dtype):
        return "Classification"

    return "Regression"


def prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares dataset for machine learning strictly preventing data leakage:
    1. Removes rows with missing target values.
    2. Splits into train and test sets FIRST.
    3. Fits imputer, scaler, and encoder ONLY on train data.
    4. Transforms both train and test data using fitted transformers.
    """
    # Validation: Check target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Validation: Check dataframe size
    if len(df) < 15:
        raise ValueError("Dataset has fewer than 15 rows. Please provide more data for reliable training and testing.")

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_col]).copy()
    if len(clean_df) < 10:
        raise ValueError("Too many missing values in the target column. Not enough samples remaining to train.")

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    if X.empty:
        raise ValueError("The dataset contains no feature columns after selecting the target.")

    # Identify numeric and categorical feature columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Stratified split for classification when feasible
    stratify_target = None
    label_encoder = None
    classes = None

    if problem_type == "Classification":
        # Ensure target is encoded to clean integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))
        classes = [str(c) for c in label_encoder.classes_]

        if len(classes) < 2:
            raise ValueError(f"Target variable has only 1 distinct class ({classes[0]}). Classification requires at least 2 classes.")

        # Check class distribution for stratification
        class_counts = pd.Series(y_encoded).value_counts()
        if (class_counts >= 2).all() and (len(class_counts) > 1):
            stratify_target = y_encoded
        else:
            stratify_target = None

        y_to_split = y_encoded
    else:
        # Regression: ensure target is numeric
        try:
            y_to_split = y.astype(float).values
        except Exception as e:
            raise ValueError(f"Selected target column '{target_col}' cannot be converted to numeric for regression: {e}")

    # Step 1: Split train and test data BEFORE fitting any preprocessor (No data leakage!)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y_to_split,
        clean_df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    # Step 2: Build ColumnTransformer
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No valid numeric or categorical columns found in the feature set.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Step 3: Fit preprocessor ONLY on X_train
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract feature names after encoding
    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        try:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(cat_features)
        except Exception:
            feature_names.extend(categorical_cols)

    # Package input defaults and ranges for the interactive prediction UI
    feature_info = {}
    for col in X.columns:
        if col in numeric_cols:
            col_series = X[col].dropna()
            min_val = float(col_series.min()) if not col_series.empty else 0.0
            max_val = float(col_series.max()) if not col_series.empty else 100.0
            mean_val = float(col_series.mean()) if not col_series.empty else 0.0
            median_val = float(col_series.median()) if not col_series.empty else 0.0
            feature_info[col] = {
                "type": "numeric",
                "min": min_val,
                "max": max_val,
                "default": median_val,
                "step": (max_val - min_val) / 100.0 if max_val != min_val else 1.0
            }
        else:
            unique_vals = X[col].dropna().unique().tolist()
            feature_info[col] = {
                "type": "categorical",
                "options": [str(v) for v in unique_vals] if unique_vals else ["Unknown"],
                "default": str(unique_vals[0]) if unique_vals else "Unknown"
            }

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "feature_info": feature_info,
        "target_col": target_col,
        "problem_type": problem_type,
        "label_encoder": label_encoder,
        "classes": classes
    }


def transform_single_input(input_dict: Dict[str, Any], preprocessor: ColumnTransformer, raw_columns: List[str]) -> np.ndarray:
    """
    Transforms a single user input instance into the preprocessed feature vector
    ready for model inference.
    """
    df_single = pd.DataFrame([input_dict], columns=raw_columns)
    return preprocessor.transform(df_single)
