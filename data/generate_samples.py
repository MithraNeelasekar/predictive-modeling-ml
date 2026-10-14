"""
generate_samples.py
Script to export sample datasets to CSV files in the data/ directory.
This ensures beginners can test CSV upload functionality offline immediately.
"""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

def generate_sample_datasets():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)

    # 1. Classification Dataset: Breast Cancer Wisconsin
    cancer = load_breast_cancer(as_frame=True)
    df_cancer = cancer.frame.copy()
    # Replace numeric target 0/1 with human-readable labels for beginner clarity
    # 0 = malignant, 1 = benign in sklearn breast cancer dataset
    df_cancer["target"] = df_cancer["target"].map({0: "Malignant", 1: "Benign"})
    cancer_path = os.path.join(data_dir, "breast_cancer.csv")
    df_cancer.to_csv(cancer_path, index=False)
    print(f"Classification dataset saved to: {cancer_path} ({df_cancer.shape[0]} rows, {df_cancer.shape[1]} cols)")

    # 2. Regression Dataset: Diabetes Progression
    diabetes = load_diabetes(as_frame=True)
    df_diabetes = diabetes.frame.copy()
    diabetes_path = os.path.join(data_dir, "diabetes_regression.csv")
    df_diabetes.to_csv(diabetes_path, index=False)
    print(f"Regression dataset saved to: {diabetes_path} ({df_diabetes.shape[0]} rows, {df_diabetes.shape[1]} cols)")

if __name__ == "__main__":
    generate_sample_datasets()
