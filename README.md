# Predictive Modeling Using Machine Learning

An end-to-end, interactive, and beginner-friendly Machine Learning web application built using **Python**, **Streamlit**, and **Scikit-Learn**. 

This project demonstrates the complete supervised machine learning lifecycle: exploratory data analysis, leakage-free automated preprocessing, multi-model training, comprehensive performance evaluation, real-time single-sample inference, and report exporting.

---

## 📌 Project Overview & Problem Statement

**Task:** Build a model to predict outcomes based on given data.

In real-world data science, choosing the best machine learning model requires comparing multiple algorithms, handling missing values and categorical data systematically without data leakage, and evaluating models using appropriate metrics rather than relying solely on raw accuracy.

This project delivers a complete, beginner-friendly web application where students and evaluators can:
1. Load built-in benchmark datasets (Breast Cancer for classification, Diabetes for regression) or upload custom CSV datasets.
2. Automatically inspect data quality, identify data types, and summarize missing values.
3. Train multiple standard supervised algorithms (Logistic/Linear Regression, Decision Trees, Random Forests).
4. Benchmark models side-by-side using Confusion Matrices, ROC-AUC curves, Actual vs. Predicted plots, and comparison tables.
5. Perform live predictions on custom user inputs with prediction probabilities.
6. Export performance reports and predictions to CSV.

---

## 🛠️ Technology Stack

| Component | Library / Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.14 | Core programming language |
| **Web UI** | Streamlit | Responsive, modern web interface |
| **Machine Learning** | Scikit-Learn | Preprocessing pipelines, algorithms, and evaluation metrics |
| **Data Processing** | Pandas & NumPy | Tabular data manipulation and numerical operations |
| **Visualization** | Matplotlib & Seaborn | Confusion matrices, ROC curves, and residual plots |

---

## 📂 Project Structure

```text
Intenship 2/
│
├── app.py                     # Main Streamlit web application
├── models.py                  # Model definitions, descriptions, and training logic
├── preprocessing.py           # Leakage-free preprocessing pipeline & data inspection
├── evaluation.py              # Performance metrics & Matplotlib/Seaborn plots
├── test_pipeline.py           # Automated test suite for end-to-end verification
├── requirements.txt           # Compatible project dependencies
├── .gitignore                 # Excludes caches, venvs, and generated files
├── README.md                  # Comprehensive project documentation
└── data/
    ├── generate_samples.py    # Script to re-generate sample datasets
    ├── breast_cancer.csv      # Wisconsin Breast Cancer classification dataset (569 rows)
    └── diabetes_regression.csv# Diabetes progression regression dataset (442 rows)
```

---

## ⚡ Installation & Execution Instructions

Follow these exact commands in your terminal or command prompt:

### Step 1: Open Terminal in Project Directory
```powershell
cd "c:\Users\MITHRA N\Desktop\Intenship 2"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Pipeline Tests
Verify that all algorithms, pipelines, and evaluation metrics function properly:
```powershell
python test_pipeline.py
```

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
After executing this command, Streamlit will open your browser automatically at:
```text
Local URL: http://localhost:8501
```

---

## 🧠 Machine Learning Fundamentals

### 1. Classification vs. Regression

| Feature | Classification | Regression |
| :--- | :--- | :--- |
| **Target Variable ($y$)** | Discrete categories or classes (e.g., Malignant vs. Benign, Yes vs. No) | Continuous numerical quantity (e.g., house price, blood pressure) |
| **Algorithms Implemented** | Logistic Regression, Decision Tree Classifier, Random Forest Classifier | Linear Regression, Decision Tree Regressor, Random Forest Regressor |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix | MAE, MSE, RMSE, R² Score, Actual vs. Predicted plot |
| **Primary Metric** | **F1-Score** (especially when classes are imbalanced) | **R² Score** / **RMSE** |

---

### 2. Supported Supervised Learning Algorithms

1. **Logistic Regression / Linear Regression:**
   - *Logistic Regression:* Fits a sigmoid function to output class probabilities. Fast, interpretable, and serves as an effective linear baseline.
   - *Linear Regression:* Fits Ordinary Least Squares (OLS) line minimizing the sum of squared residuals.
2. **Decision Trees (Classifier & Regressor):**
   - Non-parametric tree model that splits features hierarchically using information gain/Gini impurity (classification) or variance reduction (regression). Easily interpretable and handles non-linear relationships.
3. **Random Forests (Classifier & Regressor):**
   - Ensemble method (bagging) that trains an ensemble of decision trees on bootstrap samples with random feature selection. Averages predictions to significantly reduce overfitting and variance.

---

### 3. Data Leakage Prevention Explained

> **What is Data Leakage?**
> Data leakage occurs when information from the test dataset is inadvertently used to train or fit preprocessing transformations (such as computing the mean for missing value imputation or scaling features with min/max or mean/variance).
>
> **How this project prevents it:**
> 1. The dataset is split into `train` and `test` splits **first** using `train_test_split`.
> 2. The `ColumnTransformer` (with `SimpleImputer` and `StandardScaler`) is fitted **strictly on `X_train`** via `.fit_transform()`.
> 3. The test set `X_test` and any future live user inputs are processed strictly using `.transform()`.

---

## 📊 Comprehensive Metric Guide (Viva / Presentation Prep)

### Classification Metrics

- **Accuracy:** The ratio of correct predictions to total samples.
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  *Caution:* In imbalanced datasets (e.g., 95% healthy, 5% sick), a model predicting "Healthy" for every sample has 95% accuracy but is practically useless.
- **Precision:** Out of all cases predicted as positive, how many were truly positive?
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity):** Out of all actual positive cases, how many did the model identify?
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score:** The harmonic mean of Precision and Recall. Best overall metric for imbalanced classification:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix:** A $2 \times 2$ grid (or $N \times N$) displaying True Positives, False Positives, True Negatives, and False Negatives.
- **ROC Curve & AUC:** Plots True Positive Rate vs. False Positive Rate across all decision thresholds. AUC values range from $0.5$ (random guess) to $1.0$ (perfect discrimination).

### Regression Metrics

- **MAE (Mean Absolute Error):** Average magnitude of absolute errors. Easy to interpret in the original unit of the target.
- **MSE (Mean Squared Error):** Average of squared errors. Heavily penalizes large errors.
- **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}}$. Retains the penalty for large outliers while returning to original target units.
- **R² Score (Coefficient of Determination):** Proportion of variance in the target explained by the inputs. $1.0$ is perfect; $0.0$ indicates performance equivalent to predicting the mean.

---

## 💻 Sample Demonstration Workflow

1. **Step 1 — Launch the App:** Run `streamlit run app.py`.
2. **Step 2 — Explore Dataset:**
   - Click **Dataset Explorer** on the sidebar.
   - Choose the built-in **Breast Cancer (Classification)** dataset.
   - Review the 569 rows and 30 features. Notice target column is automatically set to `target`.
3. **Step 3 — Train Models:**
   - Click **Model Training** on the sidebar.
   - Keep Logistic Regression, Decision Tree Classifier, and Random Forest Classifier checked.
   - Click **🚀 Train Selected Models**. Watch the progress bar complete.
4. **Step 4 — Review Performance:**
   - Click **Performance Evaluation**.
   - Notice the **Model Comparison Table** and the **Top Performing Model** badge.
   - Inspect the **Confusion Matrix** tab and switch between models.
   - Check the **ROC & AUC Curves** tab to see the side-by-side curves.
5. **Step 5 — Make a Live Prediction:**
   - Click **Live Prediction**.
   - Adjust feature inputs or leave defaults.
   - Click **⚡ Generate Prediction** to see predicted class (`Benign` or `Malignant`) with confidence probabilities.
6. **Step 6 — Export Results:**
   - Click **Download Results** and download the CSV files for assignment documentation.

---

## 🐙 Step-by-Step GitHub Setup Guide

Follow these steps to upload your project to GitHub:

### 1. Initialize Git Repository
```powershell
git init
```

### 2. Stage and Commit All Files
```powershell
git add .
git commit -m "feat: complete predictive modeling machine learning web application"
```

### 3. Create a New Repository on GitHub
1. Log into your GitHub account at [https://github.com](https://github.com).
2. Click the **+** icon in the top right corner and select **New repository**.
3. Set the repository name to: `predictive-modeling-ml`
4. Set description to: `Interactive Machine Learning application for predictive modeling, model comparison, and evaluation using Streamlit and Scikit-Learn.`
5. Leave "Initialize this repository with a README" **unchecked** (we already have a complete README).
6. Click **Create repository**.

### 4. Link and Push to GitHub
Copy your repository URL from GitHub (replace `YOUR_USERNAME` with your actual GitHub username):
```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/predictive-modeling-ml.git
git push -u origin main
```

Your project will now be live on GitHub with complete documentation and clean version control.

---

## 👨‍🏫 Faculty Demonstration & Viva Q&A Guide

When presenting to your faculty or external examiner, follow this structured script:

### 1. Introduction (30 seconds)
> *"Good morning/afternoon Professors. My project is titled **Predictive Modeling Using Machine Learning**. The primary objective is to build an interactive system that demonstrates end-to-end supervised learning workflows, comparing linear algorithms, decision trees, and ensemble random forests on real-world datasets with rigorous evaluation."*

### 2. Technical Walkthrough (2 minutes)
- **Data Preprocessing:** Explain that to prevent data leakage, the train/test split is performed **before** any imputation or scaling. Mention `ColumnTransformer` and `StandardScaler`.
- **Model Comparison:** Show the **Performance Evaluation** tab. Highlight why Random Forest often achieves superior F1 and AUC scores due to ensemble bagging and variance reduction.
- **Evaluation Beyond Accuracy:** Explain that in critical applications (like medical diagnosis in the Breast Cancer dataset), False Negatives (missing a malignant tumor) are dangerous, so **Recall** and **F1-Score** are prioritized over raw Accuracy.
- **Inference:** Demonstrate the **Live Prediction** tab by modifying a feature and generating a prediction in real-time.

### 3. Anticipated Viva Questions & Answers

**Q1: Why did you split the dataset before scaling or imputing?**
*Answer:* Splitting the dataset first prevents **data leakage**. If we calculate the mean or variance using the entire dataset, information from the test set leaks into the training pipeline, yielding overly optimistic evaluation scores that will not generalize to unseen data.

**Q2: Why is Accuracy insufficient for imbalanced datasets?**
*Answer:* Accuracy treats all classes equally. If 98% of samples are Class A and 2% are Class B, a trivial model that always predicts Class A achieves 98% accuracy but 0% recall on Class B. F1-Score balances precision and recall, providing an honest assessment.

**Q3: How does a Random Forest improve upon a single Decision Tree?**
*Answer:* A single decision tree has high variance and is prone to overfitting training noise. Random Forest constructs an ensemble of diverse trees trained on bootstrap samples with random feature subsets. Aggregating their predictions (bagging) averages out individual errors and significantly lowers variance.

---

## ⚠️ Limitations & Future Work

- **Hyperparameter Tuning:** Currently uses sensible default parameters. Future extensions could add Grid Search or Bayesian Optimization directly in the UI.
- **Deep Learning Support:** Could be extended to include Multi-Layer Perceptrons (MLP) or PyTorch neural networks for complex non-linear tabular modeling.
- **Model Persistence:** Exporting trained models as serialized `.joblib` files for direct API deployment.
