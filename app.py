"""
app.py
Main Streamlit application for "Predictive Modeling Using Machine Learning".
A complete, beginner-friendly, and interactive ML dashboard.
"""

import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Import custom modular components
from preprocessing import (
    inspect_dataframe,
    detect_problem_type,
    prepare_dataset,
    transform_single_input
)
from models import (
    get_available_models,
    train_models,
    MODEL_DESCRIPTIONS
)
from evaluation import (
    build_comparison_summary,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_actual_vs_predicted,
    plot_residuals,
    METRIC_EXPLANATIONS
)

# Page configuration
st.set_page_config(
    page_title="Predictive Modeling Using Machine Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .best-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .header-style {
        color: #1e3a8a;
        font-weight: 700;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables if not already present
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "None"
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None
if "problem_type" not in st.session_state:
    st.session_state["problem_type"] = "Classification"
if "prep_data" not in st.session_state:
    st.session_state["prep_data"] = None
if "trained_results" not in st.session_state:
    st.session_state["trained_results"] = None
if "comparison_df" not in st.session_state:
    st.session_state["comparison_df"] = None
if "best_model" not in st.session_state:
    st.session_state["best_model"] = None
if "eval_cache" not in st.session_state:
    st.session_state["eval_cache"] = None


# Helper to load sample dataset from disk
def load_sample_dataset(dataset_choice: str) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset_choice == "Breast Cancer (Classification)":
        csv_path = os.path.join(base_dir, "data", "breast_cancer.csv")
    else:
        csv_path = os.path.join(base_dir, "data", "diabetes_regression.csv")
    return pd.read_csv(csv_path)


# Sidebar Navigation
st.sidebar.title("🤖 Navigation")
nav_selection = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Explorer",
        "⚙️ Model Training",
        "📈 Performance Evaluation",
        "🔮 Live Prediction",
        "📥 Download Results"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🎓 **Predictive Modeling Project**\nSupervised Machine Learning System")
if st.session_state["dataset"] is not None:
    st.sidebar.success(f"📂 Dataset: **{st.session_state['dataset_name']}**")
    if st.session_state["target_col"]:
        st.sidebar.info(f"🎯 Target: **{st.session_state['target_col']}** ({st.session_state['problem_type']})")
    if st.session_state["trained_results"]:
        st.sidebar.success(f"🏆 Best Model: **{st.session_state['best_model']}**")


# ==============================================================================
# 1. HOME SECTION
# ==============================================================================
if nav_selection == "🏠 Home":
    st.title("🎯 Predictive Modeling Using Machine Learning")
    st.subheader("An Interactive Supervised Learning Application for College Demonstration")

    st.markdown("""
    Welcome to the **Predictive Modeling** workbench. This application enables students, faculty, and developers to explore datasets, train multiple supervised machine learning algorithms, rigorously evaluate performance with modern metrics, and interactively generate real-time predictions.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📘 What is Supervised Learning?")
        st.info("""
        **Supervised Machine Learning** involves training an algorithm on historical input data (features) mapped to known outcomes (the target label). Once trained, the model learns the underlying mapping function:

        $$\\hat{y} = f(X)$$

        The two primary branches of Supervised Learning are:
        - **Classification:** Predicts discrete categories (e.g., *Malignant vs. Benign*, *Spam vs. Ham*).
        - **Regression:** Predicts continuous numerical quantities (e.g., *House Price*, *Disease Progression Score*).
        """)

    with col2:
        st.markdown("### 🚀 Project Key Features")
        st.success("""
        - **Leakage-Free Automated Preprocessing:** Splits train/test data first, then handles imputation and encoding strictly on training data.
        - **Multiple Model Comparison:** Benchmarks Logistic/Linear Regression, Decision Trees, and Random Forests side-by-side.
        - **Visual Performance Diagnostics:** Generates Confusion Matrices, ROC-AUC Curves, and Actual vs. Predicted plots.
        - **Class Imbalance Awareness:** Highlights why accuracy can be misleading and emphasizes F1-Score for classification.
        - **Live Inference Engine:** Enter custom feature values to test immediate predictions with confidence scores.
        - **One-Click Export:** Download performance tables and predictions as CSV files.
        """)

    st.markdown("---")
    st.markdown("### 📋 Recommended Step-by-Step Workflow")
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)

    with step_col1:
        st.markdown("""
        #### 1. Explore Data
        Navigate to **Dataset Explorer**, select a sample dataset (e.g., Breast Cancer) or upload your own CSV, review statistics, and pick the target column.
        """)

    with step_col2:
        st.markdown("""
        #### 2. Train Models
        Head to **Model Training**, choose algorithms, set the train/test ratio (default 80/20), and train all selected models with one click.
        """)

    with step_col3:
        st.markdown("""
        #### 3. Evaluate Metrics
        Inspect **Performance Evaluation** to compare metrics, see the top-performing model, and view diagnostic charts.
        """)

    with step_col4:
        st.markdown("""
        #### 4. Live Prediction
        Test the trained model in **Live Prediction** with custom values, or export results in **Download Results**.
        """)


# ==============================================================================
# 2. DATASET EXPLORER
# ==============================================================================
elif nav_selection == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.write("Inspect, validate, and configure the dataset for supervised learning.")

    data_source = st.radio(
        "Choose Data Source:",
        ["Use Built-in Sample Dataset", "Upload Custom CSV File"],
        horizontal=True
    )

    df_loaded = None
    dataset_name = "None"

    if data_source == "Use Built-in Sample Dataset":
        sample_choice = st.selectbox(
            "Select Built-in Dataset:",
            ["Breast Cancer (Classification)", "Diabetes Progression (Regression)"]
        )
        try:
            df_loaded = load_sample_dataset(sample_choice)
            dataset_name = sample_choice
        except Exception as e:
            st.error(f"Error loading sample dataset: {e}")
    else:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                df_loaded = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name
            except Exception as e:
                st.error(f"Failed to read CSV file: {e}")

    if df_loaded is not None:
        st.session_state["dataset"] = df_loaded
        st.session_state["dataset_name"] = dataset_name

        st.success(f"Successfully loaded **{dataset_name}** with **{df_loaded.shape[0]} rows** and **{df_loaded.shape[1]} columns**.")

        # Inspection summary
        info = inspect_dataframe(df_loaded)

        # Dataset metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{info['num_rows']:,}")
        m2.metric("Total Columns", info["num_cols"])
        m3.metric("Numeric Features", len(info["numeric_columns"]))
        m4.metric("Categorical Features", len(info["categorical_columns"]))

        # Preview Data Table
        st.markdown("### 🔍 Dataset Preview")
        st.dataframe(df_loaded.head(10), use_container_width=True)

        # Target Column Selection
        st.markdown("### 🎯 Target Variable Selection")
        col_t1, col_t2 = st.columns([2, 1])

        default_target_idx = 0
        if "target" in df_loaded.columns:
            default_target_idx = df_loaded.columns.get_loc("target")

        with col_t1:
            selected_target = st.selectbox(
                "Select the Target (Outcome) Column:",
                options=df_loaded.columns,
                index=default_target_idx,
                help="This is the dependent variable (Y) that the machine learning models will learn to predict."
            )

        # Inferred problem type
        inferred_type = detect_problem_type(df_loaded, selected_target)

        with col_t2:
            problem_type_override = st.selectbox(
                "Problem Type:",
                options=["Classification", "Regression"],
                index=0 if inferred_type == "Classification" else 1,
                help="Auto-detected based on the data type and unique value distribution of the target variable."
            )

        st.session_state["target_col"] = selected_target
        st.session_state["problem_type"] = problem_type_override

        # Target distribution preview
        st.markdown("#### Target Distribution")
        target_series = df_loaded[selected_target].dropna()
        if problem_type_override == "Classification":
            val_counts = target_series.value_counts(dropna=False).reset_index()
            val_counts.columns = ["Class", "Count"]
            st.bar_chart(val_counts.set_index("Class"))
        else:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.hist(target_series, bins=25, color="#3b82f6", edgecolor="black", alpha=0.7)
            ax.set_title(f"Target Distribution: {selected_target}")
            ax.set_xlabel(selected_target)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close(fig)

        # Data Quality & Missing Values
        with st.expander("🔎 Missing Values and Data Types Summary", expanded=False):
            st.dataframe(info["missing_summary"], use_container_width=True)

        # Descriptive Statistics
        with st.expander("📈 Descriptive Statistics (Numerical)", expanded=False):
            if not info["summary_statistics"].empty:
                st.dataframe(info["summary_statistics"], use_container_width=True)
            else:
                st.info("No numerical features available for descriptive statistics.")

    else:
        st.info("👆 Please select a sample dataset above or upload your own CSV file to get started.")


# ==============================================================================
# 3. MODEL TRAINING
# ==============================================================================
elif nav_selection == "⚙️ Model Training":
    st.title("⚙️ Model Training & Pipeline Configuration")

    if st.session_state["dataset"] is None or st.session_state["target_col"] is None:
        st.warning("⚠️ No dataset or target variable configured yet. Please visit the **Dataset Explorer** tab first.")
    else:
        df = st.session_state["dataset"]
        target_col = st.session_state["target_col"]
        problem_type = st.session_state["problem_type"]

        st.info(f"Target Variable: **{target_col}** | Problem Type: **{problem_type}** | Total Rows: **{len(df)}**")

        st.markdown("### 1. Select Algorithms to Train")
        available_models = get_available_models(problem_type)

        selected_models = []
        model_cols = st.columns(len(available_models))
        for i, model_name in enumerate(available_models):
            with model_cols[i]:
                checked = st.checkbox(model_name, value=True, key=f"chk_{model_name}")
                if checked:
                    selected_models.append(model_name)
                # Show description tooltip/caption
                desc = MODEL_DESCRIPTIONS.get(model_name, {})
                st.caption(desc.get("description", ""))

        st.markdown("### 2. Configure Training Hyperparameters")
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider(
                "Test Split Ratio:",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                help="Proportion of the dataset reserved for testing (e.g. 0.20 = 80% train, 20% test)."
            )
        with c2:
            random_seed = st.number_input(
                "Random Seed (Reproducibility):",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                help="Fixing the random seed ensures train/test splits and stochastic algorithms yield identical results on repeated runs."
            )

        st.markdown("---")
        train_btn = st.button("🚀 Train Selected Models", type="primary", use_container_width=True)

        if train_btn:
            if not selected_models:
                st.error("Please select at least one algorithm to train.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: Preprocessing (Leakage-Free)
                    status_text.text("Step 1/3: Splitting dataset and fitting leakage-free preprocessing pipeline...")
                    progress_bar.progress(25)
                    time.sleep(0.3)

                    prep = prepare_dataset(
                        df,
                        target_col=target_col,
                        problem_type=problem_type,
                        test_size=test_size,
                        random_state=random_seed
                    )
                    st.session_state["prep_data"] = prep
                    progress_bar.progress(50)

                    # Step 2: Model Training
                    status_text.text(f"Step 2/3: Training {len(selected_models)} selected model(s)...")
                    progress_bar.progress(70)

                    trained_res = train_models(
                        selected_models,
                        problem_type,
                        prep["X_train_proc"],
                        prep["y_train"],
                        prep["X_test_proc"],
                        random_state=random_seed
                    )
                    st.session_state["trained_results"] = trained_res
                    progress_bar.progress(90)

                    # Step 3: Performance Comparison
                    status_text.text("Step 3/3: Evaluating metrics and determining top model...")
                    comp_df, best_model, eval_cache = build_comparison_summary(
                        trained_res,
                        problem_type,
                        prep["y_test"]
                    )
                    st.session_state["comparison_df"] = comp_df
                    st.session_state["best_model"] = best_model
                    st.session_state["eval_cache"] = eval_cache

                    progress_bar.progress(100)
                    status_text.empty()
                    st.success(f"🎉 Successfully trained {len(selected_models)} model(s)! Best Model: **{best_model}**")

                    st.balloons()

                    # Quick summary metrics
                    st.markdown("### 📊 Quick Training Summary")
                    st.dataframe(comp_df, use_container_width=True)
                    st.info("💡 Head over to the **Performance Evaluation** tab to view detailed charts, confusion matrices, and ROC curves!")

                except Exception as ex:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Training failed: {ex}")


# ==============================================================================
# 4. PERFORMANCE EVALUATION
# ==============================================================================
elif nav_selection == "📈 Performance Evaluation":
    st.title("📈 Model Performance & Evaluation")

    if st.session_state["trained_results"] is None or st.session_state["comparison_df"] is None:
        st.warning("⚠️ No trained models found. Please train models in the **Model Training** tab first.")
    else:
        comp_df = st.session_state["comparison_df"]
        best_model = st.session_state["best_model"]
        problem_type = st.session_state["problem_type"]
        prep = st.session_state["prep_data"]
        trained_res = st.session_state["trained_results"]
        eval_cache = st.session_state["eval_cache"]

        # Best Model Callout Card
        st.markdown(f"""
        <div class="metric-card">
            <span class="best-badge">🏆 TOP PERFORMING MODEL</span>
            <h2 style="margin: 8px 0 4px 0; color: #1e3a8a;">{best_model}</h2>
            <p style="margin: 0; color: #4b5563;">
                Selected as the leading model based on <b>{'F1-Score' if problem_type == 'Classification' else 'R² Score'}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if problem_type == "Classification":
            st.warning("""
            ⚠️ **Important Evaluation Concept (Class Imbalance):**
            Do not rely solely on **Accuracy** when evaluating classification models on real-world datasets!
            If 95% of patients are healthy and 5% have a disease, a naive model that predicts "Healthy" 100% of the time will achieve 95% accuracy while completely failing to detect the sick patients.
            Always evaluate **Precision**, **Recall**, and **F1-Score** alongside Accuracy.
            """)

        # Model Comparison Table
        st.markdown("### 📋 Model Comparison Table")
        st.dataframe(comp_df, use_container_width=True)

        # Performance Visualizations
        st.markdown("---")
        st.markdown("### 📊 Visual Diagnostics")

        if problem_type == "Classification":
            viz_tab1, viz_tab2 = st.tabs(["🔲 Confusion Matrices", "📈 ROC & AUC Curves"])

            with viz_tab1:
                st.write("A **Confusion Matrix** visualizes correct and incorrect predictions for each class.")
                model_to_plot = st.selectbox(
                    "Select Model for Confusion Matrix:",
                    options=list(trained_res.keys()),
                    index=list(trained_res.keys()).index(best_model)
                )

                cm = eval_cache[model_to_plot]["confusion_matrix"]
                fig_cm = plot_confusion_matrix(cm, prep["classes"], model_to_plot)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

                # Show breakdown
                if len(prep["classes"]) == 2:
                    tn, fp, fn, tp = cm.ravel()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("True Negatives (TN)", tn)
                    c2.metric("False Positives (FP)", fp)
                    c3.metric("False Negatives (FN)", fn)
                    c4.metric("True Positives (TP)", tp)

            with viz_tab2:
                st.write("The **ROC Curve** (Receiver Operating Characteristic) illustrates the trade-off between True Positive Rate and False Positive Rate across classification thresholds.")
                fig_roc = plot_roc_curves(trained_res, prep["y_test"], prep["classes"])
                if fig_roc is not None:
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)
                else:
                    st.info("ROC Curve is available for binary classification models with probability predictions.")

        else:
            # Regression Visualizations
            reg_tab1, reg_tab2 = st.tabs(["🎯 Actual vs. Predicted", "📉 Residuals Distribution"])

            with reg_tab1:
                st.write("Scatter plot comparing actual target values with predicted values. Points close to the dashed red line indicate accurate predictions.")
                model_to_plot = st.selectbox(
                    "Select Model for Actual vs. Predicted:",
                    options=list(trained_res.keys()),
                    index=list(trained_res.keys()).index(best_model)
                )

                y_pred = trained_res[model_to_plot]["y_pred"]
                fig_reg = plot_actual_vs_predicted(prep["y_test"], y_pred, model_to_plot)
                st.pyplot(fig_reg)
                plt.close(fig_reg)

            with reg_tab2:
                st.write("Distribution of residual errors ($Actual - Predicted$). A well-fitted model displays residuals centered symmetrically around zero.")
                model_to_plot_res = st.selectbox(
                    "Select Model for Residuals:",
                    options=list(trained_res.keys()),
                    index=list(trained_res.keys()).index(best_model),
                    key="sb_residuals"
                )
                y_pred_res = trained_res[model_to_plot_res]["y_pred"]
                fig_res = plot_residuals(prep["y_test"], y_pred_res, model_to_plot_res)
                st.pyplot(fig_res)
                plt.close(fig_res)

        # Plain-English Metric Reference Guide
        st.markdown("---")
        with st.expander("📖 College Viva Guide: Plain-English Metric Definitions"):
            for metric_name, explanation in METRIC_EXPLANATIONS.items():
                st.markdown(f"**{metric_name}:** {explanation}")


# ==============================================================================
# 5. LIVE PREDICTION
# ==============================================================================
elif nav_selection == "🔮 Live Prediction":
    st.title("🔮 Live Model Prediction")
    st.write("Input custom feature values and generate real-time predictions using any trained model.")

    if st.session_state["trained_results"] is None or st.session_state["prep_data"] is None:
        st.warning("⚠️ Please train models first before making predictions.")
    else:
        prep = st.session_state["prep_data"]
        trained_res = st.session_state["trained_results"]
        best_model = st.session_state["best_model"]
        problem_type = st.session_state["problem_type"]
        raw_cols = prep["X_train_raw"].columns.tolist()
        feature_info = prep["feature_info"]

        # Model Selector
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            chosen_model_name = st.selectbox(
                "Select Model for Prediction:",
                options=list(trained_res.keys()),
                index=list(trained_res.keys()).index(best_model),
                help="Defaults to the top-performing model."
            )

        st.markdown("### Enter Feature Values")
        user_inputs = {}

        # Render input form in responsive columns
        cols_per_row = 3
        form_cols = st.columns(cols_per_row)

        for i, col in enumerate(raw_cols):
            with form_cols[i % cols_per_row]:
                info = feature_info.get(col, {})
                if info.get("type") == "numeric":
                    min_val = float(info["min"])
                    max_val = float(info["max"])
                    default_val = float(info["default"])
                    step_val = float(info["step"]) if info["step"] > 0 else 0.1
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        min_value=min_val - (abs(min_val) * 0.5 + 1.0),
                        max_value=max_val + (abs(max_val) * 0.5 + 1.0),
                        value=default_val,
                        step=step_val,
                        format="%.3f"
                    )
                else:
                    options = info.get("options", ["Unknown"])
                    user_inputs[col] = st.selectbox(
                        f"{col}",
                        options=options,
                        index=0
                    )

        st.markdown("---")
        predict_btn = st.button("⚡ Generate Prediction", type="primary", use_container_width=True)

        if predict_btn:
            try:
                # Preprocess input sample with fitted pipeline
                input_vec = transform_single_input(user_inputs, prep["preprocessor"], raw_cols)
                model_obj = trained_res[chosen_model_name]["model"]

                if problem_type == "Classification":
                    pred_class_idx = model_obj.predict(input_vec)[0]
                    pred_label = prep["label_encoder"].inverse_transform([pred_class_idx])[0]

                    st.markdown("### 🎯 Prediction Result")
                    st.success(f"Predicted Class: **{pred_label}** (using {chosen_model_name})")

                    # Probabilities if available
                    if hasattr(model_obj, "predict_proba"):
                        proba_vals = model_obj.predict_proba(input_vec)[0]
                        proba_df = pd.DataFrame({
                            "Class": prep["classes"],
                            "Probability": [f"{p*100:.2f}%" for p in proba_vals]
                        })
                        st.write("#### Prediction Confidence:")
                        st.dataframe(proba_df, use_container_width=True)

                else:
                    # Regression
                    pred_val = model_obj.predict(input_vec)[0]
                    st.markdown("### 🎯 Prediction Result")
                    st.success(f"Predicted Target Value: **{pred_val:.4f}** (using {chosen_model_name})")

            except Exception as e:
                st.error(f"Inference error: {e}")


# ==============================================================================
# 6. DOWNLOAD RESULTS
# ==============================================================================
elif nav_selection == "📥 Download Results":
    st.title("📥 Download Results & Reports")
    st.write("Export your model performance comparisons and test set predictions as CSV files for your project submission.")

    if st.session_state["trained_results"] is None or st.session_state["comparison_df"] is None:
        st.warning("⚠️ No trained models or evaluation results available. Please train models first.")
    else:
        comp_df = st.session_state["comparison_df"]
        prep = st.session_state["prep_data"]
        trained_res = st.session_state["trained_results"]
        problem_type = st.session_state["problem_type"]

        st.markdown("### 1. Model Comparison Metrics Table")
        st.dataframe(comp_df, use_container_width=True)

        csv_comp = comp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Comparison Metrics (CSV)",
            data=csv_comp,
            file_name="model_comparison_metrics.csv",
            mime="text/csv",
            type="primary"
        )

        st.markdown("---")
        st.markdown("### 2. Test Set Predictions Table")

        # Compile actual vs predicted for test instances
        test_df = prep["X_test_raw"].copy()
        if problem_type == "Classification":
            test_df["Actual"] = prep["label_encoder"].inverse_transform(prep["y_test"])
            for m_name, res in trained_res.items():
                test_df[f"Pred_{m_name}"] = prep["label_encoder"].inverse_transform(res["y_pred"])
        else:
            test_df["Actual"] = prep["y_test"]
            for m_name, res in trained_res.items():
                test_df[f"Pred_{m_name}"] = res["y_pred"]

        st.dataframe(test_df.head(15), use_container_width=True)

        csv_test = test_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Test Predictions (CSV)",
            data=csv_test,
            file_name="test_predictions_results.csv",
            mime="text/csv"
        )
