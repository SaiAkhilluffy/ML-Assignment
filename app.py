import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

LABEL_MAP = {
    0: "Malignant",
    1: "Benign"
}

@st.cache_resource
def load_model(filename):
    return joblib.load(
        os.path.join(MODEL_DIR, filename)
    )


st.title("🤖 Breast Cancer Classification Dashboard")

st.markdown(
    """
    Upload the test dataset, preview the data, select a machine learning
    model, and view its predictions and evaluation results.
    """
)

st.divider()

st.header("1️⃣ Upload Test Dataset")

uploaded = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Upload the test CSV file containing the 30 feature columns and target column."
)

if uploaded is None:

    st.info(
        "👆 Upload the supplied `test_data.csv` file to begin the analysis."
    )

    st.markdown(
        """
        **Expected dataset format**

        - 30 feature columns
        - `target` column
        - CSV format
        """
    )

    st.stop()

try:
    df = pd.read_csv(uploaded)

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV file: {e}"
    )

    st.stop()

st.header("2️⃣ Data Preview")

st.write(
    f"Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**."
)

st.markdown("**First 5 rows of the uploaded dataset:**")

st.dataframe(
    df.head(5),
    use_container_width=True
)

missing = [
    column
    for column in FEATURES
    if column not in df.columns
]

if missing:

    st.error(
        "The uploaded dataset is missing the following required "
        f"feature columns: {missing}"
    )

    st.stop()


has_target = "target" in df.columns


st.header("3️⃣ Select Machine Learning Model")

st.markdown(
    "Choose a model below to perform classification and display its analysis."
)

model_options = [
    "-- Select a model --"
] + list(MODEL_FILES.keys())


model_name = st.selectbox(
    "Model",
    model_options,
    index=0
)

if model_name == "-- Select a model --":

    st.info(
        "👆 Please select a machine learning model to view the analysis."
    )

    st.stop()

st.divider()

st.header(f"4️⃣ Analysis — {model_name}")


X_test = df[FEATURES].copy()



try:

    model = load_model(
        MODEL_FILES[model_name]
    )

except Exception as e:

    st.error(
        f"Unable to load the selected model: {e}"
    )

    st.stop()

try:

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:, 1]

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


st.subheader("🔮 Predictions")

result = df.copy()

result["Predicted Target"] = pred

result["Predicted Class"] = [
    LABEL_MAP.get(
        int(x),
        str(x)
    )
    for x in pred
]

result["Benign Probability"] = prob


st.dataframe(
    result.head(20),
    use_container_width=True
)


if has_target:

    y_true = df["target"].astype(int)


    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            pred
        ),

        "AUC": roc_auc_score(
            y_true,
            prob
        ),

        "Precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_true,
            pred
        ),
    }


    st.subheader("📊 Evaluation Metrics")

    cols = st.columns(6)

    for col, (name, value) in zip(
        cols,
        metrics.items()
    ):

        col.metric(
            name,
            f"{value:.4f}"
        )


    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        pred
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.imshow(cm)

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Malignant", "Benign"]
    )

    ax.set_yticklabels(
        ["Malignant", "Benign"]
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )


    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)


    st.subheader("📋 Classification Report")


    report = classification_report(
        y_true,
        pred,
        target_names=[
            "Malignant",
            "Benign"
        ],
        output_dict=True,
        zero_division=0
    )


    report_df = pd.DataFrame(
        report
    ).transpose()


    st.dataframe(
        report_df,
        use_container_width=True
    )


else:

    st.warning(
        "⚠️ No `target` column was found in the uploaded dataset. "
        "Predictions are available, but evaluation metrics cannot be calculated."
    )