# ML Assignment 2 — Classification Models and Streamlit App

## a. Problem Statement

Implement multiple classification models on one public classification dataset, evaluate them using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC), and demonstrate the models through an interactive Streamlit application.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository. The dataset is also packaged by scikit-learn as `load_breast_cancer()`.

- Instances: 569
- Features: 30 numerical features
- Target: binary classification
- Classes: 0 = malignant, 1 = benign
- Train/test split: 80/20
- Stratification: yes
- Random state: 42

## c. GitHub Repository Link

`https://github.com/SaiAkhilluffy/ML-Assignment`

## d. Models Used

The implementation follows the five named models in the assignment:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Evaluation Results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the strongest overall performance, with the highest Accuracy, AUC, Precision, Recall, F1 and MCC among the implemented models. Standardization helps the linear model work effectively with differently scaled features. |
| Decision Tree | Produced the lowest overall scores. The single tree can be sensitive to the training split and can overfit or underfit depending on its structure. |
| kNN | Performed strongly after feature standardization. Its Accuracy, Precision, Recall and F1 were close to Random Forest, but its AUC was lower. |
| Naive Bayes | Produced good results and a high AUC. Its conditional-independence assumption is restrictive, but it still works well on this numerical dataset. |
| Random Forest (Ensemble) | Performed strongly and achieved a very high AUC. Combining many decision trees improves robustness compared with a single decision tree. |

### Overall Winner

**Logistic Regression** is the overall winner for this train/test split because it achieved the best value for every reported metric.

## Streamlit Application

The application provides:

- CSV test-data upload
- Model-selection dropdown
- Predictions
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report

### Streamlit Link

`https://ml-assignment-2025ac05190.streamlit.app/`

## Repository Structure

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── breast_cancer_wisconsin_diagnostic.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── metadata.json
    ├── metrics_reference.csv
    └── train_models.py
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv`.

## Reproducibility

- `random_state = 42`
- Test size = 20%
- Stratified train/test split
- kNN uses `n_neighbors = 5`
- Random Forest uses 200 trees
- Logistic Regression and kNN use StandardScaler

