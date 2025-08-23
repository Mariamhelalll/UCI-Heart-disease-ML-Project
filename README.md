# UCI-Heart-disease-ML-Project
# 🫀 UCI Heart Disease Risk Prediction — Machine Learning Project

> Predicting risk of heart disease from clinical features using classic ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost) with a focus on **interpretability** and **healthcare relevance**.

---

## 📌 Overview
This project explores how **machine learning** can support **clinical decision‑making** by predicting the presence of heart disease using the **UCI Heart Disease** dataset. It demonstrates a complete workflow: **data cleaning → EDA → model training → evaluation → interpretability** (feature importance/SHAP), presented in a way that clinicians and non‑ML stakeholders can understand.

**Key outcomes**
- End‑to‑end, reproducible ML pipeline in notebooks.
- Transparent evaluation across multiple models (not just accuracy).
- Clinically sensible insights highlighting known cardiovascular risk factors.

---

## 🎯 Objectives
- Build a reliable preprocessing pipeline (missing values, encoding, scaling).
- Explore clinical patterns with concise EDA and visuals.
- Train and compare baseline (LogReg/DT) vs ensemble models (RF/XGB).
- Evaluate with **Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC**.
- Explain predictions with **feature importance** and (optionally) **SHAP**.
- Discuss **clinical relevance, risks, and responsible use**.

---

## 📊 Dataset
- **Source**: UCI Machine Learning Repository — Heart Disease  
  URL: https://archive.ics.uci.edu/ml/datasets/heart+disease
- **Task**: Binary classification — `target` indicates presence of heart disease.
- **Features (examples)**: `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `fbs` (fasting blood sugar), `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.

> ⚠️ **Note**: Some UCI variants contain missing or coded values (e.g., `?`). This repo documents how those are handled for reproducibility.

---

## 🧰 Tech Stack
- **Python**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
- **Optional**: `shap` for model interpretability
- **Environment**: Python 3.10+ recommended

---

## 🚀 Quickstart

### 1) Clone and create environment
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
2) Get the data

Either download the UCI dataset and place it in data/, or load it directly in the notebooks (links provided inside).

3) Run notebooks (recommended path)

Open in Jupyter/Colab and execute in order:

notebooks/01_preprocessing.ipynb

notebooks/02_eda.ipynb

notebooks/03_models.ipynb

notebooks/04_interpretability.ipynb (optional)

Colab link (replace with yours): Open in Colab

🧪 Methods
Preprocessing

Handle missing values / invalid codes

One‑hot encode categorical features

Scale numerical features (where appropriate)

Train/Validation/Test split with fixed random seed for reproducibility

Models compared

Logistic Regression (baseline, interpretable)

Decision Tree (non‑linear baseline)

Random Forest (bagging ensemble)

XGBoost (boosting ensemble)

Evaluation

Metrics: Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC

Visuals: ROC curve, PR curve, confusion matrix

Robustness: Stratified K‑Fold (optional)

In most runs, Random Forest / XGBoost outperform linear models, likely due to capturing non‑linear interactions between clinical variables.

Sample visuals (replace links with your image paths):

ROC Curve: results/roc_curve.png

PR Curve: results/pr_curve.png

Confusion Matrix: results/confusion_matrix.png

Feature Importance: results/feature_importance.png

🔎 Interpretability & Clinical Relevance

Feature importance often surfaces: chest pain type (cp), cholesterol (chol), resting blood pressure (trestbps), max heart rate (thalach), and ST depression (oldpeak) as influential.

(Optional) SHAP values provide local explanations for individual predictions.

🩺 Why this matters: Interpretability builds trust with clinicians and supports safer decision support tools.

🧭 Repository Structure
.
├── data/                      # (optional) local data storage
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_models.ipynb
│   └── 04_interpretability.ipynb
├── results/                   # saved plots & metrics
├── src/                       # (optional) reusable helpers (preprocess, metrics, viz)
├── requirements.txt
└── README.md

🔁 Reproducibility

Fixed random seeds for splits/model init (documented in notebooks).

Environment pinned in requirements.txt.

Data source and preprocessing steps fully documented.

✅ What I Learned (Interview‑Ready Talking Points)

Trade‑offs: Simplicity/interpretability (LogReg) vs performance (ensembles).

Metrics beyond accuracy are crucial for clinical risk problems (Recall, ROC/PR‑AUC).

Data quality matters: missing values and encoded categories can shift results.

Explainability builds clinician trust; feature importance/SHAP helps.

Responsible AI: models should assist, not replace, clinical judgement.

🧠 ELI5 Summary (for non‑technical stakeholders)

“I cleaned and explored a heart health dataset, tested different ML algorithms to predict who might have heart disease, and checked them with fair metrics. The best models captured complex patterns and highlighted known risk factors. I also explained why the models make certain predictions to keep them clinically trustworthy.”

⚖️ Ethics & Limitations

Not for clinical use — dataset is limited and may not generalise to all populations.

Potential bias from sampling or feature coding.

Future work: larger, diverse datasets; external validation; calibration; clinician feedback.

🗺️ Roadmap / Next Steps

Hyperparameter tuning & cross‑validation grid

Calibration (e.g., Platt/Isotonic) to improve probability estimates

SHAP dashboards for case‑by‑case explanations

Lightweight Streamlit or Power BI dashboard for demo

Extended comparison with calibrated LogReg, LightGBM, CatBoost

📎 Links

Colab Notebook: <ADD_YOUR_COLAB_LINK>

Project Slides (5‑min): results/heart_disease_project_deck.pdf

Issue Tracker: Please use GitHub Issues for bugs/feature requests.

📜 Citation

If you use this work, please cite the UCI dataset and this repository:

@misc{uci_heart_disease,
  title  = {Heart Disease Data Set},
  author = {Dheeru Dua and Casey Graff},
  year   = {2019},
  url    = {https://archive.ics.uci.edu/ml/datasets/heart+disease}
}

🧑‍💻 Author

Maryam Mohamed — Computer Science & AI undergraduate, passionate about Healthcare Informatics

LinkedIn: https://www.linkedin.com/in/maryam-mohamed-9678031b1

GitHub: https://github.com/Mariamhelalll
