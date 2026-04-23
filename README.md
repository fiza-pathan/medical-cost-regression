# Medical Insurance Cost Prediction - Regression Algorithm Showdown

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Healthcare-red)
![Type](https://img.shields.io/badge/Type-Regression-purple)

> Comparing Linear · Ridge · Lasso · ElasticNet · Polynomial Regression on real-world medical insurance data, with SHAP explainability and business insights.

**[View Kaggle Notebook](https://www.kaggle.com/code/fizapathan21/medical-insurance-cost-analysis-regression)** | **[View Portfolio]**

---

## Problem Statement

Medical insurance costs in the United States are difficult and opaque to predict. Insurers, hospitals, and policymakers need accurate cost models to allocate resources, price premiums fairly, and identify high-risk individuals for preventive care programmes.

This project answers:

> _Which regression model best predicts individual medical insurance costs, and why does it win for this specific data type and problem?_

This is not just a model comparison; it is a **study of when and why each algorithm is the right choice**, with direct business implications.

---

## Results Summary

| Model              | R² Score  | MAE (USD)  | RMSE (USD) | CV R² |
| ------------------ | --------- | ---------- | ---------- | ----- |
| Polynomial + Ridge | **0.893** | **$2,541** | **$4,287** | 0.881 |
| Ridge (L2)         | 0.871     | $2,790     | $4,801     | 0.869 |
| ElasticNet         | 0.869     | $2,812     | $4,834     | 0.867 |
| Lasso (L1)         | 0.866     | $2,843     | $4,901     | 0.864 |
| Linear Regression  | 0.862     | $2,901     | $4,978     | 0.860 |

**Winner: Polynomial Regression + Ridge** - captures the critical nonlinear smoker×BMI interaction that linear models miss.

---

## Key Findings

- **Smoker status is the dominant predictor** - smokers pay 3.8x more on average
- **The smoker×BMI interaction is nonlinear** - obese smokers cost ~4x more than healthy non-smokers; this is why Polynomial Regression wins
- **Feature engineering mattered more than model choice** - adding `smoker_bmi` and `age_squared` improved R² by 0.04 across all models
- **SHAP analysis confirmed** the engineered `smoker_bmi` feature as the top predictor, validating the EDA findings

---

## Algorithm Decision Guide

| Algorithm         | Use When                                                            |
| ----------------- | ------------------------------------------------------------------- |
| Linear Regression | Baseline, full interpretability needed, no multicollinearity        |
| Ridge (L2)        | Many correlated features, want to keep all predictors               |
| Lasso (L1)        | Automatic feature selection, sparse solutions needed                |
| ElasticNet        | Groups of correlated features, mixed L1+L2 needed                   |
| Polynomial        | EDA shows nonlinear relationships — always pair with regularisation |

---

## Project Structure

```
medical-cost-regression/
├── medical_cost_regression_showdown.ipynb   # Full analysis
├── outputs/
│   ├── 01_target_distribution.png
│   ├── 02_smoker_effect.png
│   ├── 03_age_bmi_scatter.png
│   ├── 04_categorical_analysis.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_interaction_effect.png
│   ├── 07_model_comparison.png
│   ├── 08_actual_vs_predicted.png
│   └── 09_shap_importance.png
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Tool                    | Purpose                                            |
| ----------------------- | -------------------------------------------------- |
| `pandas`, `numpy`       | Data manipulation                                  |
| `scikit-learn`          | All regression models, pipelines, cross-validation |
| `matplotlib`, `seaborn` | Visualisation                                      |
| `shap`                  | Model explainability                               |

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/medical-cost-regression.git
cd medical-cost-regression
pip install -r requirements.txt
jupyter notebook notebooks/medical_cost_regression_showdown.ipynb
```

Or run directly on Kaggle: **[Open Notebook](https://www.kaggle.com/code/fizapathan21/medical-insurance-cost-analysis-regression)**

---

## Dataset

- **Source:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle)
- **Size:** 1,338 records × 7 features
- **Target:** Individual annual medical insurance charges (USD)

---

## About

Built as part of my AI/ML engineering portfolio, working from ML fundamentals through to AI Research Engineer.

**[Portfolio](#)** · **[LinkedIn](https://www.linkedin.com/in/fizapathan/)** · **[GitHub](https://github.com/fiza-pathan)**
