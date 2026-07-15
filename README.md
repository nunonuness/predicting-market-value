# Predicting Footballers' Market Value from Performance Metrics

### A Position-Aware, Factual-Only Machine-Learning Approach

> **Master's Thesis — NOVA IMS, 2026**
> Author: Nuno Guilherme · <nunoguilherme8@gmail.com>

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red?logo=xgboost)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Methodology: CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-blueviolet)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Findings](#key-findings)
3. [Repository Structure](#repository-structure)
4. [Tech Stack](#tech-stack)
5. [Methodology — CRISP-DM Pipeline](#methodology--crisp-dm-pipeline)
6. [Models & Results](#models--results)
7. [Interpretability — SHAP & MDI](#interpretability--shap--mdi)
8. [Data Sources](#data-sources)
9. [How to Run](#how-to-run)
10. [Citation](#citation)

---

## Project Overview

This repository contains the full machine-learning pipeline underpinning the Master's thesis
*"Predicting Footballers' Market Value from Performance Metrics."*

The study tests whether a footballer's transfer-market value can be predicted from **factual
performance metrics alone** — counts and rates of events that actually occurred — deliberately
**excluding the expected-goal family** of model-derived statistics (xG, xAG, npxG, SCA, GCA)
that dominates contemporary football analytics.

A sample of **1,390 outfield player-seasons** from the five major European leagues (2022/23)
was assembled by merging **FBref** performance data with **Transfermarkt** valuations
(June 2023 snapshot). Sixteen models were trained and compared — four algorithm families
(ridge regression, random forest, XGBoost, and a multilayer perceptron) across a **global**
pipeline and three **position-specific** pipelines (forwards, midfielders, defenders) — under
a leakage-safe, uncertainty-aware protocol.

---

## Key Findings

| Finding | Value |
| --- | --- |
| **Best model (every pipeline)** | **Ridge regression** |
| **Global test R²** | **0.726** |
| **Forward test R²** | **0.736** |
| **Strongest non-linear family** | XGBoost |
| **Typical error (MAE, log scale)** | ≈ 0.49 (≈ 1.6× multiplicative) |
| **Dominant value drivers** | Age, ball involvement, league quality (Premier League premium), passing accuracy |

- A transparent **ridge regression beats the tree ensembles and the neural network in all four
  pipelines**: with factual inputs the performance–value relationship is largely additive and
  monotonic, leaving little for flexible models to exploit.
- **Splitting by position helps mainly for forwards**; midfielder and defender models do not beat
  the global benchmark, chiefly because of smaller per-position samples — motivating a **hybrid
  deployment** (global default + dedicated forward model).
- Findings challenge the assumption that ensemble methods universally outperform linear baselines,
  and show that **competitive accuracy is achievable without proprietary expected-goal inputs**.

---

## Repository Structure

```
predicting-market-value/
│
├── data/
│   └── thesis_dataset_2223.csv     # dataset used (merged FBref + Transfermarkt, 2022/23)
│
├── notebooks/
│   └── thesis_modelling.ipynb      # full pipeline: cleaning, features, models, evaluation
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Tech Stack

| Category | Library / Tool |
| --- | --- |
| **Data manipulation** | `pandas`, `numpy` |
| **Machine learning** | `scikit-learn`, `xgboost` |
| **Hyperparameter tuning** | `RandomizedSearchCV` (50 iterations, 5-fold) |
| **Explainability** | `shap` (TreeExplainer / TreeSHAP), mean decrease in impurity |
| **Visualisation** | `matplotlib` |
| **Environment** | Python 3.10+, Jupyter Notebook |

---

## Methodology — CRISP-DM Pipeline

The project follows the **CRISP-DM** framework across three phases.

### Phase 1 — Data Understanding & Preparation

- **Sources merged**: FBref (Opta/StatsBomb event data) × Transfermarkt valuations, linked by player identifier.
- **Scope**: Big Five leagues, 2022/23 season; outfield players only (goalkeepers excluded); market value = June 2023 snapshot.
- **Factual-only filtering**: the expected-goal family (xG, xAG, npxG, SCA, GCA) and exact redundancies are removed up front.
- **Unit conversion**: FBref distance metrics converted from yards to metres.
- **Correlation pruning**: highly correlated feature pairs (|r| > 0.85) reduced to the more informative member.
- **Outlier audit**: IQR screening confirms all extremes are legitimate elite performances — no rows removed or capped; skew handled by a log transform of the target.
- **Engineered age curve**: centred quadratic age term to capture the inverted-U value curve without collinearity.
- **Leakage barrier**: a stratified 80/20 train/test split is applied **before** any data-dependent transformation.

### Phase 2 — Data-Driven Feature Selection

- **Averaged permutation importance** across the four model families, normalised per model, gives a robust, algorithm-agnostic ranking.
- Features reaching a **2% retention threshold** are kept, with the age terms and the league indicator always retained on structural grounds.
- Produces a compact, **per-position feature subset** for each pipeline.

### Phase 3 — Modelling, Tuning & Evaluation

- **Sixteen models**: four families (ridge, random forest, XGBoost, MLP) × four pipelines (global + 3 positions).
- **Phase 1 tuning**: `RandomizedSearchCV`, 50 iterations, 5-fold CV, scored by RMSE.
- **Phase 2 estimation**: repeated **10×5 cross-validation** (50 evaluations) with **95% confidence intervals**.
- **Final test**: the best cross-validated model per pipeline is scored once on the **locked 20% hold-out**.
- **Interpretability**: mean decrease in impurity (MDI) and SHAP.

---

## Models & Results

Cross-validated performance on the **global pipeline** (10×5 repeated K-fold; metrics on the log scale):

| Model | R² | RMSE | MAE |
| --- | --- | --- | --- |
| **Ridge (champion)** | **0.705** | **0.657** | **0.505** |
| XGBoost | 0.688 | 0.676 | 0.526 |
| MLP | 0.686 | 0.677 | 0.523 |
| Random Forest | 0.645 | 0.721 | 0.571 |

Generalisation on the **locked 20% hold-out** (best model per pipeline — ridge in every case):

| Pipeline | Test R² |
| --- | --- |
| Global | 0.726 |
| Forward | 0.736 |
| Defender | 0.680 |
| Midfielder | 0.638 |

> Ridge is the champion on every pipeline. Test-set figures are fully consistent with
> cross-validation, with no sign of optimistic bias.

---

## Interpretability — SHAP & MDI

Model reasoning is opened with two complementary lenses:

- **Mean decrease in impurity (MDI)** summarises which features the random forest splits on most.
- **SHAP** (TreeExplainer) attributes each prediction to its features, revealing magnitude and direction.

Across positions the dominant drivers are the same economic fundamentals: the **age curve**,
total **ball involvement** (touches and minutes), **league quality** (a strong Premier League
premium) and **passing accuracy**. Position-specific plots localise these effects — goals per 90
and attacking-third touches for forwards, ball progression for midfielders.

---

## Data Sources

| Dataset | Provider | Access |
| --- | --- | --- |
| Player performance statistics | FBref (Opta / StatsBomb) | https://fbref.com |
| Market valuations | Transfermarkt | https://www.transfermarkt.com |

> `data/thesis_dataset_2223.csv` is a **derived, processed feature table** produced for this
> study by merging and cleaning the sources above. The original providers retain their rights
> over the underlying data; the file is shared for research and reproducibility purposes only.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/predicting-market-value.git
cd predicting-market-value
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
jupyter notebook notebooks/thesis_modelling.ipynb
```

The notebook reads `data/thesis_dataset_2223.csv` and reproduces the full pipeline: cleaning,
feature engineering, data-driven feature selection, the sixteen models, the two-phase evaluation
and the interpretability analysis.

---

## Citation

If you use this code or build upon this research, please cite:

```bibtex
@mastersthesis{nuno2026marketvalue,
  author  = {Nuno Guilherme},
  title   = {Predicting Footballers' Market Value from Performance Metrics},
  school  = {NOVA Information Management School (NOVA IMS)},
  year    = {2026},
  address = {Lisbon, Portugal}
}
```

---

*Built with public FBref and Transfermarkt data · NOVA IMS · 2026*
