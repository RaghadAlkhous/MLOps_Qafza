# MLOps Training 2026/2027 — Olist Delivery Prediction

This repository contains the implementation of the MLOps training project using the **Brazilian E-Commerce Public Dataset by Olist**.

The project covers the complete workflow from relational data ingestion and database setup to machine learning for **late-delivery prediction**.

---

## Project Overview

The main objective is to build a reproducible ML pipeline that predicts whether an order will be delivered **Late** or **On Time**.

The workflow is divided into two main parts:

* **Task 1:** Data ingestion, relational database design, and validation.
* **Task 2:** Data preparation, exploratory analysis, feature engineering, model training, tuning, and evaluation.

---

## Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**, containing information about:

* Customers
* Orders
* Products
* Sellers
* Payments
* Reviews
* Geolocation

The dataset consists of 9 CSV files and is stored locally under:

```text
data/raw/
```

Raw data is excluded from version control.

---

## Task 1 — Data Ingestion & Database Setup

Task 1 establishes the data foundation using **PostgreSQL** and **Docker**.

### Database

| Parameter  | Configuration    |
| ---------- | ---------------- |
| Database   | `olist_db`       |
| User       | `olist_user`     |
| PostgreSQL | `16`             |
| Port       | `5432`           |
| Container  | `olist-postgres` |

The relational schema is defined in:

```text
sql/schema.sql
```

The database was validated through row-count checks, relationship checks, aggregations, and JOIN queries.

The final order-level ML table contains:

* **99,441 orders**
* **99,441 unique `order_id` values**
* One row per order

---

## Task 2 — Late Delivery Prediction

Task 2 builds the machine learning pipeline on top of the database output.

### Pipeline

```text
PostgreSQL
    ↓
Notebook 01 — Read & Join
    ↓
Notebook 02 — Build Label
    ↓
Notebook 03 — Split Data
    ↓
Notebook 04 — EDA
    ↓
Notebook 05 — Feature Engineering
    ↓
Notebook 06 — Train, Tune & Evaluate
```

### Target

The target variable is:

```text
late
```

* `1` → delivered after the estimated delivery date
* `0` → delivered on or before the estimated delivery date

The prediction point is **order approval time**, so features unavailable at that point are excluded to prevent leakage.

### Data Split

The labeled data is divided into:

* Train: **70%**
* Validation: **15%**
* Test: **15%**

The split is stratified by `late` with `random_state = 42`.

### Models

The following models are evaluated:

* Dummy Classifier
* Logistic Regression
* Random Forest

Because late deliveries represent only about **8.11%** of labeled orders, **Average Precision (PR-AUC)** is used as the primary evaluation metric.

---

## Final Model

The selected model is a **Random Forest Classifier** tuned using the validation set.

Final test performance:

| Metric            |      Test |
| ----------------- | --------: |
| Average Precision | **0.287** |
| ROC-AUC           | **0.790** |
| Precision — Late  | **0.352** |
| Recall — Late     | **0.322** |
| F1-score — Late   | **0.336** |
| Accuracy          | **0.897** |

The test set was used only for the final evaluation and was not used for model selection or tuning.

---

## Project Structure

```text
MLOps_Olist_Delivery_Prediction/
│
├── data/
│   └── raw/                    # Raw Olist CSV files
│
├── sql/
│   └── schema.sql              # PostgreSQL schema
│
├── notebooks/
│   ├── 01_read_and_join.ipynb
│   ├── 02_build_label.ipynb
│   ├── 03_split_data.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_features.ipynb
│   └── 06_train_tune_evaluate.ipynb
│
├── artifacts/                  # Generated datasets and model artifacts
│
├── docs/                       # Supporting documentation
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Technologies

* Python
* Pandas
* Scikit-learn
* PostgreSQL
* Docker
* Jupyter Notebook
* Parquet

---

## Reproducibility

The project uses:

* Docker for the database environment
* Fixed random seeds where applicable
* Parquet artifacts between pipeline stages
* Training-only fitting for preprocessing transformations
* Separate Train, Validation, and Test datasets

This structure is intended to keep the pipeline reproducible and prevent data leakage between stages.

---

## Project Status

**Completed:**

* Relational database setup
* Data ingestion and validation
* ML table construction
* Target generation
* Train/Validation/Test splitting
* Exploratory Data Analysis
* Feature engineering and preprocessing
* Baseline model evaluation
* Random Forest tuning
* Final test evaluation
