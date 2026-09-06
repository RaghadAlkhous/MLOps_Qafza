# Notebook 04 — EDA Findings

## Scope and methodology
- Source: `artifacts/notebook_03/train.parquet` only.
- Training rows: 67,532.
- Columns in original source artifact: 22.
- Grain: one row per order.
- Prediction point used for feature-availability reasoning: **order approval time**.
- Validation and test sets were not opened.
- EDA helper columns were created in a separate `eda` working copy and are not part of the source artifact.

## Target
- On-time orders: 62,053.
- Late orders: 5,479.
- Late rate: 8.1132%.
- Because the positive class is relatively small, accuracy must not be the sole model-selection metric.

## Missing values
- order_approved_at: 12 missing (0.0178%). Late rate when missing = 0.00%; when observed = 8.11%.
- order_delivered_carrier_date: 1 missing (0.0015%). Late rate when missing = 100.00%; when observed = 8.11%.

Interpretation: missingness is treated as a data-availability signal until its source meaning and prediction-time availability are confirmed.

## Numerical variables
Top absolute skewness values observed:
- payment_count: skewness = 19.870
- total_freight_value: skewness = 14.184
- total_price: skewness = 11.029
- total_payment_value: skewness = 10.428
- unique_sellers: skewness = 9.866

Top IQR outlier rates observed:
- total_freight_value: 6783 IQR outliers (10.04%).
- item_count: 6700 IQR outliers (9.92%).
- total_price: 5402 IQR outliers (8.00%).
- total_payment_value: 5341 IQR outliers (7.91%).
- unique_products: 2192 IQR outliers (3.25%).

Numeric non-negativity checks were saved in `numeric_validity_summary.csv`; zero frequencies were saved in `zero_summary.csv`.
Outliers are not automatically removed; they require model-appropriate treatment or robust transformation if they are valid observations.

## Categorical variables
- `customer_state`: 27 categories.
- `customer_city`: 3,668 categories.
- `customer_city` therefore requires a controlled encoding strategy rather than indiscriminate one-hot expansion.
- Rare-category impact:
- order_status: 1 rare categories (50.00% of categories), affecting 5 rows (0.01%).
- customer_city: 3646 rare categories (99.40% of categories), affecting 38,315 rows (56.74%).
- customer_state: 9 rare categories (33.33% of categories), affecting 1,446 rows (2.14%).
- Formatting/collision audit was performed without automatically merging categories.

## Relationship with target
- Strongest numerical Pearson associations observed:
- unique_sellers: Pearson r = -0.028
- total_freight_value: Pearson r = 0.024
- unique_products: Pearson r = -0.023
- item_count: Pearson r = -0.019
- total_payment_value: Pearson r = 0.017
- Numerical group comparisons, categorical cross-tabs, state late rates, quantile analyses, and correlations were computed.
- Associations are descriptive; correlation or group differences do not establish causality.

## Geography
- Highest supported state late rate: AL = 25.29% (n=261).
- Lowest supported state late rate: RO = 2.37% (n=169).
- State differences are descriptive associations, not causal effects.
- Available: customer state, customer city, customer ZIP-code prefix.
- Unavailable in the current order-level training artifact: seller state, seller ZIP, seller coordinates, and customer-seller distance.
- These unavailable seller features were not reconstructed in Notebook 04.

## Dates and delivery process
- Purchase month, year-month, weekday, weekend, hour, and Brazilian holiday associations were analyzed.
- Holiday result:
- Holiday late rate = 6.31%; non-holiday late rate = 8.14%.
- Date consistency checks:
- order_purchase_timestamp <= order_approved_at: 0 invalid rows among 67,520 comparable rows (0.0000%).
- order_approved_at <= order_delivered_carrier_date: 938 invalid rows among 67,519 comparable rows (1.3892%).
- order_delivered_carrier_date <= order_delivered_customer_date: 20 invalid rows among 67,531 comparable rows (0.0296%).
- order_purchase_timestamp <= order_delivered_customer_date: 0 invalid rows among 67,532 comparable rows (0.0000%).
- Estimated delivery lead time was analyzed as a candidate predictive-time feature.
- Purchase-to-carrier, carrier-to-customer, purchase-to-customer, and delivery-delay durations were analyzed for diagnostics only.
- Post-outcome durations must not enter predictive features.

## Leakage exclusions
Excluded from predictive features:
- `order_id`, `customer_id`
- `delivery_delay_days`, `label_data_available`
- `order_delivered_carrier_date`, `order_delivered_customer_date`
- final `order_status`

## Feature-engineering direction for Notebook 05
- Engineer temporal features from purchase/approval timestamps.
- Engineer estimated delivery lead-time features.
- Encode `customer_state` and evaluate controlled treatment of high-cardinality city/ZIP.
- Verify business-time availability of payment and order-aggregate features before including them.
- Handle missing values explicitly, with missingness indicators only when justified by prediction-time availability.
- Preserve one-row-per-order grain.
- Fit preprocessing only on training data, then apply it unchanged to validation/test.

## Modeling direction
Start with a transparent baseline such as Logistic Regression and compare it with a nonlinear/tree-based model.
Use Precision, Recall, F1, PR-AUC and ROC-AUC; report the confusion matrix and accuracy as supplementary information.
