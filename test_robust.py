import pandas as pd

from src.pipeline import InferencePipeline

pipeline = InferencePipeline()

valid_input = pd.DataFrame(
    {
        "order_purchase_timestamp": ["2017-08-15 10:30:00", "2018-01-01 12:00:00"],
        "order_approved_at": ["2017-08-15 10:30:00", "2018-01-01 12:00:00"],
        "order_estimated_delivery_date": ["2017-08-25 00:00:00", "2018-01-10 00:00:00"],
        "item_count": [2, 1],
        "unique_products": [2, 1],
        "unique_sellers": [1, 1],
        "total_price": [150.50, 50.00],
        "total_freight_value": [25.00, 10.00],
        "payment_count": [1, 1],
        "payment_types_count": [1, 1],
        "customer_state": ["SP", "RJ"],
        "customer_city": ["sao paulo", "rio de janeiro"],
        "customer_zip_code_prefix": ["01000", "20000"],
    }
)

print("\n=== TEST 1: Single Row (Parity) ===")
res1 = pipeline.predict(valid_input.iloc[[0]])
print(res1)

print("\n=== TEST 2: Bad Data (Negative Price) ===")
bad_input = valid_input.iloc[[0]].copy()
bad_input["total_price"] = -50.0
res2 = pipeline.predict(bad_input)
print(res2)

print("\n=== TEST 3: Batch Prediction (Vectorized) ===")
res3 = pipeline.predict(valid_input)
print(res3)
