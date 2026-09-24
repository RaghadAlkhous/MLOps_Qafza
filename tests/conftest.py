import pandas as pd
import pytest

from src.pipeline import InferencePipeline


@pytest.fixture(scope="session")
def pipeline():
    return InferencePipeline()


@pytest.fixture()
def valid_order():
    return pd.DataFrame(
        {
            "order_purchase_timestamp": ["2017-08-15 10:30:00"],
            "order_approved_at": ["2017-08-15 10:30:00"],
            "order_estimated_delivery_date": ["2017-08-25 00:00:00"],
            "item_count": [2],
            "unique_products": [2],
            "unique_sellers": [1],
            "total_price": [150.50],
            "total_freight_value": [25.00],
            "payment_count": [1],
            "payment_types_count": [1],
            "customer_state": ["SP"],
            "customer_city": ["sao paulo"],
            "customer_zip_code_prefix": ["01000"],
        }
    )
