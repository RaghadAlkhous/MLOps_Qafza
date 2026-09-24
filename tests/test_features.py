import numpy as np
import pandas as pd
import pytest

from src.features import create_features


def _raw(**overrides):
    base = {
        "order_purchase_timestamp": ["2017-08-15 10:30:00"],
        "order_approved_at": ["2017-08-15 12:30:00"],
        "order_estimated_delivery_date": ["2017-08-25 00:00:00"],
        "item_count": [2],
        "unique_products": [2],
        "unique_sellers": [1],
        "total_price": [100.0],
        "total_freight_value": [20.0],
        "payment_count": [1],
        "payment_types_count": [1],
        "customer_state": ["SP"],
        "customer_city": ["sao paulo"],
        "customer_zip_code_prefix": ["01000"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_create_features_key_columns():
    out = create_features(_raw())
    for col in [
        "approval_delay_hours",
        "approval_delay_missing",
        "estimated_delivery_lead_days",
        "freight_ratio",
        "is_weekend",
        "purchase_month_sin",
        "customer_city",
    ]:
        assert col in out.columns


def test_approval_delay_hours_value():
    out = create_features(_raw())
    assert out["approval_delay_hours"].iloc[0] == pytest.approx(2.0)


def test_approval_delay_missing_flag():
    out = create_features(_raw(order_approved_at=[None]))
    assert out["approval_delay_missing"].iloc[0] == 1
    assert np.isnan(out["approval_delay_hours"].iloc[0])


def test_freight_ratio_zero_price():
    out = create_features(_raw(total_price=[0.0]))
    assert out["freight_ratio"].iloc[0] == 0.0


def test_is_weekend():
    sat = create_features(_raw(order_purchase_timestamp=["2017-08-19 10:00:00"]))
    mon = create_features(_raw(order_purchase_timestamp=["2017-08-21 10:00:00"]))
    assert sat["is_weekend"].iloc[0] == 1
    assert mon["is_weekend"].iloc[0] == 0


def test_no_leakage_columns():
    out = create_features(_raw())
    for col in [
        "order_status",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "delivery_delay_days",
    ]:
        assert col not in out.columns
