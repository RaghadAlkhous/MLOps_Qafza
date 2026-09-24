from src.validation import validate_order_data


def test_valid_data_passes(valid_order):
    res = validate_order_data(valid_order)
    assert res["success"] is True


def test_negative_price_rejected(valid_order):
    bad = valid_order.copy()
    bad["total_price"] = -50.0
    res = validate_order_data(bad)
    assert res["success"] is False
    assert res["status"] == "rejected"
    assert any("total_price" in d for d in res["details"])


def test_invalid_state_rejected(valid_order):
    bad = valid_order.copy()
    bad["customer_state"] = "xx"
    res = validate_order_data(bad)
    assert res["success"] is False
    assert res["status"] == "rejected"


def test_zero_item_count_rejected(valid_order):
    bad = valid_order.copy()
    bad["item_count"] = 0
    res = validate_order_data(bad)
    assert res["success"] is False


def test_missing_column_fails(valid_order):
    bad = valid_order.drop(columns=["order_purchase_timestamp"])
    res = validate_order_data(bad)
    assert res["success"] is False
