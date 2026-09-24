import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"].startswith("olist-late-delivery-rf:")


def test_get_model_info(client):
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "olist-late-delivery-rf"
    assert data["alias"] == "Production"


def test_predict_single_valid(client):
    payload = {
        "order_purchase_timestamp": "2017-08-15 10:30:00",
        "order_approved_at": "2017-08-15 10:30:00",
        "order_estimated_delivery_date": "2017-08-25 00:00:00",
        "item_count": 2, "unique_products": 2, "unique_sellers": 1,
        "total_price": 150.50, "total_freight_value": 25.00,
        "payment_count": 1, "payment_types_count": 1,
        "customer_state": "SP", "customer_city": "sao paulo", "customer_zip_code_prefix": "01000",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["probability"] == pytest.approx(0.22031052838031936, abs=1e-9)


def test_predict_single_schema_violation(client):
    """Pydantic catches negative price -> 422"""
    payload = {
        "order_purchase_timestamp": "2017-08-15 10:30:00",
        "order_approved_at": "2017-08-15 10:30:00",
        "order_estimated_delivery_date": "2017-08-25 00:00:00",
        "item_count": 2, "unique_products": 2, "unique_sellers": 1,
        "total_price": -50.0,  # Fails Pydantic ge=0
        "total_freight_value": 25.00, "payment_count": 1, "payment_types_count": 1,
        "customer_state": "SP", "customer_city": "sao paulo", "customer_zip_code_prefix": "01000",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_single_rejected_by_gx(client):
    """Passes Pydantic (length 2), but fails Great Expectations regex -> 400"""
    payload = {
        "order_purchase_timestamp": "2017-08-15 10:30:00",
        "order_approved_at": "2017-08-15 10:30:00",
        "order_estimated_delivery_date": "2017-08-25 00:00:00",
        "item_count": 2, "unique_products": 2, "unique_sellers": 1,
        "total_price": 150.50, "total_freight_value": 25.00,
        "payment_count": 1, "payment_types_count": 1,
        "customer_state": "12",  # Passes Pydantic (len 2), fails GX regex ^[A-Z]{2}$
        "customer_city": "sao paulo", "customer_zip_code_prefix": "01000",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["status"] == "rejected"
    assert "customer_state" in data["detail"]["error"]


def test_predict_single_missing_field(client):
    payload = {"order_purchase_timestamp": "2017-08-15 10:30:00", "item_count": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_valid(client):
    payload = [
        {
            "order_purchase_timestamp": "2017-08-15 10:30:00", "order_approved_at": "2017-08-15 10:30:00",
            "order_estimated_delivery_date": "2017-08-25 00:00:00", "item_count": 2, "unique_products": 2,
            "unique_sellers": 1, "total_price": 150.50, "total_freight_value": 25.00, "payment_count": 1,
            "payment_types_count": 1, "customer_state": "SP", "customer_city": "sao paulo", "customer_zip_code_prefix": "01000",
        },
        {
            "order_purchase_timestamp": "2018-01-01 12:00:00", "order_approved_at": "2018-01-01 12:00:00",
            "order_estimated_delivery_date": "2018-01-10 00:00:00", "item_count": 1, "unique_products": 1,
            "unique_sellers": 1, "total_price": 50.00, "total_freight_value": 10.00, "payment_count": 1,
            "payment_types_count": 1, "customer_state": "RJ", "customer_city": "rio de janeiro", "customer_zip_code_prefix": "20000",
        },
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_batch_empty(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "Empty batch"

