import pandas as pd
import pytest


def test_model_loaded_from_registry(pipeline):
    assert pipeline.model_version.startswith("olist-late-delivery-rf:")


def test_feature_list_size(pipeline):
    assert len(pipeline.artifacts["feature_list"]) == 50
    assert pipeline.model.n_features_in_ == 50


def test_single_prediction_known_input(pipeline, valid_order):
    res = pipeline.predict(valid_order)
    assert res["status"] == "success"
    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["probability"] == pytest.approx(0.22031052838031936, abs=1e-9)


def test_batch_returns_all_rows(pipeline, valid_order):
    batch = pd.concat([valid_order, valid_order], ignore_index=True)
    res = pipeline.predict(batch)
    assert res["status"] == "success"
    assert len(res["predictions"]) == 2


def test_bad_payload_rejected_not_crash(pipeline, valid_order):
    bad = valid_order.copy()
    bad["total_price"] = -1.0
    res = pipeline.predict(bad)
    assert res["status"] == "rejected"
