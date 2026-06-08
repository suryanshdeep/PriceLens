from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_json_contract() -> None:
    response = client.post(
        "/predict",
        json={
            "catalog_content": "Organic almond butter 16 oz jar",
            "brand": "Example Brand",
            "category": "Grocery",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["predicted_price"] > 0
    assert body["confidence_band"]["low"] < body["predicted_price"]
    assert body["confidence_band"]["high"] > body["predicted_price"]
    assert "catalog_content" in body["features_used"]


def test_model_info_contract() -> None:
    response = client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert "model_version" in body
    assert "model_type" in body
    assert "artifact_available" in body
