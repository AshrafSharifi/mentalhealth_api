import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid(client, valid_input):
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    assert isinstance(response.json()["mbi_ex"], float)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"cesd": 99}, "above the scale maximum"),
        ({"age": -1}, "below the minimum"),
        ({"age": "twenty"}, "wrong type"),
        ({"foo": 1}, "unknown field"),
    ],
)
def test_predict_invalid_values(client, valid_input, change, reason):
    response = client.post("/predict", json={**valid_input, **change})
    assert response.status_code == 422, reason


def test_predict_missing_field(client, valid_input):
    del valid_input["stai_t"]
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 422


def test_explain_returns_top_k_sorted(client, valid_input):
    response = client.post("/explain?top_k=3", json=valid_input)
    assert response.status_code == 200
    contributions = response.json()["top_contributions"]
    assert len(contributions) == 3
    sizes = [abs(c["shap_value"]) for c in contributions]
    assert sizes == sorted(sizes, reverse=True)


def test_explain_shap_values_add_up(client, valid_input):
    # SHAP's key property: base value + all contributions = the prediction.
    body = client.post("/explain?top_k=16", json=valid_input).json()
    total = body["base_value"] + sum(c["shap_value"] for c in body["top_contributions"])
    assert total == pytest.approx(body["mbi_ex"], abs=0.05)  # small tolerance for rounding


@pytest.mark.parametrize("top_k", [0, 17])
def test_explain_rejects_bad_top_k(client, valid_input, top_k):
    response = client.post(f"/explain?top_k={top_k}", json=valid_input)
    assert response.status_code == 422