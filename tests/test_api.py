from fastapi.testclient import TestClient

import api


class FixedPredictor:
    model_name = "bilstm_attention"

    def predict(self, clauses):
        labels = ["data_collection", "third_party_sharing"]
        return [
            {"label": labels[index % len(labels)], "score": 0.8}
            for index, _ in enumerate(clauses)
        ]


def test_text_analysis_returns_grouped_real_findings(monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: FixedPredictor())
    client = TestClient(api.app)

    response = client.post(
        "/api/analyze",
        json={"text": "We collect account data. We share data with service providers."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "bilstm_attention"
    assert payload["clause_count"] == 2
    assert len(payload["topics"]) == 7
    collection = next(topic for topic in payload["topics"] if topic["key"] == "data_collection")
    assert collection["count"] == 1
    assert collection["clauses"][0]["text"] == "We collect account data."
    assert collection["clauses"][0]["score"] == 0.8


def test_analysis_requires_exactly_one_input(monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: FixedPredictor())
    client = TestClient(api.app)

    response = client.post("/api/analyze", json={"text": "Text", "url": "https://example.com"})

    assert response.status_code == 422
    assert "exactly one" in response.json()["detail"]


def test_analysis_rejects_oversized_text():
    client = TestClient(api.app)

    response = client.post("/api/analyze", json={"text": "x" * (api.MAX_TEXT_CHARACTERS + 1)})

    assert response.status_code == 413


def test_url_analysis_rejects_non_http_urls():
    client = TestClient(api.app)

    response = client.post("/api/analyze", json={"url": "file:///private/policy.txt"})

    assert response.status_code == 400


def test_file_endpoint_accepts_utf8_text(monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: FixedPredictor())
    client = TestClient(api.app)

    response = client.post(
        "/api/analyze-file",
        files={"file": ("policy.txt", b"We collect account data.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["document"] == "policy.txt"
    assert response.json()["topics"][0]["clauses"][0]["text"] == "We collect account data."


def test_file_endpoint_rejects_unsupported_file_types():
    client = TestClient(api.app)

    response = client.post(
        "/api/analyze-file",
        files={"file": ("policy.docx", b"not supported", "application/octet-stream")},
    )

    assert response.status_code == 400


def test_health_reports_attention_model_ready(monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: FixedPredictor())
    client = TestClient(api.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_ready"] is True


def test_health_fails_when_trained_artifacts_are_missing(monkeypatch):
    def missing_model():
        raise FileNotFoundError("The trained model is missing.")

    monkeypatch.setattr(api, "get_predictor", missing_model)
    client = TestClient(api.app)

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "The trained model is missing."
