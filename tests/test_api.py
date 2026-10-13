# ---------- /health ----------
def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_still_ok_without_model(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


# ---------- /model-info ----------
def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["classes"] == ["ham", "spam"]
    assert body["pipeline_steps"] == ["TfidfVectorizer", "MultinomialNB"]


def test_model_info_without_model_returns_503(client_no_model):
    assert client_no_model.get("/model-info").status_code == 503


# ---------- /predict: correct behaviour ----------
def test_predict_spam(client):
    response = client.post("/predict", json={"text": "win a free prize now"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"


def test_predict_ham(client):
    response = client.post("/predict", json={"text": "are we meeting for lunch tomorrow"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "ham"


def test_predict_response_shape(client):
    body = client.post("/predict", json={"text": "free cash prize"}).json()
    assert set(body) == {"prediction", "confidence"}
    assert body["prediction"] in {"spam", "ham"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_confidence_is_null_when_model_has_no_probabilities(client_svm):
    body = client_svm.post("/predict", json={"text": "win a free prize now"}).json()
    assert body["prediction"] == "spam"
    assert body["confidence"] is None


# ---------- /predict: bad input ----------
def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_rejects_whitespace_only(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_wrong_type(client):
    assert client.post("/predict", json={"text": 123}).status_code == 422


def test_predict_rejects_too_long_text(client):
    assert client.post("/predict", json={"text": "a" * 5001}).status_code == 422


# ---------- /predict: no model ----------
def test_predict_without_model_returns_503(client_no_model):
    response = client_no_model.post("/predict", json={"text": "hello"})
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]