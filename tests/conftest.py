import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from api.main import app, state

SPAM = [
    "win a free prize now",
    "claim your free cash reward today",
    "urgent you have won a prize click here",
    "free entry text win now",
]
HAM = [
    "are we meeting for lunch tomorrow",
    "call me when you reach home",
    "did you finish the assignment",
    "see you at the library at five",
]
TEXTS = SPAM * 5 + HAM * 5
LABELS = ["spam"] * len(SPAM) * 5 + ["ham"] * len(HAM) * 5


def _train(classifier) -> Pipeline:
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", classifier)])
    return model.fit(TEXTS, LABELS)


@pytest.fixture
def client(monkeypatch):
    """API client with a small Naive Bayes model loaded (supports probabilities)."""
    monkeypatch.setitem(state, "model", _train(MultinomialNB()))
    return TestClient(app)


@pytest.fixture
def client_svm(monkeypatch):
    """API client with a LinearSVC model (has no predict_proba)."""
    monkeypatch.setitem(state, "model", _train(LinearSVC()))
    return TestClient(app)


@pytest.fixture
def client_no_model(monkeypatch):
    """API client where no model is loaded."""
    monkeypatch.setitem(state, "model", None)
    return TestClient(app)