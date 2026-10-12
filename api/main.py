import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
# MODEL_PATH can be overridden with an environment variable (useful for Docker and tests)
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "spam_model.joblib"))

# The loaded model lives here so every request can reuse it
state: dict = {"model": None}


def load_model(path: Path = MODEL_PATH) -> None:
    """Load the trained pipeline from disk into memory (once, at startup)."""
    if path.exists():
        state["model"] = joblib.load(path)
    else:
        state["model"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    state["model"] = None


app = FastAPI(
    title="Spam Classifier API",
    description="Send a text message and get back 'spam' or 'ham' with a confidence score.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["You won a free prize, click now!"],
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or only whitespace")
        return value


class PredictResponse(BaseModel):
    prediction: str
    confidence: float | None = Field(
        None, description="Probability of the predicted class (if the model supports it)"
    )


def get_model():
    model = state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train.py` to create it.",
        )
    return model


@app.get("/health")
def health() -> dict:
    """Used by Docker, CI and the hosting platform to check the service is alive."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model-info")
def model_info() -> dict:
    """Describe the model that is currently being served."""
    model = get_model()
    steps = [type(step).__name__ for _, step in model.steps]
    return {
        "model_file": MODEL_PATH.name,
        "pipeline_steps": steps,
        "classes": [str(c) for c in model.classes_],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify one message as spam or ham."""
    model = get_model()
    label = str(model.predict([request.text])[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([request.text])[0]
        confidence = round(float(max(probabilities)), 4)

    return PredictResponse(prediction=label, confidence=confidence)