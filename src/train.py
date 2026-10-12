from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_PATH = ROOT / "models" / "spam_model.joblib"

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "spam-classifier"

BEST_METRIC = "spam_f1"

def load_data(path: Path) -> pd.DataFrame:
    """Load the SMS spam dataset into a DataFrame with 'label' and 'text' columns."""
    if not path.exists():
        kaggle_path = path.parent / "spam.csv"
        if kaggle_path.exists():
            path = kaggle_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download it and put it in the data/ folder."
            )

    if path.suffix == ".csv":
        df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
    else:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])

    return df.dropna().drop_duplicates()

EXPERIMENTS = [
    {
        "run_name": "baseline_logreg",
        "model": "logreg",                  ###Basic Logistic Regression
        "ngram_max": 1,
        "stop_words": "english",
        "C": 1.0,
        "class_weight": None,
    },
    {
        "run_name": "logreg_balanced",      ###Balances the importance of spam and ham classes
        "model": "logreg",                
        "ngram_max": 1,
        "stop_words": "english",
        "C": 1.0,
        "class_weight": "balanced",
    },
    {
        "run_name": "logreg_balanced_C10",    ###Changes regularization strength
        "model": "logreg",
        "ngram_max": 1,
        "stop_words": "english",
        "C": 10.0,
        "class_weight": "balanced",
    },
    {
        "run_name": "logreg_balanced_bigrams",   ###Adds two-word text features
        "model": "logreg",
        "ngram_max": 2,
        "stop_words": "english",
        "C": 10.0,
        "class_weight": "balanced",
    },
    {
        "run_name": "logreg_balanced_keep_stopwords",    ###Keeps common words instead of removing them
        "model": "logreg",
        "ngram_max": 2,
        "stop_words": None,
        "C": 10.0,
        "class_weight": "balanced",
    },
    {
        "run_name": "naive_bayes",    ###Uses Multinomial Naive Bayes
        "model": "nb",
        "ngram_max": 1,
        "stop_words": "english",
        "alpha": 0.1,
    },
    {
        "run_name": "linear_svm_balanced",   ###Uses a linear SVM with balanced classes
        "model": "svm",
        "ngram_max": 2,
        "stop_words": None,
        "C": 1.0,
        "class_weight": "balanced",
    },
]


def build_pipeline(cfg: dict) -> Pipeline:
    """Build TF-IDF + classifier from an experiment config."""
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words=cfg["stop_words"],
        ngram_range=(1, cfg["ngram_max"]),
    )

    if cfg["model"] == "logreg":
        clf = LogisticRegression(
            C=cfg["C"], class_weight=cfg["class_weight"], max_iter=1000
        )
    elif cfg["model"] == "svm":
        clf = LinearSVC(C=cfg["C"], class_weight=cfg["class_weight"])
    elif cfg["model"] == "nb":
        clf = MultinomialNB(alpha=cfg["alpha"])
    else:
        raise ValueError(f"Unknown model type: {cfg['model']}")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """Compute the metrics we want to compare across runs."""
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "spam_precision": precision_score(y_test, preds, pos_label="spam"),
        "spam_recall": recall_score(y_test, preds, pos_label="spam"),
        "spam_f1": f1_score(y_test, preds, pos_label="spam"),
        "macro_f1": f1_score(y_test, preds, average="macro"),
    }


def main() -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages")
    print(df["label"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best = {"score": -1.0, "name": None, "model": None, "metrics": None}
    results = []

    for cfg in EXPERIMENTS:
        with mlflow.start_run(run_name=cfg["run_name"]):
            # 1. Log the settings used in this run
            params = {k: v for k, v in cfg.items() if k != "run_name"}
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_param("train_size", len(X_train))

            # 2. Train
            model = build_pipeline(cfg)
            model.fit(X_train, y_train)

            # 3. Evaluate and log the results
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # 4. Save the trained pipeline inside the MLflow run
            mlflow.sklearn.log_model(model, name="model")

            results.append({"run": cfg["run_name"], **metrics})
            print(
                f"{cfg['run_name']:<34} "
                f"acc={metrics['accuracy']:.4f}  "
                f"spam_P={metrics['spam_precision']:.3f}  "
                f"spam_R={metrics['spam_recall']:.3f}  "
                f"spam_F1={metrics['spam_f1']:.3f}"
            )

            if metrics[BEST_METRIC] > best["score"]:
                best.update(
                    score=metrics[BEST_METRIC],
                    name=cfg["run_name"],
                    model=model,
                    metrics=metrics,
                )

    # Save the winner so the API (Phase 3) can load it
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)

    print("\n" + "=" * 60)
    print(f"Best run: {best['name']}  ({BEST_METRIC} = {best['score']:.4f})")
    print(f"Best model saved to {MODEL_PATH}")
    print("=" * 60)

    sample = "Congratulations! You won a free prize, click now!"
    print(f"\nSample prediction: {best['model'].predict([sample])[0]!r} for {sample!r}")
    print("\nView the dashboard with:")
    print("    mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()