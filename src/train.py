from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"  # or data/spam.csv (Kaggle)
MODEL_PATH = ROOT / "models" / "spam_model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    """Load the SMS spam dataset into a DataFrame with 'label' and 'text' columns."""
    if not path.exists():
        # Allow the Kaggle version too
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


def build_pipeline() -> Pipeline:
    """TF-IDF features + Logistic Regression, wrapped so the model accepts raw text."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages")
    print(df["label"].value_counts(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    print(classification_report(y_test, predictions))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Quick sanity check
    sample = "Congratulations! You won a free prize, click now!"
    print(f"\nSample prediction: {model.predict([sample])[0]!r} for {sample!r}")


if __name__ == "__main__":
    main()