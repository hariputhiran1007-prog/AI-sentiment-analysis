import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "sample_reviews.csv"
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"


def load_data(
    data_path: Path,
    text_column: str,
    label_column: str,
    max_samples: int | None = None,
) -> pd.DataFrame:
    data = pd.read_csv(data_path, usecols=[text_column, label_column])
    required_columns = {text_column, label_column}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Dataset must contain '{text_column}' and '{label_column}' columns."
        )

    data = data.dropna(subset=[text_column, label_column])
    data[text_column] = data[text_column].astype(str).str.strip()
    data[label_column] = data[label_column].astype(str).str.strip()
    data = data[(data[text_column] != "") & (data[label_column] != "")]

    if max_samples and len(data) > max_samples:
        data = data.sample(n=max_samples, random_state=42)

    return data.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=1000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    text_column: str = "text",
    label_column: str = "sentiment",
    max_samples: int | None = None,
    test_size: float = 0.2,
) -> Pipeline:
    data = load_data(data_path, text_column, label_column, max_samples)
    label_counts = data[label_column].value_counts()

    if len(data) < 6:
        raise ValueError("Dataset is too small. Add more labeled review examples.")

    stratify = data[label_column] if label_counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        data[text_column],
        data[label_column],
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Dataset: {data_path}")
    print(f"Rows used: {len(data):,}")
    print("Class distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,}")
    print()
    print(f"Model saved to: {model_path}")
    print(f"Accuracy: {accuracy:.2f}")
    print()
    print(classification_report(y_test, predictions, zero_division=0))
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sentiment analysis model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to CSV data.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Output model path.")
    parser.add_argument("--text-column", default="text", help="CSV text column name.")
    parser.add_argument(
        "--label-column", default="sentiment", help="CSV sentiment label column name."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of rows to train from.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data,
        model_path=args.model,
        text_column=args.text_column,
        label_column=args.label_column,
        max_samples=args.max_samples,
        test_size=args.test_size,
    )
