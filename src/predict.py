from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python src/train_model.py' first."
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str) -> tuple[str, float]:
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text cannot be empty.")

    model = load_model()
    prediction = str(model.predict([clean_text])[0])

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([clean_text])[0]
        confidence = float(max(probabilities))

    return prediction, confidence
