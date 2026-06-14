import os
import csv
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request

from src.predict import MODEL_PATH, predict_sentiment
from src.train_model import train

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parent
HISTORY_PATH = ROOT_DIR / "reports" / "history.csv"


@app.route("/healthz")
def healthz():
    return {"status": "ok"}


def save_history(review, sentiment, confidence):
    HISTORY_PATH.parent.mkdir(exist_ok=True)

    file_exists = HISTORY_PATH.exists()

    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["date", "review", "sentiment", "confidence"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            review,
            sentiment,
            confidence
        ])


def load_history(limit=5):
    if not HISTORY_PATH.exists():
        return []

    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return rows[-limit:][::-1]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    review_text = ""
    probabilities = {}
    history = load_history()

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()

        try:
            if not MODEL_PATH.exists():
                train()

            sentiment, confidence, probabilities = predict_sentiment(review_text)

            confidence_percent = round(confidence * 100, 2)

            result = {
                "sentiment": sentiment,
                "confidence": confidence_percent,
            }

            save_history(review_text, sentiment, confidence_percent)
            history = load_history()

        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Unable to analyze sentiment: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        review_text=review_text,
        probabilities=probabilities,
        history=history
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)