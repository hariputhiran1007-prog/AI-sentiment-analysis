import os

from flask import Flask, render_template, request

from src.predict import MODEL_PATH, predict_sentiment
from src.train_model import train


app = Flask(__name__)


@app.route("/healthz")
def healthz():
    return {"status": "ok"}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "").strip()

        try:
            if not MODEL_PATH.exists():
                train()

            sentiment, confidence = predict_sentiment(review_text)
            result = {
                "sentiment": sentiment,
                "confidence": round(confidence * 100, 2),
            }
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Unable to analyze sentiment: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        review_text=review_text,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
