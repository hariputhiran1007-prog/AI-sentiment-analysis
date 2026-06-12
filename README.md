# AI Based Sentiment Analysis System

A machine learning project that analyzes review or comment text and predicts whether it is **Positive**, **Negative**, or **Neutral**.

The project uses:

- Python
- scikit-learn
- TF-IDF text features
- TF-IDF + linear logistic classifier
- Flask web interface

## Project Structure

```text
ai sentiment analysis/
|-- app.py
|-- requirements.txt
|-- run_server.py
|-- data/
|   |-- sample_reviews.csv
|   `-- reviews_500k.csv
|-- models/
|   |-- .gitkeep
|   `-- sentiment_model.joblib
|-- src/
|   |-- generate_large_dataset.py
|   |-- predict.py
|   `-- train_model.py
|-- static/
|   `-- style.css
`-- templates/
    `-- index.html
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train_model.py
```

This creates `models/sentiment_model.joblib`.

## Train With 500,000 Samples

If you already have a large CSV file, it should contain:

- `text`: review or comment text
- `sentiment`: `Positive`, `Negative`, or `Neutral`

Train from your CSV:

```bash
python src/train_model.py --data data/your_large_reviews.csv --max-samples 500000
```

For a local demo dataset with 500,000 generated samples:

```bash
python src/generate_large_dataset.py --samples 500000 --output data/reviews_500k.csv
python src/train_model.py --data data/reviews_500k.csv --max-samples 500000
```

## Run the Web App

```bash
python run_server.py
```

Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## Deploy Online

This is a Flask application, so deploy it as a Python web service.

Recommended production start command:

```bash
gunicorn wsgi:app
```

See `DEPLOYMENT.md` for step-by-step deployment instructions.

## Try Example Inputs

- `The product quality is excellent and delivery was fast.`
- `It is okay, nothing special but not bad.`
- `The service was terrible and I want a refund.`
