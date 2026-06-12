# Deploy as a Webpage

This project is a Flask web app, so deploy it as a Python web service instead of a static HTML page.

## Files Needed for Deployment

- `app.py`
- `wsgi.py`
- `requirements.txt`
- `Procfile`
- `render.yaml`
- `templates/`
- `static/`
- `src/`
- `models/sentiment_model.joblib`

The large generated file `data/reviews_500k.csv` is not needed online because the trained model is already saved in `models/sentiment_model.joblib`.

## Deploy on Render

1. Create a GitHub repository.
2. Upload this project folder to the repository.
3. Make sure `models/sentiment_model.joblib` is included.
4. Go to Render and create a new Web Service.
5. Connect your GitHub repository.
6. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn wsgi:app
Health Check Path: /healthz
```

After the deploy finishes, Render gives you a public URL like:

```text
https://your-app-name.onrender.com
```

## Local Test Before Uploading

```bash
python run_server.py
```

Then open:

```text
http://127.0.0.1:5000
```

