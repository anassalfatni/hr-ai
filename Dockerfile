FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Entraînement au build : l'image embarque des modèles versionnés avec le code (reproductibilité)
RUN python scripts/make_synthetic_salary.py && python -m src.train
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:5000/health')"
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.api:app"]
