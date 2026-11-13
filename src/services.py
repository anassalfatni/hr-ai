"""Couche service unique, partagée par l'API Flask et l'UI Streamlit."""
import json, joblib
from functools import lru_cache
from . import config as C
from .train import predict_hier
from .matching import match, extract_keywords
from .genai import generate_job_posting
from .preprocess import clean_text

@lru_cache
def _cv(): return joblib.load(C.MODELS / "cv_classifier.joblib")
@lru_cache
def _sal(): return joblib.load(C.MODELS / "salary.joblib")

def classify_cv(text: str) -> dict:
    cleaned = clean_text(text)
    if len(cleaned) < 50:
        raise ValueError("Le texte extrait du CV est trop court (au moins 50 caractères).")
    g, c = predict_hier(_cv(), cleaned)
    return {"domain": g, "specialty": c, "keywords": extract_keywords(cleaned)}

def predict_salary(age, education, job_title, years) -> dict:
    import pandas as pd
    X = pd.DataFrame([{"Age": age, "Education Level": education, "Job Title": job_title, "Years of Experience": years}])
    return {"salary_eur_equiv": round(float(_sal()["pipe"].predict(X)[0]), -2)}

def options() -> dict:
    s = _sal(); return {"titles": s["titles"], "education": s["edu"]}

def metrics() -> dict:
    return json.loads(C.METRICS_JSON.read_text())

def match_cv(cv_text: str, job_description: str) -> dict:
    return match(cv_text, job_description)


def job_posting(**kw): return generate_job_posting(**kw)
