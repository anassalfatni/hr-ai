"""Serving API (Flask) : python -m app.api"""
from flask import Flask, jsonify, request
from src import services as S
from src.preprocess import extract_pdf_text

app = Flask(__name__)

def _j(): return request.get_json(force=True, silent=True) or {}

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/metrics")
def metrics(): return jsonify(S.metrics())

@app.post("/classify")
def classify():
    uploaded_pdf = request.files.get("file")
    if uploaded_pdf is not None:
        if not uploaded_pdf.filename or not uploaded_pdf.filename.lower().endswith(".pdf"):
            return {"error": "le fichier doit être un PDF"}, 400
        try:
            text = extract_pdf_text(uploaded_pdf.read())
        except ValueError as e:
            return {"error": str(e)}, 400
    else:
        text = _j().get("text", "")
    if not isinstance(text, str) or len(text) < 50:
        return {"error": "CV requis (au moins 50 caractères de texte)"}, 400
    try:
        return jsonify(S.classify_cv(text))
    except ValueError as e:
        return {"error": str(e)}, 400

@app.post("/salary")
def salary():
    d = _j()
    try:
        return jsonify(S.predict_salary(float(d["age"]), d["education"], d["job_title"], float(d["years"])))
    except (KeyError, ValueError):
        return {"error": "champs requis: age, education, job_title, years"}, 400

@app.post("/match")
def match():
    uploaded_pdf = request.files.get("cv_file")
    if uploaded_pdf is not None:
        if not uploaded_pdf.filename or not uploaded_pdf.filename.lower().endswith(".pdf"):
            return {"error": "cv_file doit être un PDF"}, 400
        try:
            cv_text = extract_pdf_text(uploaded_pdf.read())
        except ValueError as e:
            return {"error": str(e)}, 400
        job_description = request.form.get("job_description", "")
    else:
        d = _j()
        cv_text = d.get("cv_text", "")
        job_description = d.get("job_description", "")
    if not isinstance(cv_text, str) or len(cv_text) < 50:
        return {"error": "cv_text requis (au moins 50 caractères)"}, 400
    if not isinstance(job_description, str) or len(job_description) < 50:
        return {"error": "job_description requis (au moins 50 caractères)"}, 400
    try:
        return jsonify(S.match_cv(cv_text, job_description))
    except ValueError as e:
        return {"error": str(e)}, 400

@app.post("/job-posting")
def posting():
    d = _j()
    if not d.get("title"): return {"error": "title requis"}, 400
    return jsonify(S.job_posting(title=d["title"], domain=d.get("domain", ""), skills=d.get("skills", []),
                                 level=d.get("level", "Confirmé"), salary=d.get("salary"),
                                 company=d.get("company", ""), contract_type=d.get("contract_type", ""),
                                 location=d.get("location", ""), work_mode=d.get("work_mode", ""),
                                 education=d.get("education", ""), languages=d.get("languages", []),
                                 salary_min=d.get("salary_min"), salary_max=d.get("salary_max")))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
