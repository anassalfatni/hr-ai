import json
from io import BytesIO
import numpy as np
import pandas as pd
import pytest
import urllib.error
from pypdf import PdfReader
from src import services as S, config as C
from src import preprocess as P
from src import embeddings as E
from src import train as T
from src import matching as M
from src.preprocess import mask_pii, clean_text
from src.matching import extract_keywords, match
from src import genai as G
from src.genai import generate_job_posting
from src.job_posting_pdf import create_job_posting_pdf
from app.api import app

CV = ("Senior accountant with 8 years experience in general ledger, reconciliation, accounts payable, audit and month-end close. "
      "Advanced excel, reporting and budgeting. Contact john@mail.com 06 12 34 56 78")

def test_pii_masked():
    t = clean_text(CV); assert "@" not in t and "john" not in t and "06" not in t

def test_clean_text_preserves_french_accents():
    assert "développeur" in clean_text("Développeur expérimenté")

def test_encode_texts_chunks_long_documents(monkeypatch):
    class Tokenizer:
        def tokenize(self, text):
            return list(text)
        def convert_tokens_to_ids(self, tokens):
            return tokens
        def num_special_tokens_to_add(self, pair=False):
            return 2
        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(token_ids)

    class FakeModel:
        tokenizer = Tokenizer()
        max_seq_length = 5

        def encode(self, texts, **kwargs):
            return np.array([[len(text), 1.0] for text in texts], dtype=np.float32)

    monkeypatch.setattr(E, "_model", lambda: FakeModel())
    vectors = E.encode_texts(["abcdefgh"])
    assert vectors.shape == (1, 2)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)

def test_load_resumes_from_category_pdf_folders(tmp_path, monkeypatch):
    category = tmp_path / "ACCOUNTANT"
    category.mkdir()
    (category / "resume.pdf").write_bytes(b"readable")
    (category / "unreadable.pdf").write_bytes(b"unreadable")
    monkeypatch.setattr(P, "RESUME_PDF_DIR", tmp_path)
    def fake_extract_pdf_text(content):
        if content == b"unreadable":
            raise ValueError("no text")
        return "accounting " * 120
    monkeypatch.setattr(P, "extract_pdf_text", fake_extract_pdf_text)
    with pytest.warns(RuntimeWarning, match="1 illisibles"):
        resumes = P.load_resumes()
    assert resumes.loc[0, "Category"] == "ACCOUNTANT"
    assert resumes.loc[0, "group"] == "Finance & Juridique"

def test_load_salary_does_not_require_or_filter_on_gender(tmp_path, monkeypatch):
    salary_file = tmp_path / "Salary Data.csv"
    pd.DataFrame({
        "Age": [30, 40],
        "Education Level": ["Bachelor's", "Master's"],
        "Job Title": ["Analyst", "Manager"],
        "Years of Experience": [5, 15],
        "Salary": [50000, 80000],
        "Gender": ["Female", None],
    }).to_csv(salary_file, index=False)
    monkeypatch.setattr(T.C, "SALARY_CSV", salary_file)
    salaries = T.load_salary()
    assert len(salaries) == 2
    assert "Gender" not in T.FEATS

def test_keywords():
    keywords = extract_keywords(CV)
    assert len(keywords) == 15
    assert all(any(term in keyword for keyword in keywords) for term in ("excel", "audit", "budgeting"))

def test_classify():
    r = S.classify_cv(CV); assert r["domain"] in C.HIERARCHY and r["specialty"] in C.CAT2GROUP

def test_salary_monotonic_experience():
    lo = S.predict_salary(30, S.options()["education"][0], S.options()["titles"][0], 1)["salary_eur_equiv"]
    hi = S.predict_salary(40, S.options()["education"][0], S.options()["titles"][0], 15)["salary_eur_equiv"]
    assert hi > lo

def test_matching_compares_cv_to_supplied_job_description():
    result = match(CV, "Accounting role with general ledger and audit responsibilities.")
    assert 0 <= result["score"] <= 1
    assert result["score"] == result["semantic_similarity"]
    assert {"audit", "general", "ledger"} <= set(result["common_words"])
    assert result["engine"] == "sentence-transformers (cosine similarity)"
    assert "matched_keywords" not in result
    assert "missing_keywords" not in result

def test_matching_uses_embedding_similarity_only(monkeypatch):
    encoded_texts = []

    def fake_encode_texts(texts):
        encoded_texts.extend(texts)
        return np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)

    monkeypatch.setattr(M, "encode_texts", fake_encode_texts)
    result = M.match("Python CV technique", "Python fiche de poste")

    assert encoded_texts == ["python cv technique", "python fiche de poste"]
    assert result == {
        "score": 0.8,
        "semantic_similarity": 0.8,
        "common_words": ["python"],
        "engine": "sentence-transformers (cosine similarity)",
    }

def test_match_service_delegates_to_embedding_match(monkeypatch):
    captured = {}

    def fake_match(cv_text, job_description):
        captured.update(cv_text=cv_text, job_description=job_description)
        return {"score": 0.75}

    monkeypatch.setattr(S, "match", fake_match)

    assert S.match_cv("cv", "fiche") == {"score": 0.75}
    assert captured == {"cv_text": "cv", "job_description": "fiche"}

def test_genai_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(G, "ENV_FILE", tmp_path / "missing.env")
    assert "Missions" in generate_job_posting("Data Scientist", skills=["python"])["text"]

def test_job_posting_pdf_contains_generated_content():
    pdf = create_job_posting_pdf(
        "Data Scientist",
        "# Data Scientist\n**Entreprise :** Acme Analytics\n\n"
        "## Profil recherché\nBac+5 en informatique, français et anglais.",
    )
    assert pdf.startswith(b"%PDF-")
    text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
    assert "Data Scientist" in text
    assert "Acme Analytics" in text
    assert "Bac+5" in text

def test_job_posting_pdf_cleans_markdown_and_intro():
    pdf = create_job_posting_pdf(
        "Data Scientist",
        "Voici la fiche de poste détaillée pour le poste de Data Scientist.\n"
        "***\n\n### Fiche de poste — Data Scientist\n\n"
        "#### Missions\n- Analyser les données et présenter les résultats.",
    )
    text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)

    assert "Voici la fiche" not in text
    assert "***" not in text
    assert "###" not in text
    assert "Fiche de poste" in text
    assert "Missions" in text

def test_genai_uses_configured_model_and_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
    captured = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "Fiche générée"}]},
                    "finishReason": "STOP",
                }]
            }).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(G.urllib.request, "urlopen", fake_urlopen)
    result = generate_job_posting(
        "Data Analyst",
        domain="Finance",
        skills=["sql"],
        level="Senior",
        company="Acme Analytics",
        contract_type="CDI",
        location="Paris",
        work_mode="Hybride",
        education="Bac+5",
        languages=["français", "anglais"],
        salary_min=45000,
        salary_max=60000,
    )

    assert result == {"engine": "gemini:gemini-3.5-flash-lite", "text": "Fiche générée"}
    assert "test-key-not-a-real-secret" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "test-key-not-a-real-secret"
    assert captured["body"]["generationConfig"]["maxOutputTokens"] == 2048
    prompt = captured["body"]["contents"][0]["parts"][0]["text"]
    for detail in ("Acme Analytics", "CDI", "Paris", "Hybride", "Bac+5",
                   "français, anglais", "45 000 à 60 000 €/an"):
        assert detail in prompt

def test_genai_reports_truncated_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "Fiche incomplète"}]},
                    "finishReason": "MAX_TOKENS",
                }]
            }).encode()

    monkeypatch.setattr(G.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    result = generate_job_posting("Data Analyst")

    assert result["engine"] == "template"
    assert "coupé sa réponse" in result["error"]

def test_genai_reports_http_errors_without_hiding_them(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")

    def failed_request(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, BytesIO())

    monkeypatch.setattr(G.urllib.request, "urlopen", failed_request)
    result = generate_job_posting("Data Analyst")

    assert result["engine"] == "template"
    assert "HTTP 403" in result["error"]
    assert "clé" in result["error"]
    assert "Missions" in result["text"]

def test_genai_retries_temporary_503(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
    attempts = []
    def failed_request(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.HTTPError("https://example.test", 503, "Unavailable", {}, BytesIO())

    monkeypatch.setattr(G.urllib.request, "urlopen", failed_request)
    monkeypatch.setattr(G.time, "sleep", lambda _: None)
    result = generate_job_posting("Data Analyst")

    assert len(attempts) == 3
    assert "HTTP 503" in result["error"]
    assert "indisponible" in result["error"]

def test_genai_handles_read_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
    attempts = []
    def timed_out(*args, **kwargs):
        attempts.append(1)
        raise TimeoutError("read timed out")

    monkeypatch.setattr(G.urllib.request, "urlopen", timed_out)
    monkeypatch.setattr(G.time, "sleep", lambda _: None)
    result = generate_job_posting("Data Analyst")

    assert len(attempts) == 2
    assert "n'a pas répondu" in result["error"]
    assert "Missions" in result["text"]

def test_api(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(G, "ENV_FILE", tmp_path / "missing.env")
    c = app.test_client()
    assert c.get("/health").json["status"] == "ok"
    assert c.post("/classify", json={"text": CV}).status_code == 200
    assert c.post("/classify", json={"text": "x"}).status_code == 400
    invalid_pdf = c.post(
        "/classify",
        data={"file": (BytesIO(b"not a pdf"), "cv.pdf")},
        content_type="multipart/form-data",
    )
    assert invalid_pdf.status_code == 400
    job = "Comptable Général : accounting, general ledger, reconciliation, excel, audit and reporting."
    assert c.post("/match", json={"cv_text": CV, "job_description": job}).status_code == 200
    assert c.post("/match", json={"cv_text": CV}).status_code == 400
    assert c.post("/job-posting", json={"title": "Comptable"}).status_code == 200
    posting = c.post("/job-posting", json={
        "title": "Data Analyst",
        "company": "Acme Analytics",
        "contract_type": "CDI",
        "salary_min": 45000,
        "salary_max": 60000,
    })
    assert posting.status_code == 200
    assert "Acme Analytics" in posting.json["text"]
    assert "45 000 à 60 000 €/an" in posting.json["text"]
    assert c.post("/salary", json={}).status_code == 400
    salary_data = {
        "age": 35,
        "education": S.options()["education"][0],
        "job_title": S.options()["titles"][0],
        "years": 5,
    }
    assert c.post("/salary", json=salary_data).status_code == 200

def test_metrics_gate():
    m = json.loads(C.METRICS_JSON.read_text())
    assert m["cv_classifier"]["lvl2_macro_f1"] >= C.MIN_F1_CV and m["salary"]["r2"] >= C.MIN_R2_SALARY
