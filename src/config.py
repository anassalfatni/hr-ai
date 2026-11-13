import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"
RESUME_PDF_DIR = DATA / "data_CV"
SALARY_CSV = DATA / "Salary Data.csv"          # vrai fichier du prof (si présent)
SALARY_SYNTH = DATA / "salary_synthetic.csv"   # repli synthétique, même schéma
METRICS_JSON = MODELS / "metrics.json"
MIN_F1_CV = 0.58       # quality gate CI : macro-F1 niveau 2
MIN_R2_SALARY = 0.80   # quality gate CI : R2 salaire
SEED = 42
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CV_EMBEDDING_WEIGHT = 0.25

# Niveau 1 (domaine large) -> Niveau 2 (spécialité = Category du dataset)
HIERARCHY = {
    "Tech & Digital": ["INFORMATION-TECHNOLOGY", "DIGITAL-MEDIA", "DESIGNER"],
    "Finance & Juridique": ["FINANCE", "ACCOUNTANT", "BANKING", "ADVOCATE"],
    "Business & Relation client": ["BUSINESS-DEVELOPMENT", "SALES", "CONSULTANT", "PUBLIC-RELATIONS", "BPO", "HR"],
    "Industrie & Terrain": ["ENGINEERING", "CONSTRUCTION", "AUTOMOBILE", "AVIATION", "AGRICULTURE"],
    "Santé, Éducation & Services": ["HEALTHCARE", "TEACHER", "FITNESS", "CHEF", "ARTS", "APPAREL"],
}
CAT2GROUP = {c: g for g, cs in HIERARCHY.items() for c in cs}