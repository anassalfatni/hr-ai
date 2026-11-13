"""Génère un jeu salarial synthétique au schéma de 'Salary Data.csv' (Kaggle).
Remplacé automatiquement par le vrai fichier s'il est déposé dans data/."""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SALARY_SYNTH, SEED

rng = np.random.default_rng(SEED)
base = {"Software Engineer": 85000, "Data Scientist": 95000, "Data Analyst": 65000, "Product Manager": 100000,
        "Marketing Manager": 75000, "Sales Associate": 45000, "HR Manager": 70000, "Financial Analyst": 70000,
        "Accountant": 58000, "Project Manager": 90000, "Graphic Designer": 52000, "Operations Manager": 85000,
        "Customer Service Rep": 38000, "Business Analyst": 72000, "Network Engineer": 78000}
edu = {"Bachelor's": 1.0, "Master's": 1.15, "PhD": 1.3}
n = 1500
jobs = rng.choice(list(base), n); ed = rng.choice(list(edu), n, p=[.5, .35, .15])
yoe = np.clip(rng.gamma(3, 3, n), 0, 35).round(1); age = np.clip(22 + yoe + rng.normal(3, 3, n), 21, 65).round()
sal = [base[j] * edu[e] * (1 + .035 * y) * rng.normal(1, .06) for j, e, y in zip(jobs, ed, yoe)]
df = pd.DataFrame({"Age": age, "Gender": rng.choice(["Male", "Female"], n), "Education Level": ed,
                   "Job Title": jobs, "Years of Experience": yoe, "Salary": np.round(sal, -2)})
df.to_csv(SALARY_SYNTH, index=False); print("écrit", SALARY_SYNTH, df.shape)
