"""APPROCHE 1 — Modèles LOCAUX entraînés sur nos données.
 - Tâche 1 : classification hiérarchique de CV (texte) : domaine (N1) -> spécialité (N2)
 - Tâche 2 : prédiction de salaire (tabulaire)
Sortie : models/*.joblib + models/metrics.json (consommé par la CI comme quality gate)."""
import json, sys, joblib
import mlflow
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix, hstack
from . import config as C
from .embeddings import encode_texts
from .preprocess import load_resumes

def train_cv():
    """Entraîne un classifieur supervisé sur les signaux lexicaux et sémantiques."""
    print("Chargement des CV PDF...", flush=True)
    df = load_resumes()
    tr, te = train_test_split(df, test_size=.2, stratify=df["Category"], random_state=C.SEED)
    print(f"CV prêts : {len(tr)} entraînement, {len(te)} test.", flush=True)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=.9,
                            sublinear_tf=True, max_features=60000)
    lexical_train = tfidf.fit_transform(tr["text"])
    lexical_test = tfidf.transform(te["text"])
    print(f"Encodage sémantique du jeu d'entraînement ({len(tr)} CV)...", flush=True)
    semantic_train = encode_texts(tr["text"].tolist())
    print(f"Encodage sémantique du jeu de test ({len(te)} CV)...", flush=True)
    semantic_test = encode_texts(te["text"].tolist())
    x_train = hstack(
        [lexical_train, csr_matrix(semantic_train * C.CV_EMBEDDING_WEIGHT)],
        format="csr",
    )
    x_test = hstack(
        [lexical_test, csr_matrix(semantic_test * C.CV_EMBEDDING_WEIGHT)],
        format="csr",
    )
    classifier = LinearSVC(C=0.5, class_weight="balanced", random_state=C.SEED).fit(x_train, tr["Category"])
    print("Évaluation et sauvegarde du classifieur CV...", flush=True)
    pc = classifier.predict(x_test); pg = [C.CAT2GROUP[c] for c in pc]
    m = {"n_train": len(tr), "n_test": len(te),
         "lvl1_accuracy": round(accuracy_score(te["group"], pg), 4),
         "lvl1_macro_f1": round(f1_score(te["group"], pg, average="macro"), 4),
         "lvl2_accuracy": round(accuracy_score(te["Category"], pc), 4),
         "lvl2_macro_f1": round(f1_score(te["Category"], pc, average="macro"), 4),
         "embedding_model": C.EMBEDDING_MODEL,
         "embedding_weight": C.CV_EMBEDDING_WEIGHT}
    joblib.dump(
        {"tfidf": tfidf, "classifier": classifier, "embedding_weight": C.CV_EMBEDDING_WEIGHT},
        C.MODELS / "cv_classifier.joblib",
    )
    return m

def predict_hier(model, text):
    lexical = model["tfidf"].transform([text])
    semantic = encode_texts([text])
    features = hstack(
        [lexical, csr_matrix(semantic * model["embedding_weight"])],
        format="csr",
    )
    c = model["classifier"].predict(features)[0]
    return C.CAT2GROUP[c], c

def load_salary() -> pd.DataFrame:
    p = C.SALARY_CSV if C.SALARY_CSV.exists() else C.SALARY_SYNTH
    df = pd.read_csv(p)
    required = FEATS + ["Salary"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {p.name}: {', '.join(missing)}")
    df = df.dropna(subset=required)
    df.attrs["source"] = p.name
    return df

FEATS = ["Age", "Education Level", "Job Title", "Years of Experience"]

def train_salary():
    print("Entraînement du modèle salarial...", flush=True)
    df = load_salary()
    X, y = df[FEATS], df["Salary"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=C.SEED)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["Education Level", "Job Title"])],
                            remainder="passthrough")
    pipe = Pipeline([("pre", pre), ("gb", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05, random_state=C.SEED))]).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    joblib.dump({"pipe": pipe, "titles": sorted(df["Job Title"].unique()), "edu": sorted(df["Education Level"].unique())}, C.MODELS / "salary.joblib")
    return {"source": df.attrs["source"], "n_rows": len(df), "mae": round(mean_absolute_error(yte, p), 1), "r2": round(r2_score(yte, p), 4)}


def _log_training_run(metrics, quality_gate_passed):
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="train-models"):
        mlflow.log_params({
            "seed": C.SEED,
            "embedding_model": C.EMBEDDING_MODEL,
            "cv_embedding_weight": C.CV_EMBEDDING_WEIGHT,
            "cv_classifier": "LinearSVC",
            "cv_classifier_c": 0.5,
            "cv_class_weight": "balanced",
            "cv_tfidf_ngram_range": "1,2",
            "cv_tfidf_min_df": 3,
            "cv_tfidf_max_df": 0.9,
            "cv_tfidf_max_features": 60000,
            "salary_regressor": "GradientBoostingRegressor",
            "salary_n_estimators": 300,
            "salary_max_depth": 3,
            "salary_learning_rate": 0.05,
        })

        logged_metrics = {}
        for section, values in metrics.items():
            for name, value in values.items():
                if isinstance(value, (int, float, np.number)):
                    logged_metrics[f"{section}_{name}"] = float(value)
                elif isinstance(value, str):
                    mlflow.log_param(f"{section}_{name}", value)
        mlflow.log_metrics(logged_metrics)
        mlflow.set_tag("quality_gate", "passed" if quality_gate_passed else "failed")

        mlflow.log_artifact(str(C.MODELS / "cv_classifier.joblib"), artifact_path="models")
        mlflow.log_artifact(str(C.MODELS / "salary.joblib"), artifact_path="models")
        mlflow.log_artifact(str(C.METRICS_JSON), artifact_path="metrics")


def main():
    C.MODELS.mkdir(exist_ok=True)
    print(f"Modèle d'embeddings : {C.EMBEDDING_MODEL}", flush=True)
    metrics = {"cv_classifier": train_cv(), "salary": train_salary()}
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    ok = metrics["cv_classifier"]["lvl2_macro_f1"] >= C.MIN_F1_CV and metrics["salary"]["r2"] >= C.MIN_R2_SALARY
    _log_training_run(metrics, ok)
    print("QUALITY GATE:", "OK" if ok else "ECHEC", flush=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
