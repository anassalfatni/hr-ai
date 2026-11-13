"""Pré-traitement texte + masquage des données personnelles (classification : DONNÉES PERSONNELLES)."""
from io import BytesIO
import html
import re
import warnings

import pandas as pd
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import CAT2GROUP, RESUME_PDF_DIR

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

def mask_pii(t: str) -> str:
    return PHONE.sub(" ", EMAIL.sub(" ", t))

def clean_text(t: str) -> str:
    t = html.unescape(str(t))
    t = mask_pii(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^\w+#.\s]", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()

def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            raise ValueError("Le PDF est protégé par mot de passe.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as exc:
        raise ValueError("Le fichier n'est pas un PDF lisible.") from exc
    if not text.strip():
        raise ValueError("Aucun texte détecté dans le PDF. Les PDF scannés nécessitent un OCR.")
    return text

def load_resumes() -> pd.DataFrame:
    if not RESUME_PDF_DIR.is_dir():
        raise FileNotFoundError(f"Dossier de CV introuvable : {RESUME_PDF_DIR}")

    category_dirs = sorted(p for p in RESUME_PDF_DIR.iterdir() if p.is_dir())
    if not category_dirs:
        raise ValueError(f"Aucun dossier de catégorie dans {RESUME_PDF_DIR}")

    unknown = sorted(p.name for p in category_dirs if p.name not in CAT2GROUP)
    if unknown:
        raise ValueError(f"Catégories absentes de HIERARCHY dans src/config.py : {', '.join(unknown)}")

    records = []
    unreadable = 0
    too_short = 0
    for category_dir in category_dirs:
        pdfs = sorted(
            p for p in category_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        )
        for pdf_path in pdfs:
            try:
                raw_text = extract_pdf_text(pdf_path.read_bytes())
            except (OSError, ValueError):
                unreadable += 1
                continue
            text = clean_text(raw_text)
            if len(text) <= 100:
                too_short += 1
                continue
            records.append({"text": text, "Category": category_dir.name})

    if not records:
        raise ValueError("Aucun CV PDF exploitable trouvé dans les dossiers de catégories.")

    if unreadable or too_short:
        warnings.warn(
            f"CV PDF ignorés : {unreadable} illisibles, chiffrés ou sans texte extractible "
            f"(OCR requis pour les scans), {too_short} avec moins de 100 caractères.",
            RuntimeWarning,
            stacklevel=2,
        )

    df = pd.DataFrame(records).drop_duplicates("text")
    df["group"] = df["Category"].map(CAT2GROUP)
    return df[["text", "Category", "group"]].reset_index(drop=True)
