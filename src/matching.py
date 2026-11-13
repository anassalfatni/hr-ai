"""Matching CV / fiche de poste par similarité cosinus des embeddings."""
import html
import re
from collections import Counter
 
import numpy as np
 
from .embeddings import encode_texts
from .preprocess import clean_text, mask_pii
 
 
STOPWORDS = set("""
a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me même mes moi mon ne nos notre nous
on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l à m n s t y été être
est sont sera serez êtes avez avoir ont fait faire plus tout tous toute toutes très aussi comme ainsi donc car si
sans sous entre chez vers dont où après avant pendant cette cet ci là notamment afin lors
a an and are as at be been but by for from has have in into is it its of on or our that the their them they this to
was were will with you your we us who which what when where how not than then there these those such can may must
should would could also more most other some any all each per via etc
poste offre mission missions profil recherché année années ans expérience experience job description
candidate candidates required requirements technical skill skills department departments internal provide provided
""".split())
 
TOKEN = re.compile(r"[^\W\d_][\w+#.]*")
SKILL_SECTION = re.compile(
    r"\b(?:skills?|competenc(?:e|es|y|ies)|qualifications?|requirements?|knowledge|profil recherché)\b",
    re.IGNORECASE,
)
NON_SKILL_SECTION = re.compile(
    r"\b(?:responsibilities|duties|missions?|what you(?:'ll| will) do|"
    r"job description|about (?:the )?role|role overview|description du poste)\b",
    re.IGNORECASE,
)


def _candidate_text(text: str) -> str:
    text = html.unescape(str(text))
    text = mask_pii(text)
    return re.sub(r"<[^>]+>", " ", text)


def _select_skill_section(text: str) -> tuple[str, bool]:
    raw_text = _candidate_text(text)
    selected_lines = []
    in_skill_section = False
    found_skill_section = False

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_skill_section:
                selected_lines.append("")
            continue

        is_heading = stripped.startswith("#")
        heading = re.sub(r"^#{1,6}\s*", "", stripped).strip()
        is_bullet = heading.startswith(("-", "*", "•"))
        if is_bullet:
            heading = heading[1:].strip()

        inline = re.match(r"^(.{1,80}?):\s*(.*)$", heading)
        label = inline.group(1).strip() if inline else heading
        if not is_bullet and len(label.split()) <= 8:
            if SKILL_SECTION.search(label):
                found_skill_section = True
                in_skill_section = True
                if inline and inline.group(2).strip():
                    selected_lines.append(inline.group(2).strip())
                continue
            if NON_SKILL_SECTION.search(label) or is_heading:
                in_skill_section = False
                continue

        if in_skill_section:
            selected_lines.append(stripped)

    selected_text = "\n".join(selected_lines).strip()
    if found_skill_section and selected_text:
        return selected_text, True
    return raw_text, False
 
 
def _tokens(text: str) -> list[str]:
    return [t.strip(".").lower() for t in TOKEN.findall(text) if len(t.strip(".")) > 1]
 
 
def _candidates(text: str, max_n: int = 3, limit: int = 150) -> list[str]:
    """N-grammes de mots 'pleins' (aucun mot vide), les plus fréquents d'abord."""
    counts = Counter()
    segments = re.split(r"[\r\n,;:!?]+|(?<=[.!?])\s+", _candidate_text(text))
    for segment in segments:
        toks = _tokens(segment)
        for n in range(1, max_n + 1):
            for i in range(len(toks) - n + 1):
                gram = toks[i:i + n]
                if any(t in STOPWORDS or len(t) < 2 for t in gram):
                    continue
                counts[" ".join(gram)] += 1
    ranked = sorted(counts, key=lambda g: (-counts[g], -len(g)))
    return ranked[:limit]
 
 
def _mmr(doc_vec: np.ndarray, vecs: np.ndarray, top_n: int, diversity: float) -> list[int]:
    rel = vecs @ doc_vec
    selected = [int(rel.argmax())]
    while len(selected) < min(top_n, len(vecs)):
        redundancy = (vecs @ vecs[selected].T).max(axis=1)
        score = (1 - diversity) * rel - diversity * redundancy
        score[selected] = -np.inf
        selected.append(int(score.argmax()))
    return selected
 
 
def _top_keywords(text: str, top_n: int = 15, diversity: float = 0.5):
    """Retourne (mots-clés, vecteurs) des expressions les plus représentatives du texte."""
    cands = _candidates(text)
    if not cands:
        return [], np.empty((0, 0), dtype=np.float32)
    vecs = encode_texts(cands)
    doc_vec = encode_texts([clean_text(text)])[0]
    idx = _mmr(doc_vec, vecs, top_n, diversity)
    return [cands[i] for i in idx], vecs[idx]
 
 
def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    """Utilisé aussi par services.classify_cv pour afficher les compétences détectées."""
    keyword_text, _ = _select_skill_section(text)
    return _top_keywords(keyword_text, top_n)[0]
 
 
def match(cv_text: str, job_description: str) -> dict:
    cv, job = clean_text(cv_text), clean_text(job_description)
    if not cv or not job:
        raise ValueError("Le CV et la fiche de poste doivent contenir du texte exploitable.")

    cv_vector, job_vector = encode_texts([cv, job])
    similarity = float(cv_vector @ job_vector)
    score = min(1.0, max(0.0, similarity))
    common_words = sorted((set(_tokens(cv)) & set(_tokens(job))) - STOPWORDS)
    return {
        "score": round(score, 3),
        "semantic_similarity": round(similarity, 3),
        "common_words": common_words,
        "engine": "sentence-transformers (cosine similarity)",
    }