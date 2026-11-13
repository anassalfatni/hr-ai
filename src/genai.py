"""APPROCHE 3 — IA GÉNÉRATIVE (API Gemini via Google AI Studio) : génération de fiches de poste.
Sans GEMINI_API_KEY ou hors-ligne : gabarit déterministe (le service reste disponible)."""
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

from dotenv import dotenv_values, load_dotenv

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE)

URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_SECONDS = 45

def _salary_text(salary=None, salary_min=None, salary_max=None):
    def format_amount(value):
        return f"{value:,.0f}".replace(",", " ")

    if salary_min is not None and salary_max is not None:
        return f"{format_amount(salary_min)} à {format_amount(salary_max)} €/an"
    if salary_min is not None:
        return f"à partir de {format_amount(salary_min)} €/an"
    if salary_max is not None:
        return f"jusqu'à {format_amount(salary_max)} €/an"
    if salary is not None:
        return f"{format_amount(salary)} €/an"
    return ""


def _prompt(title, domain, skills, level, salary, company="", contract_type="",
            location="", work_mode="", education="", languages=None,
            salary_min=None, salary_max=None):
    details = [
        f"Entreprise : {company}" if company else "Entreprise : non précisée, ne pas en inventer.",
        f"Type de contrat : {contract_type}" if contract_type else "",
        f"Lieu : {location}" if location else "",
        f"Mode de travail : {work_mode}" if work_mode else "",
        f"Formation ou diplôme requis : {education}" if education else "",
        f"Langues requises : {', '.join(languages)}" if languages else "",
    ]
    salary_range = _salary_text(salary, salary_min, salary_max)
    if salary_range:
        details.append(f"Rémunération : {salary_range}")
    detail_text = "\n".join(detail for detail in details if detail)
    return (f"Rédige une fiche de poste professionnelle en français pour « {title} » "
            f"(domaine : {domain or 'à préciser'}, niveau : {level}).\n"
            f"{detail_text}\n"
            f"Compétences clés : {', '.join(skills) or 'à préciser'}.\n"
            "Rédige une fiche complète et détaillée d'environ 400 à 600 mots. "
            "Inclus les sections suivantes : Intitulé, Entreprise, Département et niveau, "
            "Missions (5 puces détaillées), Profil recherché, Compétences, Conditions. "
            "Commence directement par le titre principal en Markdown, sans phrase d'introduction "
            "ni ligne de séparation. Utilise des titres Markdown pour structurer les sections. "
            "Ne t'arrête pas au milieu d'une phrase et termine toutes les sections. "
            "N'invente pas les informations qui ne sont pas fournies. "
            "Ton neutre et inclusif, sans critère discriminatoire (âge, genre, origine).")


def _template(title, domain, skills, level, salary, company="", contract_type="",
              location="", work_mode="", education="", languages=None,
              salary_min=None, salary_max=None):
    s = "\n".join(f"- {x}" for x in skills) or "- À préciser"
    details = [f"**Entreprise :** {company}" if company else "",
               f"**Type de contrat :** {contract_type}" if contract_type else "",
               f"**Lieu :** {location}" if location else "",
               f"**Mode de travail :** {work_mode}" if work_mode else "",
               f"**Formation requise :** {education}" if education else "",
               f"**Langues :** {', '.join(languages)}" if languages else ""]
    salary_range = _salary_text(salary, salary_min, salary_max)
    if salary_range:
        details.append(f"**Rémunération indicative :** {salary_range}")
    metadata = "\n".join(detail for detail in details if detail)
    return (f"# {title}\n**Domaine :** {domain} — **Niveau :** {level}\n{metadata}\n\n## Missions\n- Prendre en charge les activités clés du poste de {title}\n"
            f"- Collaborer avec les équipes transverses\n- Suivre et reporter les indicateurs de performance\n- Contribuer à l'amélioration continue\n"
            f"- Assurer la conformité aux règles internes\n\n## Compétences\n{s}\n\n"
            "_(Version gabarit : définir GEMINI_API_KEY pour la génération IA.)_")


def _fallback(title, domain, skills, level, salary, error=None, **details) -> dict:
    result = {
        "engine": "template",
        "text": _template(title, domain, skills, level, salary, **details),
    }
    if error:
        result["error"] = error
    return result

def _http_error_message(status: int) -> str:
    if status == 503:
        return "Gemini est temporairement indisponible (HTTP 503). Réessaie dans quelques minutes."
    if status == 429:
        return "Limite ou quota Gemini atteint (HTTP 429). Attends puis réessaie."
    if status in (401, 403):
        return f"Gemini a refusé la requête (HTTP {status}). Vérifie la clé et son accès à l'API."
    if status == 404:
        return "Le modèle Gemini demandé est introuvable (HTTP 404)."
    if 500 <= status:
        return f"Le service Gemini rencontre un problème temporaire (HTTP {status}). Réessaie plus tard."
    return f"Gemini a refusé la requête (HTTP {status}). Vérifie les paramètres de la requête."

def generate_job_posting(title, domain="", skills=None, level="Confirmé", salary=None,
                         model=DEFAULT_MODEL, company="", contract_type="",
                         location="", work_mode="", education="", languages=None,
                         salary_min=None, salary_max=None) -> dict:
    skills = skills or []
    if isinstance(languages, str):
        languages = [language.strip() for language in languages.split(",") if language.strip()]
    else:
        languages = [language.strip() for language in (languages or []) if language.strip()]
    details = {
        "company": company,
        "contract_type": contract_type,
        "location": location,
        "work_mode": work_mode,
        "education": education,
        "languages": languages,
        "salary_min": salary_min,
        "salary_max": salary_max,
    }
    key = os.getenv("GEMINI_API_KEY") or dotenv_values(ENV_FILE).get("GEMINI_API_KEY")
    if key:
        body = json.dumps({
            "contents": [{
                "parts": [{
                    "text": _prompt(title, domain, skills, level, salary, **details)
                }]
            }],
            "generationConfig": {"maxOutputTokens": 2048},
        }).encode()
        req = urllib.request.Request(
            URL.format(m=model),
            body,
            {"Content-Type": "application/json", "x-goog-api-key": key},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as r:
                    response = json.load(r)
                candidate = response["candidates"][0]
                if candidate.get("finishReason") == "MAX_TOKENS":
                    return _fallback(
                        title, domain, skills, level, salary,
                        "Gemini a atteint la limite de longueur et a coupé sa réponse. Réessaie.",
                        **details,
                    )
                txt = "".join(
                    part.get("text", "")
                    for part in candidate["content"]["parts"]
                    if isinstance(part.get("text"), str)
                )
                if not txt.strip():
                    return _fallback(
                        title, domain, skills, level, salary,
                        "Gemini n'a renvoyé aucun texte. Réessaie.",
                        **details,
                    )
                return {"engine": f"gemini:{model}", "text": txt}
            except TimeoutError:
                if attempt < 1:
                    time.sleep(1 + random.uniform(0, 0.5))
                    continue
                return _fallback(
                    title, domain, skills, level, salary,
                    f"Gemini n'a pas répondu après {REQUEST_TIMEOUT_SECONDS} secondes. Réessaie plus tard.",
                    **details,
                )
            except urllib.error.HTTPError as e:
                retryable = e.code in (408, 429) or e.code >= 500
                if retryable and attempt < 2:
                    time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                    continue
                return _fallback(
                    title, domain, skills, level, salary, _http_error_message(e.code), **details
                )
            except urllib.error.URLError:
                if attempt < 2:
                    time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                    continue
                return _fallback(
                    title, domain, skills, level, salary,
                    "Gemini est inaccessible. Vérifie la connexion réseau et réessaie.",
                    **details,
                )
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                return _fallback(
                    title, domain, skills, level, salary,
                    "Gemini a renvoyé une réponse inattendue.",
                    **details,
                )
    return _fallback(title, domain, skills, level, salary, **details)
