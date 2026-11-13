# HR-AI — Assistant RH intelligent (CV + données salariales)

## 1. Contexte & fonctionnalités (livrable 1)
Cabinet de recrutement / service RH qui reçoit des centaines de CV. L'application :
| # | Fonctionnalité | Donnée | Approche |
|---|---|---|---|
| 1 | Classer un CV par **domaine** puis **spécialité** (multi-niveaux) | PDF du CV | **1. TF-IDF + embeddings pré-entraînés, classifieur local LinearSVC** |
| 2 | **Prédire le salaire** d'un profil | Tabulaire (`Salary Data.csv`) | **1. Modèle local** (Gradient Boosting, sans variable de genre) |
| 3 | **Matching CV ↔ fiche de poste fournie** + mots-clés manquants | Texte (CV + fiche de poste) | **2. Modèle pré-entraîné multilingue** (embeddings sentence-transformers) |
| 4 | **Génération de fiches de poste** | Paramètres du poste, PDF téléchargeable | **3. IA générative** (API Gemini) |

Deux types de données : **CV PDF** rangés dans `data/data_CV/<CATEGORIE>/` (un dossier par catégorie) et **tabulaire** (salaires : âge, diplôme, poste, expérience, salaire). Le genre éventuellement présent dans le CSV salarial n'est pas utilisé par le modèle.

## 2. Classification des données (sensibilisation demandée par le prof)
| Donnée | Niveau | Traitement |
|---|---|---|
| CV (nom, e-mail, tel.) | **Personnelle / critique** (RGPD) | E-mails & téléphones masqués avant tout traitement ; **jamais envoyés à l'API GenAI**. Seuls les paramètres de la fiche de poste saisis dans l'onglet de génération sont envoyés. |
| Salaires | **Confidentielle** | Agrégés, pas d'identité |
| Offres d'emploi | Publique / interne | Libre |
| Fiches générées | Interne | Relecture humaine |
Éthique : la variable de genre est exclue du modèle salarial ; l'outil **assiste**, la décision d'embauche reste humaine (AI Act : recrutement = usage « haut risque »).

## 3. Architecture
```
Pour voir immédiatement les étapes d'entraînement dans PowerShell, utilise plutôt `.\.venv\Scripts\python.exe -u -m src.train` (`-u` désactive le tampon de sortie Python).
data_CV/<CATEGORIE>/*.pdf ─► extraction + preprocess ─► TF-IDF + embeddings Sentence Transformers ─► LinearSVC entraîné ─► models/*.joblib + metrics.json
Salary.csv ─►                                                       │ (quality gate)
                                                         src/services.py  (couche unique)
                                                          ├─ app/api.py  Flask  : /classify /salary /match /job-posting /metrics /health
                                                          └─ app/streamlit_app.py : UI 5 onglets
Matching : similarité cosinus des embeddings du CV et de la fiche complète │ GenAI : Gemini API (repli gabarit)
```
## 4. MLOps (automatisation obligatoire)
- `make all` : données → entraînement → tests. `src/train.py` **échoue (exit 1) si macro-F1 < 0.58 ou R² < 0.80** → la CI bloque un mauvais modèle.
- **GitHub Actions** (`.github/workflows/mlops.yml`) : à chaque push + **ré-entraînement hebdomadaire** : install → train + gate → pytest → artefacts → build Docker.
- **Docker / docker-compose** : l'image entraîne les modèles au build (modèle versionné avec le code), healthcheck, API gunicorn + UI.
- Pas de Colab : serving API local/Docker, comme recommandé.

## 5. Lancer
```bash
pip install -r requirements.txt
make all            # données, entraînement, tests
make api            # http://localhost:5000     (POST /classify avec {"text": "..."})
make ui             # http://localhost:8501
```
Pour Gemini en local, ouvre `.env` à la racine du projet et colle ta clé après `GEMINI_API_KEY=` (les guillemets autour de la valeur sont facultatifs). Ce fichier est ignoré par Git. Une copie sans secret est fournie dans `.env.example`. Le projet charge automatiquement le fichier et utilise `gemini-3.5-flash-lite`, modèle adapté aux tâches à faible latence. La clé reste sur ta machine ; ne la colle pas dans le chat et ne la commits pas. Les erreurs temporaires (HTTP 429 ou 5xx) sont réessayées jusqu'à trois fois avec attente progressive ; les délais de réponse déclenchent une seconde tentative puis un message adapté. La disponibilité gratuite et les quotas dépendent du compte et des conditions Google AI Studio.
Le modèle d'embeddings utilisé est `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, téléchargé gratuitement depuis Hugging Face puis exécuté localement (pas d'envoi des CV à Hugging Face). Une connexion Internet est nécessaire au premier téléchargement. Les CV longs sont découpés en segments pour ne pas perdre le texte au-delà de la limite du modèle. Pour la classification, le LinearSVC combine TF-IDF et ces embeddings et apprend les catégories à partir des CV classés dans `data/data_CV/`. `all-MiniLM-L6-v2` n'est pas retenu car il est destiné à l'anglais, alors que les CV et fiches visés sont en français.
Pour utiliser le vrai fichier salaire : déposer `Salary Data.csv` dans `data/` (colonnes requises : Age, Education Level, Job Title, Years of Experience, Salary ; une colonne Gender supplémentaire est ignorée), puis `make train`. L'API `POST /salary` attend `age`, `education`, `job_title` et `years`.

Pour classer un CV, téléverser un PDF dans l'onglet « Classer un CV ». Le matching accepte aussi un PDF avec la fiche de poste collée dans l'onglet correspondant. L'API accepte un PDF dans `file` sur `/classify`, ou dans `cv_file` avec `job_description` sur `/match`. Les PDF scannés sans texte sélectionnable nécessitent un OCR, qui n'est pas inclus.

Le matching encode le texte complet du CV et de la fiche avec le modèle multilingue puis calcule leur similarité cosinus. Il affiche aussi les termes exacts présents dans les deux textes (les synonymes ou variantes ne sont pas inclus dans cette liste). Le score reste une mesure globale indicative, et non une validation de chaque compétence ni une décision de recrutement.

Dans l'onglet « Fiche de poste », renseigner le niveau, l'entreprise, le contrat, le lieu, le mode de travail, la formation, les langues et éventuellement une fourchette salariale. La fiche est générée et proposée au téléchargement au format PDF. Les paramètres renseignés sont transmis à Gemini ; aucun CV n'est utilisé pour cette génération.

## 6. Résultats (`models/metrics.json`)
- CV PDF (24 catégories) : métriques recalculées lors de l'entraînement avec le modèle multilingue ; un PDF sans texte extractible est ignoré (OCR non configuré).
- Salaire (`Salary Data.csv`, 373 lignes, sans genre) : MAE **10 882.3**, R² **0.8627**.
- Expérience menée : une cascade domaine → spécialité (deux niveaux entraînés) donne un F1 spécialité **plus faible** (0.56) qu'un modèle direct : les erreurs du niveau 1 se propagent. On a donc gardé le modèle direct + domaine déduit.
- Limites : classes déséquilibrées (BPO 22 CV, AUTOMOBILE 36), catégories qui se chevauchent (FINANCE/BANKING/ACCOUNTANT) = plafond du F1.

## 7. Question finale du prof : quand choisir quelle approche ?
| | **Modèle local** | **Pré-entraîné** | **IA générative (API)** |
|---|---|---|---|
| Quand | Tâche **précise**, données étiquetées dispo : le classifieur LinearSVC utilise les catégories annotées ; également pour la prédiction de salaire | Tâche **générique** (similarité sémantique, embeddings) avec **peu/pas** de données étiquetées | Tâche de **génération/rédaction**, ou besoin de flexibilité sans entraînement |
| Coût | Entraînement CPU, inférence ≈ gratuite | Téléchargement du modèle, inférence modérée | Payant à l'usage, quota |
| Confidentialité | ✅ Données restent chez nous | ✅ Si exécuté en local | ⚠️ Données envoyées à un tiers → ne jamais y mettre de PII |
| Contrôle / reproductibilité | ✅ Total, explicable | Moyen | ❌ Non déterministe, hallucinations possibles |
| Latence | Très faible | Faible | Élevée |
**Notre réponse** : local pour classer et prédire (données propres, confidentielles, métriques mesurables) ; pré-entraîné pour le matching (comprend « comptable » ≈ « accounting » sans étiquettes) ; GenAI uniquement pour rédiger, avec relecture humaine et sans données personnelles.
