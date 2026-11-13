"""UI : streamlit run app/streamlit_app.py"""
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import streamlit as st
from src import services as S
from src.preprocess import extract_pdf_text
from src.job_posting_pdf import create_job_posting_pdf

st.set_page_config(page_title="HR-AI", layout="wide")
st.title("HR-AI — Assistant RH intelligent")
st.caption("⚠️ Données de CV = données personnelles : e-mails/téléphones masqués avant traitement. Décision finale = humaine.")
t1, t2, t3, t4, t5 = st.tabs(["① Classer un CV", "② Prédire un salaire", "③ Matching CV/offres", "④ Fiche de poste (GenAI)", "Métriques"])
with t1:
    cv_pdf = st.file_uploader("Téléversez le CV au format PDF", type=["pdf"], key="cvpdf1")
    if st.button("Classer"):
        if cv_pdf is None:
            st.warning("Veuillez téléverser un CV au format PDF.")
        else:
            try:
                cv_text = extract_pdf_text(cv_pdf.getvalue())
                r = S.classify_cv(cv_text)
                st.success(f"Domaine : **{r['domain']}** → Spécialité : **{r['specialty']}**")
                st.write("Compétences détectées :", r["keywords"])
            except ValueError as e:
                st.error(str(e))
with t2:
    o = S.options(); c = st.columns(4)
    age = c[0].number_input("Âge", 18, 70, 32)
    ed = c[1].selectbox("Diplôme", o["education"])
    jt = c[2].selectbox("Poste", o["titles"])
    yr = c[3].number_input("Expérience (ans)", 0.0, 40.0, 5.0)
    if st.button("Prédire"): st.metric("Salaire estimé", f"{S.predict_salary(age, ed, jt, yr)['salary_eur_equiv']:,.0f}")
with t3:
    cv_pdf = st.file_uploader("Téléversez le CV au format PDF", type=["pdf"], key="cvpdf3")
    job = st.text_area("Collez la fiche de poste", height=200, key="job3")
    if st.button("Comparer le CV et la fiche de poste"):
        if cv_pdf is None:
            st.warning("Veuillez téléverser un CV au format PDF.")
        elif len(job) < 50:
            st.warning("La fiche de poste doit contenir au moins 50 caractères.")
        else:
            try:
                cv_text = extract_pdf_text(cv_pdf.getvalue())
                m = S.match_cv(cv_text, job)
                st.metric("Similarité sémantique globale", f"{m['score']:.0%}")
                st.caption(f"Moteur : {m['engine']}")
                st.caption(
                    "Le score compare les embeddings des textes dans leur ensemble. "
                    "La liste ci-dessous contient uniquement les termes exacts communs; "
                    "les synonymes ou formes différentes ne sont pas inclus."
                )
                st.subheader("Mots présents dans le CV et la fiche")
                if m["common_words"]:
                    st.write(", ".join(m["common_words"]))
                else:
                    st.info("Aucun mot pertinent exact n'est commun aux deux textes.")
            except ValueError as e:
                st.error(str(e))
with t4:
    with st.form("job_posting_form"):
        a, b, c = st.columns(3)
        title = a.text_input("Intitulé", "Data Scientist")
        dom = b.text_input("Domaine", "Tech & Digital")
        level = c.selectbox(
            "Niveau d'expérience",
            ["Débutant", "Junior", "Confirmé", "Senior", "Expert"],
            index=2,
        )
        a, b = st.columns(2)
        company = a.text_input("Entreprise", placeholder="Nom de l'entreprise")
        contract_type = b.selectbox(
            "Type de contrat",
            ["À préciser", "CDI", "CDD", "Stage", "Alternance", "Freelance", "Intérim"],
        )
        a, b = st.columns(2)
        location = a.text_input("Lieu du poste", placeholder="Ville, région ou pays")
        work_mode = b.selectbox(
            "Mode de travail",
            ["À préciser", "Sur site", "Hybride", "Télétravail"],
        )
        sk = st.text_input("Compétences (séparées par des virgules)", "python, sql, machine learning")
        a, b = st.columns(2)
        education = a.text_input("Formation / diplôme requis", placeholder="Ex. Bac+5 en informatique")
        languages = b.text_input("Langues requises", placeholder="Ex. français, anglais")
        include_salary = st.checkbox("Indiquer une fourchette salariale annuelle")
        salary_min = salary_max = None
        if include_salary:
            a, b = st.columns(2)
            salary_min = a.number_input("Salaire minimum (€)", min_value=0, value=40000, step=1000)
            salary_max = b.number_input("Salaire maximum (€)", min_value=0, value=60000, step=1000)
        generate = st.form_submit_button("Générer la fiche en PDF")

    if generate:
        st.session_state.pop("job_posting_pdf", None)
        if not title.strip():
            st.warning("Veuillez saisir l'intitulé du poste.")
        elif include_salary and salary_min > salary_max:
            st.warning("Le salaire minimum ne peut pas dépasser le salaire maximum.")
        else:
            params = {
                "title": title.strip(),
                "domain": dom.strip(),
                "skills": [skill.strip() for skill in sk.split(",") if skill.strip()],
                "level": level,
                "company": company.strip(),
                "contract_type": "" if contract_type == "À préciser" else contract_type,
                "location": location.strip(),
                "work_mode": "" if work_mode == "À préciser" else work_mode,
                "education": education.strip(),
                "languages": languages,
                "salary_min": salary_min,
                "salary_max": salary_max,
            }
            with st.spinner("Génération de la fiche de poste..."):
                r = S.job_posting(**params)
            st.session_state["job_posting_pdf"] = create_job_posting_pdf(
                params["title"], r["text"]
            )
            st.session_state["job_posting_engine"] = r["engine"]
            st.session_state["job_posting_error"] = r.get("error")
            safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", params["title"]).strip("_")
            st.session_state["job_posting_filename"] = (
                f"fiche_de_poste_{safe_title or 'poste'}.pdf"
            )

    if st.session_state.get("job_posting_pdf"):
        st.caption(f"Moteur : {st.session_state['job_posting_engine']}")
        if st.session_state.get("job_posting_error"):
            st.error(st.session_state["job_posting_error"])
        st.download_button(
            "Télécharger la fiche de poste (PDF)",
            data=st.session_state["job_posting_pdf"],
            file_name=st.session_state["job_posting_filename"],
            mime="application/pdf",
        )
with t5: st.json(S.metrics())
