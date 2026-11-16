"""
Agent de recherche documentaire — interface Streamlit.
Lancement : streamlit run app.py

Organisation :
- Écran d'accueil : connexion / création de compte.
- Barre latérale : CORPUS de documents (PDF, Word, texte, HTML, articles web),
  sélection des documents interrogés, ajout et suppression.
- Onglet « Poser une question » : l'agent choisit l'outil adapté, répond avec
  des références numérotées [n], résume les sources utilisées et montre ses étapes.
- Onglet « Résumer les sources » : résumé d'un document ou synthèse du corpus.
- Onglet « Évaluation » : qualité du découpage, comparaison des embeddings,
  métriques de recherche et pertinence des citations.
"""
import os
import json
import html
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ingest import load_config
from src.loaders import UPLOAD_TYPES
from src.corpus import (list_documents, load_selection, save_selection, add_file, add_url,
                        is_processed, process_document, delete_document,
                        load_corpus_retriever, summary_paths, chunking_reports)
from src.agent import ResearchAgent, match_document
from src.summarize import get_or_create_summary, summarize_corpus
from src.llm import LLMError, stream_llm
from src.generate import (ANSWER_TEMPLATE, NO_ANSWER, format_context, split_answer_sections,
                          extract_citations, remove_citations, is_no_answer)
from src.auth import register_user, verify_login

st.set_page_config(page_title="Agent de recherche documentaire", layout="wide")

LOGO_PATH = "assets/logo.png"  # ancien logo (plus utilisé)

# Logo : petit robot de recherche documentaire (SVG dessiné à la main, animé en CSS)
ROBOT_SVG = """
<svg class="robot" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Robot de recherche documentaire">
  <circle cx="60" cy="64" r="54" fill="#1FA98D" opacity="0.10"/>
  <g class="robot-float">
    <line x1="60" y1="19" x2="60" y2="31" stroke="#8FB3C4" stroke-width="3" stroke-linecap="round"/>
    <circle class="robot-antenna" cx="60" cy="15" r="5.5" fill="#D9A62E"/>
    <rect x="20" y="47" width="9" height="18" rx="3.5" fill="#1FA98D"/>
    <rect x="91" y="47" width="9" height="18" rx="3.5" fill="#1FA98D"/>
    <rect x="28" y="31" width="64" height="50" rx="16" fill="#1E2B37" stroke="#1FA98D" stroke-width="2.5"/>
    <rect x="36" y="40" width="48" height="29" rx="10" fill="#0C1218"/>
    <circle class="robot-eye" cx="49" cy="53" r="5" fill="#3DDBB8"/>
    <circle class="robot-eye" cx="71" cy="53" r="5" fill="#3DDBB8"/>
    <path d="M51 62 Q60 67.5 69 62" stroke="#3DDBB8" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    <rect x="40" y="84" width="40" height="24" rx="8" fill="#1E2B37" stroke="#1FA98D" stroke-width="2"/>
    <rect x="51" y="89" width="18" height="14" rx="2.5" fill="#D9A62E"/>
    <line x1="54.5" y1="93.5" x2="65.5" y2="93.5" stroke="#16202B" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="54.5" y1="97.5" x2="62" y2="97.5" stroke="#16202B" stroke-width="1.6" stroke-linecap="round"/>
    <path d="M80 93 L89 90" stroke="#1FA98D" stroke-width="4" stroke-linecap="round"/>
    <circle cx="97" cy="88" r="8.5" fill="#0C1218" fill-opacity="0.6" stroke="#D9A62E" stroke-width="3"/>
    <line x1="103" y1="94.5" x2="110" y2="101.5" stroke="#D9A62E" stroke-width="4" stroke-linecap="round"/>
  </g>
</svg>
"""
ROBOT_SVG = " ".join(ROBOT_SVG.split())  # une seule ligne : évite que Markdown la transforme en bloc de code
VOICE_LANGUAGES = {"Français": "fr-FR", "English": "en-US", "العربية": "ar-SA"}
TYPE_ICONS = {"pdf": "📄", "docx": "📝", "web": "🌐", "texte": "🗒️"}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 600; letter-spacing: -0.01em; }

.hero-eyebrow { color: #90A0AC; font-size: 0.95rem; margin-bottom: 0.2rem; }
.hero-title { font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 600;
              color: #E8EDF1; margin-bottom: 1.4rem; line-height: 1.25; }

.answer-card {
    background: #16202B;
    border-left: 4px solid #1FA98D;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.1rem;
}
.answer-label { font-family: 'Fraunces', serif; font-size: 1.05rem; color: #E8EDF1;
                 margin-bottom: 0.5rem; }

.tag-page {
    display: inline-block;
    background: rgba(217, 166, 46, 0.15);
    color: #D9A62E;
    border: 1px solid rgba(217, 166, 46, 0.4);
    border-radius: 999px;
    padding: 0.05rem 0.6rem;
    font-size: 0.78rem;
    margin-top: 0.35rem;
}

.notice-empty {
    background: #16202B;
    border-left: 4px solid #C9633B;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    color: #E8EDF1;
    margin-bottom: 1.1rem;
}

.source-block {
    background: #16202B;
    border: 1px solid #26333F;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.source-meta { color: #90A0AC; font-size: 0.82rem; margin-bottom: 0.3rem; }

section[data-testid="stSidebar"] { background: #0C1218; border-right: 1px solid #26333F; }

/* --- Logo robot --- */
.brand { display: flex; flex-direction: column; align-items: center; margin: 0.5rem 0 0.4rem; }
.brand .robot { width: 150px; height: 150px; filter: drop-shadow(0 8px 24px rgba(31,169,141,0.25)); }
.brand-sub { color: #90A0AC; font-size: 1rem; text-align: center; margin: -0.6rem 0 1.4rem; }
.brand-badges { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; margin-bottom: 1.4rem; }
.brand-badges span { border: 1px solid #26333F; background: #16202B; color: #C9D4DC; border-radius: 999px;
                     padding: 0.2rem 0.75rem; font-size: 0.8rem; }
.side-brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem; }
.side-brand .robot { width: 46px; height: 46px; }
.side-brand-name { font-family: 'Fraunces', serif; font-size: 1.05rem; color: #E8EDF1; line-height: 1.2; }
.side-brand-tag { color: #90A0AC; font-size: 0.75rem; }
.robot-float { animation: robot-float 3.2s ease-in-out infinite; }
.robot-eye { transform-box: fill-box; transform-origin: center; animation: robot-blink 4.5s infinite; }
.robot-antenna { animation: robot-pulse 1.8s ease-in-out infinite; }
@keyframes robot-float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
@keyframes robot-blink { 0%, 92%, 100% { transform: scaleY(1); } 95% { transform: scaleY(0.1); } }
@keyframes robot-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AUTHENTIFICATION — écran de connexion / création de compte
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_full_name = ""

if not st.session_state.authenticated:
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(f'<div class="brand">{ROBOT_SVG}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-title" style="text-align:center;">Agent de recherche documentaire</div>'
            '<div class="brand-sub">Posez vos questions, obtenez des réponses sourcées.</div>'
            '<div class="brand-badges"><span>PDF · Word · Web</span><span>Citations [1] [2]</span>'
            '<span>Résumés</span><span>RAG + LangChain</span></div>',
            unsafe_allow_html=True,
        )

        # Formulaires (st.form) : les valeurs saisies sont envoyées toutes ensemble
        # au clic, ce qui évite qu'une case soit vue comme vide.
        mode = st.radio("Mode", ["Connexion", "Créer un compte"], horizontal=True,
                        label_visibility="collapsed", key="auth_mode")

        if mode == "Connexion":
            with st.form("login_form"):
                login_user = st.text_input("Nom d'utilisateur")
                login_pass = st.text_input("Mot de passe", type="password")
                submitted = st.form_submit_button("Se connecter", type="primary",
                                                  use_container_width=True)
            if submitted:
                ok, full_name = verify_login(login_user, login_pass)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.user_full_name = full_name
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect.")
        else:
            with st.form("signup_form"):
                su_name = st.text_input("Nom complet")
                su_user = st.text_input("Nom d'utilisateur")
                su_pass = st.text_input("Mot de passe (6 caractères minimum)", type="password")
                su_pass2 = st.text_input("Confirmer le mot de passe", type="password")
                submitted = st.form_submit_button("Créer mon compte", type="primary",
                                                  use_container_width=True)
            if submitted:
                if su_pass != su_pass2:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    ok, msg = register_user(su_user, su_pass, su_name)
                    if ok:
                        st.success(msg + " Choisissez « Connexion » ci-dessus.")
                    else:
                        st.error(msg)

    st.stop()


# ---------------------------------------------------------------------------
# Chargement du corpus (mis en cache tant que la sélection ne change pas)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du corpus et des modèles...")
def get_retriever(hashes, text_model):
    return load_corpus_retriever(load_config(), list(hashes))


def refresh_corpus():
    get_retriever.clear()
    st.session_state.pop("last_output", None)


def run_processing(cfg, label):
    bar = st.progress(0, text=f"{label} : préparation...")
    stats = process_document(cfg, progress=lambda pct, msg: bar.progress(pct, text=f"{label} : {msg}"))
    bar.empty()
    return stats


base_cfg = load_config()
documents = list_documents()
known_hashes = {d["hash"] for d in documents}
selection = [h for h in load_selection() if h in known_hashes]

# ---------------------------------------------------------------------------
# Barre latérale — corpus
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="side-brand">{ROBOT_SVG}<div><div class="side-brand-name">Agent documentaire</div>'
                f'<div class="side-brand-tag">Recherche · Résumés · Citations</div></div></div>',
                unsafe_allow_html=True)
    st.caption(f"Connecté(e) : **{st.session_state.user_full_name}**")
    if st.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()

    st.markdown("### Corpus")
    st.caption("Cochez les documents à interroger ensemble. Chaque document est traité une seule fois.")
    new_selection = []
    for d in documents:
        c1, c2 = st.columns([6, 1])
        with c1:
            icon = TYPE_ICONS.get(d.get("source_type"), "📄")
            if st.checkbox(f"{icon} {d['name']}", value=d["hash"] in selection, key=f"sel_{d['hash']}"):
                new_selection.append(d["hash"])
        with c2:
            if st.button("🗑", key=f"del_{d['hash']}", help="Supprimer ce document"):
                delete_document(d["hash"])
                refresh_corpus()
                st.rerun()
    if new_selection != selection:
        save_selection(new_selection)
        selection = new_selection
        refresh_corpus()
    if not documents:
        st.info("Aucun document pour l'instant.")

    st.divider()
    st.markdown("**Ajouter des fichiers**")
    st.caption("PDF, Word, texte, Markdown ou pages HTML internes.")
    uploaded_files = st.file_uploader("Fichiers", type=UPLOAD_TYPES, accept_multiple_files=True,
                                      label_visibility="collapsed")
    if st.button("Traiter les fichiers", type="primary", disabled=not uploaded_files,
                 use_container_width=True):
        for up in uploaded_files:
            doc_hash, cfg = add_file(base_cfg, up.name, bytes(up.getbuffer()))
            try:
                if not is_processed(cfg):
                    stats = run_processing(cfg, up.name)
                    st.success(f"{up.name} : {stats['pages']} pages/sections, {stats['images']} images.")
                if doc_hash not in selection:
                    selection.append(doc_hash)
            except Exception as e:
                st.error(f"{up.name} : échec du traitement ({e})")
        save_selection(selection)
        refresh_corpus()
        st.rerun()

    st.markdown("**Ajouter un article web**")
    url = st.text_input("URL", placeholder="https://...", label_visibility="collapsed")
    if st.button("Télécharger et indexer", disabled=not url.strip(), use_container_width=True):
        try:
            with st.spinner("Téléchargement de l'article..."):
                doc_hash, cfg, title = add_url(base_cfg, url)
            if not is_processed(cfg):
                run_processing(cfg, title)
            if doc_hash not in selection:
                selection.append(doc_hash)
            save_selection(selection)
            refresh_corpus()
            st.rerun()
        except Exception as e:
            st.error(f"Impossible d'ajouter cet article : {e}")

    st.divider()
    with st.expander("Réglages de l'agent"):
        planner = st.radio("Choix de l'outil", ["LLM (agent)", "Règles (économise le quota)"],
                           index=0 if base_cfg.get("agent", {}).get("planner", "llm") == "llm" else 1)
        top_k = st.slider("Passages retrouvés par question", 3, 10, base_cfg["retrieval"]["top_k_text"])

# ---------------------------------------------------------------------------
# Zone principale
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-eyebrow">Agent de recherche documentaire</div>', unsafe_allow_html=True)

if not selection:
    st.markdown('<div class="hero-title">Ajoutez ou cochez des documents pour commencer</div>',
                unsafe_allow_html=True)
    st.write("Utilisez le panneau de gauche : PDF, documents Word, fichiers texte, pages internes "
             "ou articles web. Ils seront découpés, vectorisés et interrogeables ensemble.")
    st.stop()

try:
    retriever = get_retriever(tuple(sorted(selection)), base_cfg["embeddings"]["text_model"])
except Exception as e:
    st.error(f"Impossible de charger le corpus : {e}")
    st.stop()

run_cfg = json.loads(json.dumps(base_cfg))  # copie modifiable par les réglages
run_cfg["agent"]["planner"] = "llm" if planner.startswith("LLM") else "rules"
run_cfg["retrieval"]["top_k_text"] = top_k
retriever.config = run_cfg
paths_for_summary = summary_paths(base_cfg, selection)

n_docs = len(retriever.doc_names)
title = retriever.doc_names[0] if n_docs == 1 else f"{n_docs} documents"
st.markdown(f'<div class="hero-title">Interrogez votre corpus — {html.escape(title)}</div>',
            unsafe_allow_html=True)

tab_ask, tab_summary, tab_eval = st.tabs(["🔎 Poser une question", "📝 Résumer les sources", "📊 Évaluation"])


def render_references(result, text_results):
    if not result["references"]:
        return
    st.markdown("**Références**")
    for ref in result["references"]:
        link = f" — [lien]({ref['source']})" if str(ref.get("source", "")).startswith("http") else ""
        st.markdown(f"**[{ref['num']}]** {ref['doc']} · {ref['loc']}{link}")
    if result["invalid"]:
        st.caption(f"⚠️ Citations ignorées car elles ne correspondent à aucune source : {result['invalid']}")


def render_output(out, voice_lang_code):
    rtl = voice_lang_code.startswith("ar")
    if out["type"] == "resume":
        st.markdown(f"#### {out['titre']}")
        with st.container(border=True):
            st.markdown(out["texte"])
        for doc, summ in out.get("details", {}).items():
            with st.expander(f"Résumé détaillé — {doc}"):
                st.markdown(summ)
        text_to_read = out["texte"]
    else:
        result, text_results = out["result"], out["text_results"]
        if result["no_answer"]:
            st.markdown('<div class="notice-empty">Aucune information trouvée dans les documents '
                        'pour cette question.</div>', unsafe_allow_html=True)
            st.caption(result["body"])
        else:
            with st.container(border=True):
                st.markdown("**Réponse**")
                body = result["body"]
                st.markdown(f'<div dir="rtl">\n\n{body}\n\n</div>' if rtl else body,
                            unsafe_allow_html=rtl)
            render_references(result, text_results)
            if result["sources_summary"]:
                with st.expander("Résumé des sources utilisées", expanded=True):
                    st.markdown(result["sources_summary"])
            if out.get("image_results"):
                st.markdown("**Images pertinentes**")
                cols = st.columns(min(len(out["image_results"]), 4))
                for col, img in zip(cols, out["image_results"]):
                    with col:
                        st.image(img["path"], use_container_width=True)
                        st.markdown(f'<span class="tag-page">{html.escape(img["doc"])} · '
                                    f'page {img["page"]}</span>', unsafe_allow_html=True)
        with st.expander(f"Tous les passages retrouvés ({len(text_results)})"):
            for i, r in enumerate(text_results, start=1):
                mark = "✅ cité" if i in result["cited"] else "non cité"
                st.markdown(
                    f'<div class="source-block"><div class="source-meta">[{i}] {html.escape(r["doc"])} · '
                    f'{r["loc"]} · pertinence {r["score"]:.2f} · {mark}</div>'
                    f'{html.escape(r["content"])}</div>', unsafe_allow_html=True)
        text_to_read = result["body"]

    with st.expander("Étapes suivies par l'agent"):
        for step in out["trace"]:
            st.markdown(f"- **{step['etape']}** — {step['detail']}")

    components.html(f"""
    <button id="tts-btn" style="background:#16202B; color:#D9A62E; border:1px solid #D9A62E;
        border-radius:6px; padding:0.4rem 0.9rem; font-size:0.9rem; cursor:pointer;">
      🔊 Écouter la réponse
    </button>
    <script>
    document.getElementById("tts-btn").addEventListener("click", () => {{
        const text = {json.dumps(text_to_read)}.replace(/\\[\\d+\\]/g, "").replace(/[#*_]/g, "");
        const u = new SpeechSynthesisUtterance(text);
        u.lang = "{voice_lang_code}";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
    }});
    </script>
    """, height=50)


# ---------------------------------------------------------------------------
# Onglet 1 — Questions (agent)
# ---------------------------------------------------------------------------
with tab_ask:
    voice_lang_label = st.selectbox("Langue de la voix", list(VOICE_LANGUAGES.keys()), key="voice_lang")
    voice_lang_code = VOICE_LANGUAGES[voice_lang_label]

    voice_query_value = ""
    if "voice_query" in st.query_params:
        voice_query_value = st.query_params["voice_query"]
        st.query_params.clear()

    components.html(f"""
    <div style="margin-bottom: 10px;">
      <button id="mic-btn" style="background:#1FA98D; color:white; border:none; border-radius:6px;
          padding:0.5rem 1rem; font-size:0.95rem; cursor:pointer;">🎤 Poser la question à l'oral</button>
      <span id="mic-status" style="margin-left:10px; color:#90A0AC; font-size:0.85rem;"></span>
    </div>
    <script>
    const btn = document.getElementById("mic-btn");
    const status = document.getElementById("mic-status");
    btn.addEventListener("click", () => {{
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {{ status.textContent = "Reconnaissance vocale non supportée (utilisez Chrome ou Edge)."; return; }}
        const rec = new SR();
        rec.lang = "{voice_lang_code}";
        rec.interimResults = false;
        status.textContent = "Écoute en cours... parlez maintenant.";
        rec.start();
        rec.onresult = (e) => {{
            const t = e.results[0][0].transcript;
            status.textContent = "Reconnu : " + t;
            const url = new URL(window.top.location.href);
            url.searchParams.set("voice_query", t);
            window.top.location.href = url.toString();
        }};
        rec.onerror = (e) => {{ status.textContent = "Erreur de reconnaissance : " + e.error; }};
    }});
    </script>
    """, height=60)

    st.caption("Exemples : « Quelles sont les conditions pour ... ? » · « Résume le document X » · "
               "« Compare ce que disent les documents sur ... » · « Quels documents sont disponibles ? »")
    col_input, col_button = st.columns([5, 1])
    with col_input:
        query = st.text_input("Question", value=voice_query_value, label_visibility="collapsed",
                              placeholder="Posez votre question en langage naturel...")
    with col_button:
        search_clicked = st.button("Rechercher", type="primary", use_container_width=True)

    if search_clicked and query.strip():
        q = query.strip()
        t_start = time.time()
        agent = ResearchAgent(retriever, run_cfg, summary_path_fn=paths_for_summary.get)
        agent.trace = []
        decision = agent.plan(q)

        if decision["outil"] == "rechercher":
            # --- Chemin rapide : recherche + réponse affichée EN DIRECT ---
            doc_filter = None
            asked_doc = decision.get("arguments", {}).get("document")
            if asked_doc and decision.get("planificateur") == "LLM":
                matched = match_document(asked_doc, retriever.doc_names)
                doc_filter = [matched] if matched else None
            agent.log("Planification", f"Outil choisi : rechercher ({decision.get('planificateur')})")
            text_results, image_results = retriever.retrieve(q, doc_filter=doc_filter)
            agent.log("Recherche", f"{len(text_results)} passages retrouvés en "
                                   f"{(time.time() - t_start) * 1000:.0f} ms (sémantique + BM25)")
            raw = ""
            if text_results:
                prompt = ANSWER_TEMPLATE.format(no_answer=NO_ANSWER,
                                                context=format_context(text_results), question=q)
                live = st.empty()
                try:
                    with live.container(border=True):
                        st.caption(f"Question : {q} — {len(text_results)} passages trouvés, rédaction en direct...")
                        raw = st.write_stream(stream_llm(prompt, run_cfg))
                except LLMError as e:
                    raw = f"{NO_ANSWER} : réponse indisponible ({e})"
                live.empty()
                if isinstance(raw, list):
                    raw = "".join(str(x) for x in raw)
            else:
                raw = f"{NO_ANSWER} : aucun passage pertinent n'a été trouvé."

            # Vérification des citations (numéros inventés retirés)
            body, sources_summary = split_answer_sections(raw)
            cited, invalid = extract_citations(body, len(text_results))
            if invalid:
                body = remove_citations(body, invalid)
                sources_summary = remove_citations(sources_summary, invalid)
            agent.log("Vérification", f"{len(cited)} source(s) citée(s)"
                      + (f", citations inventées retirées : {invalid}" if invalid else ""))
            agent.log("Durée", f"réponse complète en {time.time() - t_start:.1f} s")
            st.session_state.last_output = {
                "type": "reponse", "outil": "rechercher", "trace": list(agent.trace),
                "text_results": text_results, "image_results": image_results,
                "result": {"answer": raw, "body": body, "sources_summary": sources_summary,
                           "cited": cited, "invalid": invalid,
                           "references": [{"num": n, **text_results[n - 1]} for n in cited],
                           "no_answer": is_no_answer(body)},
            }
            st.session_state.last_query = q
        else:
            # --- Autres outils (résumé, comparaison, liste) : exécutés par l'agent ---
            with st.spinner("L'agent prépare la réponse..."):
                try:
                    tool = agent.tools[decision["outil"]]
                    args = {k: v for k, v in decision.get("arguments", {}).items() if v}
                    args.setdefault("question", q)
                    agent.log("Planification", f"Outil choisi : {tool.name} ({decision.get('planificateur')})")
                    output = tool.func(**args)
                    output["outil"] = tool.name
                    output["trace"] = list(agent.trace)
                    st.session_state.last_output = output
                    st.session_state.last_query = q
                except LLMError as e:
                    st.error(f"Le LLM est indisponible : {e}")

    if "last_output" in st.session_state:
        st.markdown(f"*Question : {st.session_state.get('last_query', '')}* — outil utilisé : "
                    f"`{st.session_state.last_output.get('outil', '')}`")
        render_output(st.session_state.last_output, voice_lang_code)

# ---------------------------------------------------------------------------
# Onglet 2 — Résumés
# ---------------------------------------------------------------------------
with tab_summary:
    st.write("Résumé map-reduce : chaque partie du document est résumée, puis les résumés sont "
             "fusionnés. Les résumés sont gardés en cache.")
    options = ["Tout le corpus (synthèse comparative)"] + retriever.doc_names
    choice = st.selectbox("Que voulez-vous résumer ?", options)
    force = st.checkbox("Régénérer (ignorer le résumé en cache)")
    if st.button("Générer le résumé", type="primary"):
        try:
            with st.spinner("Résumé en cours..."):
                def doc_summary(doc):
                    chunks = [c for c in retriever.text_meta if c["doc"] == doc]
                    return get_or_create_summary(doc, chunks, run_cfg,
                                                 cache_path=paths_for_summary.get(doc), force=force)
                if choice == options[0]:
                    details = {doc: doc_summary(doc) for doc in retriever.doc_names}
                    st.session_state.summary_out = ("Synthèse du corpus",
                                                     summarize_corpus(details, run_cfg), details)
                else:
                    st.session_state.summary_out = (f"Résumé — {choice}", doc_summary(choice), {})
        except LLMError as e:
            st.error(f"Résumé impossible : {e}")
    if "summary_out" in st.session_state:
        t, txt, details = st.session_state.summary_out
        st.markdown(f"#### {t}")
        with st.container(border=True):
            st.markdown(txt)
        for doc, s in details.items():
            with st.expander(f"Résumé — {doc}"):
                st.markdown(s)
        st.download_button("Télécharger le résumé (.md)", txt, file_name="resume.md")

# ---------------------------------------------------------------------------
# Onglet 3 — Évaluation
# ---------------------------------------------------------------------------
with tab_eval:
    from src import evaluate as ev
    from src import compare_embeddings as ce

    st.markdown("#### 1. Qualité du découpage (chunking)")
    st.caption("Calculé automatiquement à l'ajout de chaque document : la meilleure méthode est retenue.")
    for doc, rep in chunking_reports(base_cfg, selection).items():
        with st.expander(f"{doc} — méthode retenue : {rep['best_method']}"):
            st.dataframe(pd.DataFrame(rep["results"]).T, use_container_width=True)

    st.markdown("#### 2. Jeu de test")
    n_q = st.number_input("Nombre de questions à générer", 5, 50, 15, step=5)
    if os.path.exists(ev.EVAL_SET_PATH):
        items = ev.load_eval_set(doc_names=set(retriever.doc_names))
        st.caption(f"Jeu de test actuel : {len(items)} questions sur ce corpus ({ev.EVAL_SET_PATH}).")
        with st.expander("Voir les questions"):
            st.dataframe(pd.DataFrame(items)[["question", "doc", "page"]] if items else pd.DataFrame(),
                         use_container_width=True)
    if st.button("Générer le jeu de test"):
        try:
            with st.spinner("Le LLM rédige des questions à partir de passages du corpus..."):
                ev.generate_eval_set(retriever, run_cfg, n=int(n_q))
            st.rerun()
        except Exception as e:
            st.error(f"Échec : {e}")

    st.markdown("#### 3. Choix du modèle d'embedding")
    if st.button("Comparer les modèles d'embedding"):
        try:
            with st.spinner("Encodage du corpus avec chaque modèle (quelques minutes)..."):
                ce.run(retriever, run_cfg)
        except Exception as e:
            st.error(f"Échec : {e}")
    if os.path.exists(ce.REPORT_MD_PATH):
        with open(ce.REPORT_MD_PATH, encoding="utf-8") as f:
            st.markdown(f.read())

    st.markdown("#### 4. Recherche et pertinence des citations")
    max_ans = st.slider("Questions utilisées pour évaluer les citations (1 appel LLM chacune)", 3, 30,
                        run_cfg["evaluation"]["max_answer_questions"])
    judge = st.checkbox("Ajouter le juge LLM (plus précis, plus d'appels)",
                        value=run_cfg["evaluation"].get("llm_judge", False))
    if st.button("Lancer l'évaluation", type="primary"):
        try:
            with st.spinner("Évaluation en cours..."):
                ev.run_full_evaluation(retriever, run_cfg, max_answer_questions=max_ans,
                                       use_llm_judge=judge)
        except Exception as e:
            st.error(f"Échec : {e}")
    if os.path.exists(ev.REPORT_MD_PATH):
        with open(ev.REPORT_MD_PATH, encoding="utf-8") as f:
            st.markdown(f.read())
        with open(ev.REPORT_PATH, encoding="utf-8") as f:
            detail = json.load(f).get("detail_citations", [])
        if detail:
            with st.expander("Détail par question"):
                st.dataframe(pd.DataFrame(detail), use_container_width=True)