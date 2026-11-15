"""
Interface de démo (Streamlit) — RAG Multimodal générique.
Lancement : streamlit run app.py

Organisation :
- Écran d'accueil : connexion / création de compte (avec logo).
- Barre latérale : gestion des documents (upload, cache, document actif).
- Zone principale : recherche et réponse, centrée sur la question posée.
"""
import os
import json
import hashlib
import copy
import streamlit as st
import streamlit.components.v1 as components

from src.ingest import load_config, ingest_pdfs
from src.chunk_compare import main as run_chunk_comparison
from src.ocr_utils import run_ocr_on_images
from src.embed_index import build_text_index, build_image_index
from src.retrieve import Retriever
from src.generate import generate_answer
from src.auth import register_user, verify_login

st.set_page_config(page_title="Recherche documentaire — RAG Multimodal", layout="wide")

CACHE_ROOT = "cache/documents"
ACTIVE_POINTER = "active_document.json"
LOGO_PATH = "assets/logo.png"  # place ton fichier logo ici, sous ce nom exact

VOICE_LANGUAGES = {
    "Français": "fr-FR",
    "English": "en-US",
    "العربية": "ar-SA",
}


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
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=200)

        st.markdown(
            '<div class="hero-title" style="text-align:center;">Recherche documentaire intelligente</div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["Connexion", "Créer un compte"])

        with tab_login:
            login_user = st.text_input("Nom d'utilisateur", key="login_user")
            login_pass = st.text_input("Mot de passe", type="password", key="login_pass")
            if st.button("Se connecter", type="primary", use_container_width=True):
                ok, full_name = verify_login(login_user, login_pass)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.user_full_name = full_name
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect.")

        with tab_signup:
            su_name = st.text_input("Nom complet", key="su_name")
            su_user = st.text_input("Nom d'utilisateur", key="su_user")
            su_pass = st.text_input("Mot de passe", type="password", key="su_pass")
            su_pass2 = st.text_input("Confirmer le mot de passe", type="password", key="su_pass2")
            if st.button("Créer mon compte", type="primary", use_container_width=True):
                if su_pass != su_pass2:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    ok, msg = register_user(su_user, su_pass, su_name)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    st.stop()


# ---------------------------------------------------------------------------
# Fonctions cache
# ---------------------------------------------------------------------------
def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()[:16]


def build_runtime_cfg(base_cfg, doc_hash):
    cfg = copy.deepcopy(base_cfg)
    doc_dir = os.path.join(CACHE_ROOT, doc_hash)
    cfg["paths"]["raw_pdf_dir"] = os.path.join(doc_dir, "raw")
    cfg["paths"]["processed_dir"] = os.path.join(doc_dir, "processed")
    cfg["paths"]["images_dir"] = os.path.join(doc_dir, "processed", "images")
    cfg["paths"]["index_dir"] = os.path.join(doc_dir, "index")
    return cfg


def is_already_processed(cfg):
    return os.path.exists(os.path.join(cfg["paths"]["index_dir"], "text.index"))


def save_active_pointer(doc_hash, filename):
    with open(ACTIVE_POINTER, "w", encoding="utf-8") as f:
        json.dump({"hash": doc_hash, "filename": filename}, f, ensure_ascii=False, indent=2)


def load_active_pointer():
    if os.path.exists(ACTIVE_POINTER):
        with open(ACTIVE_POINTER, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def list_cached_documents():
    results = []
    if not os.path.exists(CACHE_ROOT):
        return results
    for h in os.listdir(CACHE_ROOT):
        raw_dir = os.path.join(CACHE_ROOT, h, "raw")
        if os.path.isdir(raw_dir):
            files = os.listdir(raw_dir)
            if files:
                results.append((h, files[0]))
    return results


@st.cache_resource
def get_retriever(doc_hash):
    base_cfg = load_config()
    cfg = build_runtime_cfg(base_cfg, doc_hash)
    return cfg, Retriever(cfg)


# ---------------------------------------------------------------------------
# Barre latérale — gestion des documents
# ---------------------------------------------------------------------------
base_cfg = load_config()
active = load_active_pointer()

with st.sidebar:
    st.caption(f"Connecté(e) : **{st.session_state.user_full_name}**")
    if st.button("Se déconnecter"):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()

    st.markdown("### Documents")
    st.caption("Chaque document est traité une seule fois puis mis en cache.")

    cached_docs = list_cached_documents()
    if cached_docs:
        for h, fname in cached_docs:
            is_active = active is not None and active.get("hash") == h
            label = f"● {fname}" if is_active else fname
            if st.button(label, key=f"use_{h}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                save_active_pointer(h, fname)
                get_retriever.clear()
                st.rerun()

    st.divider()
    st.markdown("**Charger un nouveau document**")
    uploaded_file = st.file_uploader("Fichier PDF", type=["pdf"], label_visibility="collapsed")
    process_clicked = st.button("Traiter le document", type="primary",
                                 disabled=uploaded_file is None, use_container_width=True)

    if process_clicked and uploaded_file is not None:
        file_bytes = uploaded_file.getbuffer()
        doc_hash = file_hash(bytes(file_bytes))
        cfg = build_runtime_cfg(base_cfg, doc_hash)

        if is_already_processed(cfg):
            st.success("Document déjà traité — chargement instantané.")
            save_active_pointer(doc_hash, uploaded_file.name)
            get_retriever.clear()
            st.rerun()
        else:
            raw_dir = cfg["paths"]["raw_pdf_dir"]
            os.makedirs(raw_dir, exist_ok=True)
            save_path = os.path.join(raw_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(file_bytes)

            progress = st.progress(0, text="Extraction du texte et des images...")
            pages = ingest_pdfs(cfg)
            nb_images = sum(len(p["image_paths"]) for p in pages)
            progress.progress(25, text=f"{len(pages)} pages, {nb_images} images extraites")

            progress.progress(30, text="Comparaison des méthodes de chunking...")
            report = run_chunk_comparison(cfg)
            progress.progress(55, text=f"Méthode retenue : {report['best_method']}")

            progress.progress(60, text="Lecture OCR des images...")
            run_ocr_on_images(cfg)
            progress.progress(75, text="OCR terminé")

            progress.progress(80, text="Indexation texte et image...")
            build_text_index(cfg)
            build_image_index(cfg)
            progress.progress(100, text="Terminé")

            st.success(f"Document indexé — {len(pages)} pages, {nb_images} images.")
            save_active_pointer(doc_hash, uploaded_file.name)
            get_retriever.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Zone principale — recherche
# ---------------------------------------------------------------------------
active = load_active_pointer()

if active is None:
    st.markdown('<div class="hero-eyebrow">Recherche documentaire intelligente</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Ajoutez un document pour commencer</div>', unsafe_allow_html=True)
    st.write("Utilisez le panneau à gauche pour charger un PDF — texte et images seront indexés automatiquement.")
    st.stop()

cfg, retriever = get_retriever(active["hash"])

st.markdown('<div class="hero-eyebrow">Recherche documentaire intelligente</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-title">Posez une question sur {active["filename"]}</div>', unsafe_allow_html=True)

voice_lang_label = st.selectbox("Langue de la voix", list(VOICE_LANGUAGES.keys()),
                                 label_visibility="visible", key="voice_lang")
voice_lang_code = VOICE_LANGUAGES[voice_lang_label]

voice_query_value = ""
if "voice_query" in st.query_params:
    voice_query_value = st.query_params["voice_query"]
    st.query_params.clear()

components.html(f"""
<div style="margin-bottom: 10px;">
  <button id="mic-btn" style="
      background:#1FA98D; color:white; border:none; border-radius:6px;
      padding:0.5rem 1rem; font-size:0.95rem; cursor:pointer;">
    🎤 Poser la question à l'oral
  </button>
  <span id="mic-status" style="margin-left:10px; color:#90A0AC; font-size:0.85rem;"></span>
</div>
<script>
const btn = document.getElementById("mic-btn");
const status = document.getElementById("mic-status");
btn.addEventListener("click", () => {{
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {{
        status.textContent = "Reconnaissance vocale non supportée sur ce navigateur (utilisez Chrome ou Edge).";
        return;
    }}
    const recognition = new SpeechRecognition();
    recognition.lang = "{voice_lang_code}";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    status.textContent = "Écoute en cours... parlez maintenant.";
    recognition.start();

    recognition.onresult = (event) => {{
        const transcript = event.results[0][0].transcript;
        status.textContent = "Reconnu : " + transcript;
        const url = new URL(window.top.location.href);
        url.searchParams.set("voice_query", transcript);
        window.top.location.href = url.toString();
    }};
    recognition.onerror = (event) => {{
        status.textContent = "Erreur de reconnaissance : " + event.error;
    }};
}});
</script>
""", height=60)

col_input, col_button = st.columns([5, 1])
with col_input:
    query = st.text_input("Question", value=voice_query_value,
                           placeholder="Ex : quelles sont les caractéristiques décrites page 12 ?",
                           label_visibility="collapsed")
with col_button:
    search_clicked = st.button("Rechercher", type="primary", use_container_width=True)

if search_clicked and query.strip():
    with st.spinner("Recherche dans le document..."):
        text_results, image_results = retriever.retrieve(query)

    with st.spinner("Rédaction de la réponse..."):
        try:
            answer = generate_answer(query, text_results, cfg)
        except Exception as e:
            answer = f"Réponse indisponible : {e}\n\nLes extraits pertinents trouvés sont affichés ci-dessous."

    answer_says_dont_know = "je ne sais pas" in answer.strip().lower()

    if answer_says_dont_know:
        st.markdown(
            f'<div class="notice-empty">Aucune information trouvée dans le document pour cette question.</div>',
            unsafe_allow_html=True,
        )
    else:
        rtl_style = "direction: rtl; text-align: right;" if voice_lang_code.startswith("ar") else ""
        st.markdown(
            f'<div class="answer-card" style="{rtl_style}"><div class="answer-label">Réponse</div>{answer}</div>',
            unsafe_allow_html=True,
        )

        answer_json = json.dumps(answer)
        components.html(f"""
        <button id="tts-btn" style="
            background:#16202B; color:#D9A62E; border:1px solid #D9A62E; border-radius:6px;
            padding:0.4rem 0.9rem; font-size:0.9rem; cursor:pointer;">
          🔊 Écouter la réponse
        </button>
        <script>
        document.getElementById("tts-btn").addEventListener("click", () => {{
            const text = {answer_json};
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = "{voice_lang_code}";
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
        }});
        </script>
        """, height=50)

        if image_results:
            st.markdown("**Images pertinentes**")
            cols = st.columns(min(len(image_results), 4))
            for col, img in zip(cols, image_results):
                with col:
                    st.image(img["path"], use_container_width=True)
                    st.markdown(
                        f'<span class="tag-page">{img["doc"]} · page {img["page"]}</span>',
                        unsafe_allow_html=True,
                    )

    with st.expander(f"Sources utilisées ({len(text_results)})"):
        for r in text_results:
            st.markdown(
                f"""<div class="source-block">
                    <div class="source-meta">{r['doc']} · page {r['page']} · pertinence {r['score']:.2f}</div>
                    {r['content']}
                </div>""",
                unsafe_allow_html=True,
            )
