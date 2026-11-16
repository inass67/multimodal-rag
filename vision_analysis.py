"""
Analyse vision GÉNÉRIQUE des pages d'un document — détecte toute information
encodée visuellement (couleur, surlignage, position, graphique) qui ne
serait pas capturée par l'extraction texte classique.

Générique par conception : ce module ne connaît RIEN du domaine du document
(pas de mention de papillons, de calendrier, de factures...). Le modèle
vision décide lui-même, page par page, si un contenu visuel porteur de sens
existe, et le décrit en texte libre. Cette description devient ensuite un
chunk normal, indexé et cherchable comme n'importe quel autre texte —
fonctionne donc sur n'importe quel type de document.

⚠️ COÛT ET QUOTA — À LIRE AVANT DE LANCER :
Ce script appelle l'API vision UNE FOIS PAR PAGE contenant au moins une
image. Sur un document de plusieurs dizaines de pages, cela peut consommer
une grande partie (voire la totalité) du quota gratuit quotidien de l'API.
Une limite de sécurité (MAX_PAGES) est fixée ci-dessous par défaut à 5 pages
pour tester sans risque -- augmente-la seulement si ton quota le permet.

Ce script fonctionne sur le DOCUMENT ACTUELLEMENT ACTIF dans l'application
(celui pointé par active_document.json), quel qu'il soit -- aucune
information sur son contenu n'est supposée à l'avance.

Utilisation :
    $env:GOOGLE_API_KEY="ta_clé"
    python vision_analysis.py
"""
import os
import json
import copy
import pypdfium2 as pdfium

from src.ingest import load_config
from src.embed_index import build_text_index

MAX_PAGES = 40  # limite de sécurité pour le quota -- augmente si besoin
CACHE_ROOT = "cache/documents"
ACTIVE_POINTER = "active_document.json"

GENERIC_VISION_PROMPT = """Regarde attentivement cette page de document.

Certaines informations sont parfois encodées UNIQUEMENT de façon visuelle
(couleur, surlignage, position, forme) et ne sont pas capturées par une
extraction de texte classique. Exemples : une frise chronologique où des
sections sont surlignées, un graphique à barres, une carte avec des zones
colorées, un diagramme avec des éléments mis en évidence, un tableau de
statuts avec des couleurs (vert/rouge), etc.

Si cette page contient ce type d'information visuelle porteuse de sens qui
NE serait PAS évidente à partir du texte seul, décris-la clairement en une
ou deux phrases en français, de façon factuelle et directement exploitable
pour répondre à une question (ex : "Le graphique en haut de page indique
une période d'activité de avril à octobre.").

Si cette page ne contient AUCUNE information visuelle de ce type (texte
normal, photo illustrative sans code couleur porteur de sens, etc.),
réponds uniquement : NONE
"""


def build_runtime_cfg(doc_hash):
    base_cfg = load_config()
    cfg = copy.deepcopy(base_cfg)
    doc_dir = os.path.join(CACHE_ROOT, doc_hash)
    cfg["paths"]["raw_pdf_dir"] = os.path.join(doc_dir, "raw")
    cfg["paths"]["processed_dir"] = os.path.join(doc_dir, "processed")
    cfg["paths"]["images_dir"] = os.path.join(doc_dir, "processed", "images")
    cfg["paths"]["index_dir"] = os.path.join(doc_dir, "index")
    return cfg


def render_page_image(pdf_path, page_index_0based, scale=2.5):
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_index_0based]
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def make_vision_model(model_name, api_key):
    """Retourne une fonction (prompt, image) -> texte, avec le SDK google-genai
    (ou l'ancien google-generativeai s'il est installé à la place)."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return lambda prompt, img: client.models.generate_content(
            model=model_name, contents=[prompt, img]).text
    except ImportError:
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        m = genai_old.GenerativeModel(model_name)
        return lambda prompt, img: m.generate_content([prompt, img]).text


def analyze_pages_with_vision(cfg, max_pages=MAX_PAGES):
    api_key = os.environ.get(cfg["llm"]["api_key_env"])
    if not api_key:
        raise ValueError("Clé API manquante pour l'analyse vision.")
    model = make_vision_model(cfg["llm"]["model"], api_key)

    processed_dir = cfg["paths"]["processed_dir"]
    raw_dir = cfg["paths"]["raw_pdf_dir"]

    with open(os.path.join(processed_dir, "pages_raw.json"), "r", encoding="utf-8") as f:
        pages = json.load(f)
    with open(os.path.join(processed_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    next_id = max([c["id"] for c in chunks], default=-1) + 1
    added = 0

    pages_with_images = [p for p in pages if p["image_paths"]][:max_pages]
    print(f"[vision] {len(pages_with_images)} page(s) à analyser (limite : {max_pages})")

    for p in pages_with_images:
        pdf_path = os.path.join(raw_dir, p["doc"])
        try:
            img = render_page_image(pdf_path, p["page"] - 1)
            description = (model(GENERIC_VISION_PROMPT, img) or "").strip()
        except Exception as e:
            print(f"[vision] Erreur page {p['page']} : {e}")
            continue

        if description and description.upper() != "NONE":
            chunks.append({
                "id": next_id,
                "type": "vision_description",
                "content": description,
                "doc": p["doc"],
                "page": p["page"],
                "image_paths": p["image_paths"],
            })
            next_id += 1
            added += 1
            print(f"[vision] Page {p['page']} -> {description}")
        else:
            print(f"[vision] Page {p['page']} -> rien de visuellement significatif détecté")

    with open(os.path.join(processed_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[vision] {added} description(s) visuelle(s) ajoutée(s) au document.")
    return added


if __name__ == "__main__":
    from src.corpus import load_selection, list_documents

    selected = load_selection()
    if not selected:
        print("Aucun document sélectionné. Ouvre l'application et coche un document d'abord.")
        raise SystemExit(1)

    names = {d["hash"]: d["name"] for d in list_documents()}
    for doc_hash in selected:
        print(f"Document : {names.get(doc_hash, doc_hash)}")
        cfg = build_runtime_cfg(doc_hash)
        nb_added = analyze_pages_with_vision(cfg)
        if nb_added > 0:
            print("\nReconstruction de l'index texte pour inclure les nouvelles descriptions...")
            build_text_index(cfg)
            print("Terminé. Relance l'application (ou pose une nouvelle question) pour en profiter.")
        else:
            print("Aucune description ajoutée, index inchangé.")
