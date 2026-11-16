"""
Gestion du CORPUS de documents (partagée par l'application et les scripts).

Chaque document (PDF, Word, texte, page HTML ou article web) est traité une
seule fois puis mis en cache dans `cache/documents/<empreinte>/` :
    raw/        fichier d'origine (+ _source.json pour un article web)
    processed/  pages_raw.json, chunks.json, images, résumé...
    index/      index FAISS texte et image + index_info.json

L'utilisateur choisit ensuite les documents qui forment le corpus interrogé
(un ou plusieurs). La sélection est enregistrée dans active_document.json.
"""
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime

from src.loaders import fetch_url, safe_filename, source_type_from_ext
from src.ingest import ingest_sources, SOURCE_META_FILE

CACHE_ROOT = "cache/documents"
SELECTION_FILE = "active_document.json"
DOC_INFO_FILE = "doc_info.json"


def content_hash(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()[:16]


def build_runtime_cfg(base_cfg, doc_hash):
    cfg = copy.deepcopy(base_cfg)
    doc_dir = os.path.join(CACHE_ROOT, doc_hash)
    cfg["paths"]["raw_pdf_dir"] = os.path.join(doc_dir, "raw")
    cfg["paths"]["processed_dir"] = os.path.join(doc_dir, "processed")
    cfg["paths"]["images_dir"] = os.path.join(doc_dir, "processed", "images")
    cfg["paths"]["index_dir"] = os.path.join(doc_dir, "index")
    return cfg


def is_processed(cfg):
    return os.path.exists(os.path.join(cfg["paths"]["index_dir"], "text.index"))


# ---------------------------------------------------------------------------
# Liste des documents en cache
# ---------------------------------------------------------------------------
def _read_doc_info(doc_hash):
    doc_dir = os.path.join(CACHE_ROOT, doc_hash)
    p = os.path.join(doc_dir, DOC_INFO_FILE)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    raw_dir = os.path.join(doc_dir, "raw")  # anciens caches sans doc_info.json
    files = [f for f in os.listdir(raw_dir) if f != SOURCE_META_FILE] if os.path.isdir(raw_dir) else []
    if not files:
        return None
    ext = os.path.splitext(files[0])[1]
    return {"name": files[0], "filename": files[0], "source_type": source_type_from_ext(ext), "url": ""}


def list_documents():
    """Retourne la liste des documents traités : [{hash, name, source_type, url}]."""
    docs = []
    if not os.path.isdir(CACHE_ROOT):
        return docs
    for h in sorted(os.listdir(CACHE_ROOT)):
        info = _read_doc_info(h)
        if info and os.path.exists(os.path.join(CACHE_ROOT, h, "index", "text.index")):
            docs.append({"hash": h, **info})
    return docs


def delete_document(doc_hash):
    path = os.path.join(CACHE_ROOT, doc_hash)
    if os.path.isdir(path) and os.path.abspath(path).startswith(os.path.abspath(CACHE_ROOT)):
        shutil.rmtree(path)
    sel = [h for h in load_selection() if h != doc_hash]
    save_selection(sel)


# ---------------------------------------------------------------------------
# Sélection du corpus interrogé
# ---------------------------------------------------------------------------
def load_selection():
    if not os.path.exists(SELECTION_FILE):
        return []
    with open(SELECTION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "selected" in data:
        return list(data["selected"])
    if "hash" in data:  # ancien format : un seul document actif
        return [data["hash"]]
    return []


def save_selection(hashes):
    with open(SELECTION_FILE, "w", encoding="utf-8") as f:
        json.dump({"selected": list(hashes)}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Ajout d'un document au corpus
# ---------------------------------------------------------------------------
def add_file(base_cfg, filename, file_bytes):
    """Enregistre un fichier téléversé dans le cache. Retourne (hash, cfg)."""
    doc_hash = content_hash(file_bytes)
    cfg = build_runtime_cfg(base_cfg, doc_hash)
    raw_dir = cfg["paths"]["raw_pdf_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    name = safe_filename(os.path.splitext(filename)[0]) + os.path.splitext(filename)[1].lower()
    with open(os.path.join(raw_dir, name), "wb") as f:
        f.write(file_bytes)
    _write_doc_info(doc_hash, {
        "name": filename, "filename": name,
        "source_type": source_type_from_ext(os.path.splitext(filename)[1]),
        "url": "", "added_at": datetime.now().isoformat(timespec="seconds"),
    })
    return doc_hash, cfg


def add_url(base_cfg, url):
    """Télécharge un article web et l'enregistre dans le cache. Retourne (hash, cfg, titre)."""
    url = url.strip()
    title, html, text = fetch_url(url)
    if len(text.split()) < 30:
        raise ValueError("Page trop courte ou illisible (contenu dynamique, accès protégé...).")
    doc_hash = content_hash(url)
    cfg = build_runtime_cfg(base_cfg, doc_hash)
    raw_dir = cfg["paths"]["raw_pdf_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    fname = safe_filename(title)
    if html:
        fname += ".html"
        content = html
    else:  # texte déjà extrait par le chargeur LangChain
        fname += ".txt"
        content = text
    with open(os.path.join(raw_dir, fname), "w", encoding="utf-8") as f:
        f.write(content)
    with open(os.path.join(raw_dir, SOURCE_META_FILE), "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title, "source_type": "web"}, f, ensure_ascii=False, indent=2)
    _write_doc_info(doc_hash, {
        "name": title, "filename": fname, "source_type": "web", "url": url,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    })
    return doc_hash, cfg, title


def _write_doc_info(doc_hash, info):
    doc_dir = os.path.join(CACHE_ROOT, doc_hash)
    os.makedirs(doc_dir, exist_ok=True)
    with open(os.path.join(doc_dir, DOC_INFO_FILE), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def process_document(cfg, progress=None):
    """Pipeline complet : ingestion -> choix du découpage -> OCR -> indexation.
    `progress(pourcentage, message)` est appelé à chaque étape (optionnel)."""
    from src.chunk_compare import main as run_chunk_comparison
    from src.ocr_utils import run_ocr_on_images
    from src.embed_index import build_text_index, build_image_index

    def report(pct, msg):
        if progress:
            progress(pct, msg)
        print(f"[pipeline] {pct}% {msg}")

    report(5, "Extraction du texte et des images...")
    pages = ingest_sources(cfg)
    if not any(p["text"].strip() for p in pages):
        raise ValueError("Aucun texte extrait de ce document (fichier vide ou scanné sans OCR).")
    nb_images = sum(len(p["image_paths"]) for p in pages)
    report(25, f"{len(pages)} pages/sections, {nb_images} images extraites")

    report(30, "Comparaison des méthodes de découpage (chunking)...")
    chunk_report = run_chunk_comparison(cfg)
    report(55, f"Découpage retenu : {chunk_report['best_method']}")

    report(60, "Lecture OCR des images...")
    run_ocr_on_images(cfg)

    report(75, "Calcul des embeddings et indexation (base vectorielle FAISS)...")
    build_text_index(cfg)
    build_image_index(cfg)
    report(100, "Terminé")
    return {"pages": len(pages), "images": nb_images, "chunking": chunk_report["best_method"]}


def ensure_current_embedding(cfg):
    """Réindexe le texte si l'index a été construit avec un autre modèle que
    celui de config.yaml (ex. après le passage à un modèle multilingue).
    Retourne True si une réindexation a eu lieu."""
    from src.embed_index import read_index_info, build_text_index
    info = read_index_info(cfg["paths"]["index_dir"])
    if info.get("text_model") != cfg["embeddings"]["text_model"]:
        build_text_index(cfg)
        return True
    return False


def load_corpus_retriever(base_cfg, hashes):
    """Construit un Retriever unique sur tous les documents sélectionnés."""
    from src.retrieve import Retriever
    index_dirs = []
    for h in hashes:
        cfg = build_runtime_cfg(base_cfg, h)
        if is_processed(cfg):
            ensure_current_embedding(cfg)
            index_dirs.append(cfg["paths"]["index_dir"])
    if not index_dirs:
        raise ValueError("Aucun document traité dans la sélection.")
    return Retriever(base_cfg, index_dirs=index_dirs)


def chunks_by_doc(retriever):
    """Regroupe les passages du corpus par document (pour les résumés)."""
    groups = {}
    for c in retriever.text_meta:
        groups.setdefault(c["doc"], []).append(c)
    return groups


def summary_paths(base_cfg, hashes):
    """{nom du document: chemin du résumé en cache} pour les documents sélectionnés."""
    paths = {}
    for h in hashes:
        cfg = build_runtime_cfg(base_cfg, h)
        try:
            with open(os.path.join(cfg["paths"]["index_dir"], "text_meta.json"), "r", encoding="utf-8") as f:
                names = {c["doc"] for c in json.load(f)}
        except Exception:
            continue
        for n in names:
            paths[n] = os.path.join(cfg["paths"]["processed_dir"], "summary.md")
    return paths


def chunking_reports(base_cfg, hashes):
    """Rapports de comparaison du découpage, par document."""
    reports = {}
    for d in list_documents():
        if d["hash"] not in hashes:
            continue
        p = os.path.join(build_runtime_cfg(base_cfg, d["hash"])["paths"]["processed_dir"],
                         "chunking_comparison_report.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                reports[d["name"]] = json.load(f)
    return reports


def summary_path_for_doc(base_cfg, doc_name):
    """Chemin du résumé en cache d'un document, retrouvé à partir de son nom."""
    for d in list_documents():
        cfg = build_runtime_cfg(base_cfg, d["hash"])
        meta_path = os.path.join(cfg["paths"]["index_dir"], "text_meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                first = json.load(f)[:1]
        except Exception:
            continue
        if first and first[0]["doc"] == doc_name:
            return os.path.join(cfg["paths"]["processed_dir"], "summary.md")
    return None
