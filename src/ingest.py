"""
Étape 1 - Ingestion de sources hétérogènes.

Pour chaque fichier déposé dans le dossier `raw` d'un document :
- PDF  : texte de chaque page (pdfplumber) + images embarquées (pypdfium2),
         en filtrant les petites icônes et en dédoublonnant les images répétées ;
- TXT / MD / DOCX / HTML (et articles web téléchargés) : texte découpé en
         sections (voir loaders.py), chaque section jouant le rôle d'une page.

Chaque entrée produite garde sa provenance : nom du document, numéro de
page/section, type de source et URL d'origine. C'est ce qui permet ensuite
de répondre avec des références précises.
Aucune connaissance du domaine n'est codée en dur : le pipeline est générique.
"""
import os
import json
import hashlib
import yaml

from src.loaders import SUPPORTED_EXT, source_type_from_ext, load_non_pdf

SOURCE_META_FILE = "_source.json"  # métadonnées d'origine (ex. URL d'un article)


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ingest_pdf(doc_path, fname, images_dir, min_image_size):
    import pdfplumber
    import pypdfium2 as pdfium

    base_name = os.path.splitext(fname)[0]
    with pdfplumber.open(doc_path) as pdf:
        texts_by_page = [(page.extract_text() or "") for page in pdf.pages]

    pdf_img = pdfium.PdfDocument(doc_path)
    seen_hashes = set()
    images_by_page = [[] for _ in range(len(pdf_img))]
    for page_idx in range(len(pdf_img)):
        page = pdf_img[page_idx]
        local_count = 0
        for obj in page.get_objects():
            if obj.type != pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                continue
            try:
                pil_img = obj.get_bitmap().to_pil()
            except Exception:
                continue
            w, h = pil_img.size
            if w < min_image_size or h < min_image_size:
                continue  # icône/puce décorative
            thumb_hash = hashlib.md5(pil_img.resize((16, 16)).tobytes()).hexdigest()
            if thumb_hash in seen_hashes:
                continue
            seen_hashes.add(thumb_hash)
            img_name = f"{base_name}_p{page_idx + 1}_{local_count}.png"
            img_path = os.path.join(images_dir, img_name)
            pil_img.save(img_path)
            images_by_page[page_idx].append(img_path)
            local_count += 1

    pages = []
    for page_idx, text in enumerate(texts_by_page):
        pages.append({
            "page": page_idx + 1,
            "text": text,
            "image_paths": images_by_page[page_idx] if page_idx < len(images_by_page) else [],
        })
    return pages


def ingest_sources(config, min_image_size=60):
    raw_dir = config["paths"]["raw_pdf_dir"]
    images_dir = config["paths"]["images_dir"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    source_meta = {}
    meta_path = os.path.join(raw_dir, SOURCE_META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            source_meta = json.load(f)

    files = sorted(
        f for f in os.listdir(raw_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )
    if not files:
        print(f"[ingest] Aucun document pris en charge dans {raw_dir}.")

    pages_data = []
    for fname in files:
        doc_path = os.path.join(raw_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        stype = source_meta.get("source_type") or source_type_from_ext(ext)
        url = source_meta.get("url", "")
        doc_name = source_meta.get("title") or fname

        if ext == ".pdf":
            pages = _ingest_pdf(doc_path, fname, images_dir, min_image_size)
        else:
            _, sections = load_non_pdf(doc_path)
            pages = [{"page": i + 1, "text": s, "image_paths": []}
                     for i, s in enumerate(sections)]

        for p in pages:
            pages_data.append({
                "doc": doc_name,
                "page": p["page"],
                "text": p["text"],
                "image_paths": p["image_paths"],
                "source_type": stype,
                "source": url or fname,
            })
        nb_img = sum(len(p["image_paths"]) for p in pages)
        unit = "pages" if stype == "pdf" else "sections"
        print(f"[ingest] {doc_name} ({stype}) : {len(pages)} {unit}, {nb_img} images utiles")

    out_path = os.path.join(processed_dir, "pages_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    print(f"[ingest] TOTAL {len(pages_data)} pages/sections ingérées -> {out_path}")
    return pages_data


# Compatibilité avec l'ancien nom de la fonction
ingest_pdfs = ingest_sources


if __name__ == "__main__":
    ingest_sources(load_config())
