"""
Etape 1 - Ingestion (version testée en conditions réelles sur Guide-papillons.pdf)
Extrait, pour CHAQUE PDF trouvé dans data/raw :
- le texte de chaque page (via pdfplumber)
- les images embarquées de chaque page (via pypdfium2), en filtrant les
  petites icônes/puces décoratives et en dédoublonnant les images répétées.
Ne connaît rien du domaine (papillons, oiseaux, admin...) -> 100% générique.
"""
import os
import json
import hashlib
import yaml
import pdfplumber
import pypdfium2 as pdfium


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ingest_pdfs(config, min_image_size=60):
    raw_dir = config["paths"]["raw_pdf_dir"]
    images_dir = config["paths"]["images_dir"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    pages_data = []

    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[ingest] Aucun PDF trouvé dans {raw_dir}. Dépose ta thèse (ou tout autre PDF) ici.")

    for fname in pdf_files:
        doc_path = os.path.join(raw_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # --- Texte ---
        with pdfplumber.open(doc_path) as pdf:
            texts_by_page = [(page.extract_text() or "") for page in pdf.pages]

        # --- Images (filtrées + dédoublonnées) ---
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
                    continue  # icône/puce décorative, pas une vraie photo
                thumb_hash = hashlib.md5(pil_img.resize((16, 16)).tobytes()).hexdigest()
                if thumb_hash in seen_hashes:
                    continue
                seen_hashes.add(thumb_hash)
                img_name = f"{base_name}_p{page_idx + 1}_{local_count}.png"
                img_path = os.path.join(images_dir, img_name)
                pil_img.save(img_path)
                images_by_page[page_idx].append(img_path)
                local_count += 1

        for page_idx in range(len(texts_by_page)):
            pages_data.append({
                "doc": fname,
                "page": page_idx + 1,
                "text": texts_by_page[page_idx],
                "image_paths": images_by_page[page_idx] if page_idx < len(images_by_page) else [],
            })

        print(f"[ingest] {fname} : {len(texts_by_page)} pages, "
              f"{sum(len(p) for p in images_by_page)} images utiles extraites")

    out_path = os.path.join(processed_dir, "pages_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    print(f"[ingest] TOTAL {len(pages_data)} pages ingérées -> {out_path}")
    return pages_data


if __name__ == "__main__":
    cfg = load_config()
    ingest_pdfs(cfg)
