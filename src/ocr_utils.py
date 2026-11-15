"""
Etape 3 - OCR
Passe chaque image extraite dans Tesseract pour capter le texte
qu'elle contient (légendes, texte dans les figures...).
Ces textes deviennent des chunks de type "ocr_image", cherchables comme du texte normal.
"""
import json
import os
import pytesseract
from PIL import Image
from src.ingest import load_config


def run_ocr_on_images(config):
    processed_dir = config["paths"]["processed_dir"]
    chunks_path = os.path.join(processed_dir, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    next_id = max([c["id"] for c in chunks], default=-1) + 1
    seen_images = set()
    ocr_chunks = []

    for c in chunks:
        for img_path in c.get("image_paths", []):
            if img_path in seen_images:
                continue
            seen_images.add(img_path)
            try:
                img = Image.open(img_path)
                ocr_text = pytesseract.image_to_string(img, lang="fra+eng").strip()
            except Exception as e:
                print(f"[ocr] Erreur sur {img_path}: {e}")
                ocr_text = ""
            if len(ocr_text) >= 5:
                ocr_chunks.append({
                    "id": next_id,
                    "type": "ocr_image",
                    "content": ocr_text,
                    "doc": c["doc"],
                    "page": c["page"],
                    "image_paths": [img_path],
                })
                next_id += 1

    all_chunks = chunks + ocr_chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[ocr] {len(ocr_chunks)} chunks OCR ajoutés (total {len(all_chunks)})")
    return all_chunks


if __name__ == "__main__":
    cfg = load_config()
    run_ocr_on_images(cfg)
