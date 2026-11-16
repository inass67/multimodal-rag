"""
Étape 3 - OCR
Passe chaque image extraite dans Tesseract pour capter le texte qu'elle
contient (légendes, texte dans les figures, documents scannés...).
Ces textes deviennent des chunks de type "ocr_image", cherchables comme du
texte normal. Si Tesseract n'est pas installé, l'étape est simplement sautée
(le reste du pipeline fonctionne quand même).
"""
import json
import os
from PIL import Image
from src.ingest import load_config


def tesseract_available():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def run_ocr_on_images(config):
    processed_dir = config["paths"]["processed_dir"]
    chunks_path = os.path.join(processed_dir, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    has_images = any(c.get("image_paths") for c in chunks)
    if not has_images:
        print("[ocr] Aucune image -> OCR inutile")
        return chunks
    if not tesseract_available():
        print("[ocr] Tesseract introuvable -> OCR ignoré (voir README pour l'installer)")
        return chunks

    import pytesseract
    next_id = max([c["id"] for c in chunks], default=-1) + 1
    seen_images = set()
    ocr_chunks = []

    for c in chunks:
        for img_path in c.get("image_paths", []):
            if img_path in seen_images:
                continue
            seen_images.add(img_path)
            try:
                ocr_text = pytesseract.image_to_string(Image.open(img_path), lang="fra+eng").strip()
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
                    "source_type": c.get("source_type", "pdf"),
                    "source": c.get("source", c["doc"]),
                })
                next_id += 1

    all_chunks = chunks + ocr_chunks
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[ocr] {len(ocr_chunks)} chunks OCR ajoutés (total {len(all_chunks)})")
    return all_chunks


if __name__ == "__main__":
    run_ocr_on_images(load_config())
