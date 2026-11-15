"""
Etape 4 - Embeddings + Indexation
Deux index séparés :
1) Index TEXTE (sentence-transformers) sur tous les chunks (texte natif + OCR).
2) Index IMAGE (CLIP) sur les images extraites, pour pouvoir retrouver une image
   directement à partir du texte d'une question (recherche cross-modale).
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.ingest import load_config


def build_text_index(config):
    processed_dir = config["paths"]["processed_dir"]
    index_dir = config["paths"]["index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    with open(os.path.join(processed_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(config["embeddings"]["text_model"])
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # produit scalaire = cosine car vecteurs normalisés
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "text.index"))
    with open(os.path.join(index_dir, "text_meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[index] Index TEXTE créé : {index.ntotal} vecteurs")


def build_image_index(config):
    processed_dir = config["paths"]["processed_dir"]
    index_dir = config["paths"]["index_dir"]

    with open(os.path.join(processed_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    image_meta = {}
    for c in chunks:
        for img_path in c.get("image_paths", []):
            if img_path not in image_meta:
                image_meta[img_path] = {"doc": c["doc"], "page": c["page"], "path": img_path}

    if not image_meta:
        print("[index] Aucune image trouvée -> index image ignoré")
        return

    from PIL import Image
    clip_model = SentenceTransformer(config["embeddings"]["clip_model"])
    paths = list(image_meta.keys())
    images = []
    valid_paths = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as e:
            print(f"[index] Image illisible ignorée {p}: {e}")

    if not images:
        print("[index] Aucune image exploitable -> index image ignoré")
        return

    embeddings = clip_model.encode(images, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "image.index"))
    meta_list = [image_meta[p] for p in valid_paths]
    with open(os.path.join(index_dir, "image_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    print(f"[index] Index IMAGE créé : {index.ntotal} vecteurs")


if __name__ == "__main__":
    cfg = load_config()
    build_text_index(cfg)
    build_image_index(cfg)
