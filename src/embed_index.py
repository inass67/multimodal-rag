"""
Étape 4 - Embeddings + indexation dans la base vectorielle (FAISS).

Deux index séparés :
1) Index TEXTE (sentence-transformers) sur tous les chunks (texte natif + OCR).
2) Index IMAGE (CLIP) sur les images extraites, pour retrouver une image
   directement à partir du texte d'une question (recherche cross-modale).

Le nom du modèle d'embedding utilisé est enregistré dans `index_info.json` :
une question doit être encodée avec le même modèle que l'index, sinon les
vecteurs ne sont pas comparables. Si on change de modèle dans config.yaml,
l'application réindexe automatiquement les anciens documents.
"""
import json
import os
import numpy as np
from src.ingest import load_config

LEGACY_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # modèle des anciens index
_MODEL_CACHE = {}


def get_embedding_model(name):
    """Charge un modèle sentence-transformers une seule fois (cache mémoire)."""
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def encode_texts(model_name, texts, is_query=False, show_progress_bar=False):
    """Encode des textes. Gère le préfixe query:/passage: exigé par les modèles E5."""
    model = get_embedding_model(model_name)
    if "e5" in model_name.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=show_progress_bar)
    return np.asarray(emb, dtype="float32")


def read_index_info(index_dir):
    path = os.path.join(index_dir, "index_info.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"text_model": LEGACY_TEXT_MODEL}


def write_index_info(index_dir, **kwargs):
    info = read_index_info(index_dir) if os.path.exists(os.path.join(index_dir, "index_info.json")) else {}
    info.update(kwargs)
    with open(os.path.join(index_dir, "index_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def build_text_index(config, model_name=None):
    import faiss
    processed_dir = config["paths"]["processed_dir"]
    index_dir = config["paths"]["index_dir"]
    os.makedirs(index_dir, exist_ok=True)
    model_name = model_name or config["embeddings"]["text_model"]

    with open(os.path.join(processed_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = encode_texts(model_name, [c["content"] for c in chunks], show_progress_bar=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # produit scalaire = cosinus (vecteurs normalisés)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "text.index"))
    with open(os.path.join(index_dir, "text_meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    write_index_info(index_dir, text_model=model_name, nb_chunks=len(chunks))
    print(f"[index] Index TEXTE créé : {index.ntotal} vecteurs ({model_name})")


def build_image_index(config):
    import faiss
    from PIL import Image
    processed_dir = config["paths"]["processed_dir"]
    index_dir = config["paths"]["index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    with open(os.path.join(processed_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    image_meta = {}
    for c in chunks:
        for img_path in c.get("image_paths", []):
            if img_path not in image_meta:
                image_meta[img_path] = {
                    "doc": c["doc"], "page": c["page"], "path": img_path,
                    "source_type": c.get("source_type", "pdf"),
                }
    if not image_meta:
        print("[index] Aucune image trouvée -> index image ignoré")
        return

    images, valid_paths = [], []
    for p in image_meta:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as e:
            print(f"[index] Image illisible ignorée {p}: {e}")
    if not images:
        print("[index] Aucune image exploitable -> index image ignoré")
        return

    clip_name = config["embeddings"]["clip_model"]
    embeddings = get_embedding_model(clip_name).encode(
        images, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "image.index"))
    with open(os.path.join(index_dir, "image_meta.json"), "w", encoding="utf-8") as f:
        json.dump([image_meta[p] for p in valid_paths], f, ensure_ascii=False, indent=2)
    write_index_info(index_dir, clip_model=clip_name)
    print(f"[index] Index IMAGE créé : {index.ntotal} vecteurs")


if __name__ == "__main__":
    cfg = load_config()
    build_text_index(cfg)
    build_image_index(cfg)
