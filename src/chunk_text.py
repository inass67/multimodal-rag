"""
Étape 2 - Chunking simple (taille fixe en mots, avec chevauchement).
Utilisé en ligne de commande. L'application utilise chunk_compare.py, qui
compare plusieurs méthodes et garde automatiquement la meilleure.
La provenance (doc, page, type de source, URL, images) est conservée.
"""
import json
import os
from src.ingest import load_config


def simple_chunk(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        i += max(chunk_size - overlap, 1)
    return chunks


def build_chunks(config):
    processed_dir = config["paths"]["processed_dir"]
    with open(os.path.join(processed_dir, "pages_raw.json"), "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunk_size = config["chunking"]["chunk_size"]
    overlap = config["chunking"]["chunk_overlap"]

    all_chunks = []
    chunk_id = 0
    for p in pages:
        for tc in simple_chunk(p["text"], chunk_size, overlap):
            if len(tc.strip()) < 5:
                continue
            all_chunks.append({
                "id": chunk_id,
                "type": "text",
                "content": tc,
                "doc": p["doc"],
                "page": p["page"],
                "image_paths": p["image_paths"],
                "source_type": p.get("source_type", "pdf"),
                "source": p.get("source", p["doc"]),
            })
            chunk_id += 1

    out_path = os.path.join(processed_dir, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[chunk] {len(all_chunks)} chunks texte -> {out_path}")
    return all_chunks


if __name__ == "__main__":
    build_chunks(load_config())
