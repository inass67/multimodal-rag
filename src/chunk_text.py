"""
Etape 2 - Chunking
Découpe le texte de chaque page en chunks avec chevauchement,
en conservant la source (doc, page) et les images associées à la page.
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
        text_chunks = simple_chunk(p["text"], chunk_size, overlap)
        for tc in text_chunks:
            if len(tc.strip()) < 5:
                continue
            all_chunks.append({
                "id": chunk_id,
                "type": "text",
                "content": tc,
                "doc": p["doc"],
                "page": p["page"],
                "image_paths": p["image_paths"],
            })
            chunk_id += 1

    out_path = os.path.join(processed_dir, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[chunk] {len(all_chunks)} chunks texte -> {out_path}")
    return all_chunks


if __name__ == "__main__":
    cfg = load_config()
    build_chunks(cfg)
