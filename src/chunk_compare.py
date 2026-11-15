"""
Comparaison de méthodes de chunking (version allégée de la consigne du cours :
4 méthodes au lieu de 7, métriques principales du tableau du TP2-Séance3).

Méthodes testées :
1. fixed_small     : découpage par taille fixe (mots), chunks courts
2. fixed_large     : découpage par taille fixe (mots), chunks longs (config actuelle)
3. by_sentence     : découpage par groupes de phrases
4. recursive_like  : découpage hiérarchique (paragraphe -> phrase -> mot), à la LangChain

Métriques calculées (sous-ensemble du tableau du cours, calculable sans jeu de
questions/réponses de référence, qu'on n'a pas dans le temps imparti) :
- Nombre de chunks
- Longueur moyenne (mots)
- Écart-type de la longueur
- Similarité intra (cohérence interne d'un chunk : similarité entre sa 1ère et sa 2e moitié)
- Similarité inter (redondance entre chunks : similarité moyenne entre chunks voisins)

Recall@k / Precision@k / F1-score du tableau du cours nécessitent un jeu de
questions-réponses annotées manuellement (vérité terrain), qu'on n'a pas ici faute
de temps -> non calculés, mentionné explicitement dans le rapport de sortie.

La méthode gagnante (meilleur compromis cohérence/non-redondance) est utilisée
pour régénérer chunks.json, en conservant la métadonnée (doc, page, images).
"""
import json
import os
import re
import statistics
import numpy as np
from sentence_transformers import SentenceTransformer
from src.ingest import load_config


# ---------------------------------------------------------------------------
# Méthodes de chunking (chacune prend un texte de page et retourne une liste
# de chunks texte ; le découpage se fait page par page pour garder la
# traçabilité doc/page simple, comme dans le pipeline principal)
# ---------------------------------------------------------------------------

def fixed_words(text, size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + size]
        if chunk:
            chunks.append(" ".join(chunk))
        i += max(size - overlap, 1)
    return chunks


def by_sentence(text, sentences_per_chunk=5):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def recursive_like(text, target_words=180, overlap_words=30):
    """Découpage hiérarchique simplifié : d'abord par paragraphe (retours à la
    ligne), puis fusionne les paragraphes courts jusqu'à la taille cible, et
    redécoupe par phrase si un paragraphe dépasse largement la cible."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    def flush():
        if current:
            chunks.append(" ".join(current))

    for p in paragraphs:
        p_words = p.split()
        if len(p_words) > target_words * 1.5:
            # paragraphe trop long -> redécoupe par phrase
            for sub in by_sentence(p, sentences_per_chunk=3):
                current.append(sub)
                current_len += len(sub.split())
                if current_len >= target_words:
                    flush()
                    # chevauchement : on garde les derniers mots
                    overlap_text = " ".join(" ".join(current).split()[-overlap_words:])
                    current = [overlap_text] if overlap_text else []
                    current_len = len(overlap_text.split())
        else:
            current.append(p)
            current_len += len(p_words)
            if current_len >= target_words:
                flush()
                overlap_text = " ".join(" ".join(current).split()[-overlap_words:])
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text.split())
    flush()
    return [c for c in chunks if c.strip()]


METHODS = {
    "fixed_small (100 mots / 20 overlap)": lambda t: fixed_words(t, 100, 20),
    "fixed_large (220 mots / 40 overlap)": lambda t: fixed_words(t, 220, 40),
    "by_sentence (5 phrases/chunk)": lambda t: by_sentence(t, 5),
    "recursive_like (para->phrase, ~180 mots)": lambda t: recursive_like(t, 180, 30),
}


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(all_chunks_text, model, max_sample=250):
    """Calcule les métriques sur un échantillon (pour rester rapide)."""
    sample = all_chunks_text[:max_sample]
    lengths = [len(c.split()) for c in sample]

    nb_chunks_total = len(all_chunks_text)
    longueur_moyenne = statistics.mean(lengths) if lengths else 0
    ecart_type = statistics.pstdev(lengths) if len(lengths) > 1 else 0

    # Similarité intra : cohérence entre 1ère et 2e moitié de chaque chunk
    halves_a, halves_b, valid_idx = [], [], []
    for i, c in enumerate(sample):
        words = c.split()
        if len(words) < 6:
            continue
        mid = len(words) // 2
        halves_a.append(" ".join(words[:mid]))
        halves_b.append(" ".join(words[mid:]))
        valid_idx.append(i)

    if halves_a:
        emb_a = model.encode(halves_a, normalize_embeddings=True, show_progress_bar=False)
        emb_b = model.encode(halves_b, normalize_embeddings=True, show_progress_bar=False)
        sims_intra = np.sum(emb_a * emb_b, axis=1)  # cosine (vecteurs normalisés)
        similarite_intra = float(np.mean(sims_intra))
    else:
        similarite_intra = 0.0

    # Similarité inter : redondance moyenne entre chunks consécutifs (proxy)
    if len(sample) > 1:
        embs = model.encode(sample, normalize_embeddings=True, show_progress_bar=False)
        sims_inter = [float(np.dot(embs[i], embs[i + 1])) for i in range(len(embs) - 1)]
        similarite_inter = statistics.mean(sims_inter)
    else:
        similarite_inter = 0.0

    return {
        "nb_chunks_total": nb_chunks_total,
        "longueur_moyenne_mots": round(longueur_moyenne, 1),
        "ecart_type_longueur": round(ecart_type, 1),
        "similarite_intra": round(similarite_intra, 4),
        "similarite_inter": round(similarite_inter, 4),
    }


def score_method(metrics):
    """Score composite simple : favorise une cohérence interne élevée
    (similarite_intra) et une redondance faible entre chunks (similarite_inter
    basse), avec une pénalité si les chunks sont trop courts (<30 mots, peu
    informatifs) ou trop irréguliers (écart-type élevé)."""
    score = metrics["similarite_intra"] - metrics["similarite_inter"]
    if metrics["longueur_moyenne_mots"] < 30:
        score -= 0.2
    if metrics["ecart_type_longueur"] > metrics["longueur_moyenne_mots"]:
        score -= 0.1
    return score


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main(cfg=None):
    if cfg is None:
        cfg = load_config()
    processed_dir = cfg["paths"]["processed_dir"]

    with open(os.path.join(processed_dir, "pages_raw.json"), "r", encoding="utf-8") as f:
        pages = json.load(f)

    print("Chargement du modèle d'embedding (peut prendre quelques secondes)...")
    model = SentenceTransformer(cfg["embeddings"]["text_model"])

    results = {}
    chunks_by_method = {}

    for name, fn in METHODS.items():
        print(f"\n--- Méthode : {name} ---")
        method_chunks = []  # avec métadonnées
        for p in pages:
            for tc in fn(p["text"]):
                if len(tc.strip()) < 5:
                    continue
                method_chunks.append({
                    "content": tc,
                    "doc": p["doc"],
                    "page": p["page"],
                    "image_paths": p["image_paths"],
                })
        texts_only = [c["content"] for c in method_chunks]
        metrics = compute_metrics(texts_only, model)
        metrics["score"] = round(score_method(metrics), 4)
        results[name] = metrics
        chunks_by_method[name] = method_chunks
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # --- Tableau récapitulatif ---
    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF")
    print("=" * 100)
    header = f"{'Méthode':45} {'Nb chunks':>10} {'Long. moy.':>11} {'Écart-type':>11} {'Sim. intra':>11} {'Sim. inter':>11} {'Score':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:45} {m['nb_chunks_total']:>10} {m['longueur_moyenne_mots']:>11} "
              f"{m['ecart_type_longueur']:>11} {m['similarite_intra']:>11} {m['similarite_inter']:>11} {m['score']:>8}")

    best_name = max(results, key=lambda n: results[n]["score"])
    print(f"\n>>> Méthode retenue automatiquement : {best_name} (score={results[best_name]['score']})")
    print(">>> Note méthodologique : Recall@k, Precision@k et F1-score du tableau du cours")
    print("    nécessitent une vérité terrain (questions/réponses annotées manuellement),")
    print("    non disponible dans le temps imparti -> non calculés ici, à mentionner en limite du travail.")

    # --- Sauvegarde du rapport comparatif ---
    report = {
        "results": results,
        "best_method": best_name,
        "note": "Recall@k / Precision@k / F1-score non calculés : nécessitent une vérité terrain annotée manuellement.",
    }
    with open(os.path.join(processed_dir, "chunking_comparison_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRapport sauvegardé -> {os.path.join(processed_dir, 'chunking_comparison_report.json')}")

    # --- Régénère chunks.json avec la méthode gagnante ---
    final_chunks = []
    for i, c in enumerate(chunks_by_method[best_name]):
        final_chunks.append({
            "id": i,
            "type": "text",
            "content": c["content"],
            "doc": c["doc"],
            "page": c["page"],
            "image_paths": c["image_paths"],
        })
    with open(os.path.join(processed_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    print(f"chunks.json régénéré avec la méthode '{best_name}' ({len(final_chunks)} chunks) -> prêt pour l'OCR puis l'indexation.")

    return report


if __name__ == "__main__":
    main()
