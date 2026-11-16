"""
Comparaison de modèles d'EMBEDDING (point clé du sujet : « choix de l'embedding »).

Pour chaque modèle listé dans config.yaml (evaluation.embedding_models) :
  1. on encode tous les passages du corpus ;
  2. on encode les questions du jeu de test (eval/eval_set.json) ;
  3. on mesure, en recherche sémantique pure, Recall@1/3/5, Precision@k et MRR ;
  4. on mesure aussi le coût : temps d'encodage et dimension des vecteurs.

Le meilleur compromis qualité / coût est recommandé, et c'est ce modèle qu'on
inscrit dans config.yaml (embeddings.text_model).

Pourquoi c'est important ici : l'ancien modèle (all-MiniLM-L6-v2) a été
entraîné presque uniquement sur de l'anglais, alors que nos documents sont en
français. Un modèle multilingue retrouve en général bien mieux les passages
pertinents pour des questions en français.

Utilisation (après avoir créé le jeu de test avec src.evaluate) :
    python -m src.compare_embeddings
"""
import json
import os
import time
from datetime import datetime

import numpy as np

from src.embed_index import encode_texts
from src.evaluate import EVAL_DIR, load_eval_set, ranking_metrics

REPORT_PATH = os.path.join(EVAL_DIR, "embeddings_report.json")
REPORT_MD_PATH = os.path.join(EVAL_DIR, "embeddings_report.md")

DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
]


def compare_embedding_models(chunks, items, models, ks=(1, 3, 5)):
    texts = [c["content"] for c in chunks]
    questions = [it["question"] for it in items]
    report = {}
    for name in models:
        print(f"[embeddings] Modèle {name} ...")
        try:
            t0 = time.time()
            passage_emb = encode_texts(name, texts)
            t_passages = time.time() - t0
            q_emb = encode_texts(name, questions, is_query=True)
        except Exception as e:
            print(f"[embeddings] Échec pour {name} : {e}")
            report[name] = {"erreur": str(e)}
            continue
        scores = q_emb @ passage_emb.T
        top = np.argsort(-scores, axis=1)[:, :max(ks)]
        ranked = [[chunks[j] for j in row] for row in top]
        metrics = ranking_metrics(ranked, items, ks)
        metrics["dimension"] = int(passage_emb.shape[1])
        metrics["temps_encodage_s"] = round(t_passages, 2)
        metrics["ms_par_passage"] = round(1000 * t_passages / max(len(texts), 1), 2)
        report[name] = metrics
        print(f"[embeddings] {name}: {metrics}")
    return report


def best_model(report):
    """Meilleur modèle : MRR le plus élevé, puis Recall@5, puis le plus rapide."""
    valid = {k: v for k, v in report.items() if "erreur" not in v}
    if not valid:
        return None
    return max(valid, key=lambda m: (valid[m]["MRR"], valid[m]["Recall@5"], -valid[m]["ms_par_passage"]))


def to_markdown(full):
    lines = [f"# Comparaison des modèles d'embedding ({full['date']})", "",
             f"- Passages : {full['nb_passages']} — Questions : {full['nb_questions']}", "",
             "| Modèle | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | Dimension | ms / passage |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in full["resultats"].items():
        if "erreur" in m:
            lines.append(f"| {name} | erreur : {m['erreur'][:60]} |||||||")
            continue
        lines.append(f"| {name} | {m['Recall@1']:.2%} | {m['Recall@3']:.2%} | {m['Recall@5']:.2%} | "
                     f"{m['Precision@5']:.2%} | {m['MRR']:.3f} | {m['dimension']} | {m['ms_par_passage']} |")
    lines += ["", f"**Modèle recommandé : `{full['meilleur_modele']}`**", ""]
    return "\n".join(lines)


def run(retriever, config):
    models = config.get("evaluation", {}).get("embedding_models", DEFAULT_MODELS)
    items = load_eval_set(doc_names=set(retriever.doc_names))
    if not items:
        raise ValueError("Jeu de test introuvable : lancez d'abord `python -m src.evaluate --generate`.")
    results = compare_embedding_models(retriever.text_meta, items, models)
    full = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "nb_passages": len(retriever.text_meta),
        "nb_questions": len(items),
        "resultats": results,
        "meilleur_modele": best_model(results),
    }
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(to_markdown(full))
    print(to_markdown(full))
    return full


if __name__ == "__main__":
    from src.ingest import load_config
    from src.corpus import load_selection, list_documents, load_corpus_retriever
    cfg = load_config()
    hashes = load_selection() or [d["hash"] for d in list_documents()]
    run(load_corpus_retriever(cfg, hashes), cfg)
