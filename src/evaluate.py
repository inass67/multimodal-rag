"""
Évaluation du système (point clé du sujet : « évaluation de la pertinence des citations »).

1. JEU DE TEST (vérité terrain)
   Écrire des centaines de questions à la main prend du temps. On génère donc
   un jeu de test SYNTHÉTIQUE : on tire au hasard des passages du corpus et le
   LLM écrit, pour chacun, une question dont la réponse se trouve dans ce
   passage. La vérité terrain est alors connue : le (document, page) du passage.
   Le fichier eval/eval_set.json peut aussi être complété ou corrigé à la main.

2. ÉVALUATION DE LA RECHERCHE (retrieval), 3 configurations comparées :
   sémantique seule / BM25 seul / hybride (celle utilisée par l'application).
   - Recall@k    : la bonne page est-elle dans les k premiers résultats ?
   - Precision@k : proportion des k résultats qui viennent de la bonne page.
   - MRR         : 1 / rang du premier bon résultat (moyenne).

3. ÉVALUATION DES CITATIONS (réponses générées) :
   - Taux de réponse            : questions où l'agent ne répond pas « Je ne sais pas ».
   - Validité des citations     : % de numéros [n] qui correspondent à une vraie source.
   - Précision des citations    : % des sources citées qui viennent de la bonne page.
   - Taux de succès des citations : % de réponses citant au moins la bonne source.
   - Support (fidélité)         : % de phrases citées dont le contenu est réellement
                                  proche de la source citée (similarité d'embedding).
   - (option) Juge LLM          : le LLM vérifie si chaque source citée soutient la phrase.

Utilisation :
    python -m src.evaluate --generate --n 15     # génère le jeu de test puis évalue
    python -m src.evaluate                       # réutilise eval/eval_set.json
"""
import argparse
import json
import os
import random
import re
import time
from datetime import datetime

import numpy as np

from src.llm import run_chain, LLMError
from src.generate import answer_question
from src.embed_index import encode_texts

EVAL_DIR = "eval"
EVAL_SET_PATH = os.path.join(EVAL_DIR, "eval_set.json")
REPORT_PATH = os.path.join(EVAL_DIR, "evaluation_report.json")
REPORT_MD_PATH = os.path.join(EVAL_DIR, "evaluation_report.md")

QGEN_TEMPLATE = """Tu crées un jeu de test pour évaluer un moteur de recherche documentaire.
Pour CHAQUE passage numéroté ci-dessous, écris UNE question précise et naturelle :
- dont la réponse se trouve dans ce passage ;
- compréhensible sans voir le passage (n'écris pas « selon ce passage ») ;
- dans la langue du passage.
Donne aussi la réponse attendue en une phrase courte.

Réponds UNIQUEMENT avec une liste JSON de la forme :
[{{"passage": 1, "question": "...", "reponse": "..."}}]

Passages :
{passages}
"""

JUDGE_TEMPLATE = """La phrase suivante est-elle entièrement justifiée par la source ?
Réponds uniquement par OUI ou NON.

Phrase : {sentence}

Source : {source}
"""


# ---------------------------------------------------------------------------
# 1. Jeu de test
# ---------------------------------------------------------------------------
def generate_eval_set(retriever, config, n=15, seed=42, batch_size=5):
    rng = random.Random(seed)
    candidates = [(i, c) for i, c in enumerate(retriever.text_meta)
                  if c.get("type", "text") == "text" and len(c["content"].split()) >= 40]
    if not candidates:
        raise ValueError("Pas assez de texte dans le corpus pour générer un jeu de test.")

    # Répartition équitable entre les documents (tirage tour à tour)
    by_doc = {}
    for i, c in candidates:
        by_doc.setdefault(c["doc"], []).append((i, c))
    for lst in by_doc.values():
        rng.shuffle(lst)
    picked = []
    while len(picked) < n and any(by_doc.values()):
        for doc in list(by_doc):
            if by_doc[doc] and len(picked) < n:
                picked.append(by_doc[doc].pop())

    items = []
    for start in range(0, len(picked), batch_size):
        batch = picked[start:start + batch_size]
        passages = "\n\n".join(f"[{k + 1}] {c['content'][:1500]}" for k, (_, c) in enumerate(batch))
        raw = run_chain(QGEN_TEMPLATE, {"passages": passages}, config, temperature=0.3)
        m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        try:
            generated = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            generated = []
        for g in generated:
            k = int(g.get("passage", 0)) - 1
            if 0 <= k < len(batch) and g.get("question"):
                idx, c = batch[k]
                items.append({
                    "question": g["question"].strip(),
                    "reponse_attendue": g.get("reponse", "").strip(),
                    "doc": c["doc"], "page": c["page"], "chunk_index": idx,
                })

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[eval] {len(items)} questions générées -> {EVAL_SET_PATH}")
    return items


def load_eval_set(path=EVAL_SET_PATH, doc_names=None):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if doc_names is not None:  # ne garder que les questions sur les documents du corpus
        items = [it for it in items if it["doc"] in doc_names]
    return items


# ---------------------------------------------------------------------------
# 2. Métriques de recherche
# ---------------------------------------------------------------------------
def is_relevant(result, item):
    return result["doc"] == item["doc"] and result["page"] == item["page"]


def ranking_metrics(ranked_lists, items, ks=(1, 3, 5)):
    """ranked_lists[i] = liste ordonnée de résultats pour la question i."""
    out = {f"Recall@{k}": [] for k in ks}
    out.update({f"Precision@{k}": [] for k in ks})
    out["MRR"] = []
    for results, item in zip(ranked_lists, items):
        rel = [is_relevant(r, item) for r in results]
        for k in ks:
            out[f"Recall@{k}"].append(1.0 if any(rel[:k]) else 0.0)
            out[f"Precision@{k}"].append(sum(rel[:k]) / k)
        first = next((i for i, x in enumerate(rel) if x), None)
        out["MRR"].append(1.0 / (first + 1) if first is not None else 0.0)
    return {name: round(float(np.mean(v)), 4) if v else 0.0 for name, v in out.items()}


def evaluate_retrieval(retriever, items, ks=(1, 3, 5)):
    configs = {"Sémantique seule": 1.0, "BM25 seul": 0.0,
               "Hybride (application)": retriever.config["retrieval"].get("alpha", 0.5)}
    report = {}
    for name, alpha in configs.items():
        t0 = time.time()
        ranked = [retriever.retrieve_text(it["question"], k=max(ks), alpha=alpha) for it in items]
        metrics = ranking_metrics(ranked, items, ks)
        metrics["temps_moyen_ms"] = round(1000 * (time.time() - t0) / max(len(items), 1), 1)
        report[name] = metrics
        print(f"[eval] Recherche {name}: {metrics}")
    return report


# ---------------------------------------------------------------------------
# 3. Métriques des citations
# ---------------------------------------------------------------------------
CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def cited_sentences(body):
    """Découpe la réponse en phrases et retourne [(phrase_sans_citation, [numéros])]."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        nums = []
        for g in CITATION_RE.findall(sent):
            nums.extend(int(x) for x in re.split(r"[,;]", g))
        clean = CITATION_RE.sub("", sent).strip(" -*#\t")
        if nums and len(clean.split()) >= 3:
            out.append((clean, nums))
    return out


def citation_support(body, text_results, model_name, threshold):
    """Fidélité : chaque phrase citée est-elle sémantiquement proche de sa source ?"""
    pairs = [(s, [n for n in nums if 1 <= n <= len(text_results)])
             for s, nums in cited_sentences(body)]
    pairs = [(s, nums) for s, nums in pairs if nums]
    if not pairs:
        return None, []
    sent_emb = encode_texts(model_name, [s for s, _ in pairs], is_query=True)
    details = []
    for (s, nums), e in zip(pairs, sent_emb):
        src_emb = encode_texts(model_name, [text_results[n - 1]["content"] for n in nums])
        best = float(np.max(src_emb @ e))
        details.append({"phrase": s, "sources": nums, "similarite": round(best, 3),
                        "soutenue": best >= threshold})
    rate = sum(d["soutenue"] for d in details) / len(details)
    return rate, details


def llm_judge(details, text_results, config):
    verdicts = []
    for d in details:
        src = "\n".join(text_results[n - 1]["content"] for n in d["sources"])
        try:
            raw = run_chain(JUDGE_TEMPLATE, {"sentence": d["phrase"], "source": src[:3000]},
                            config, temperature=0.0)
            verdicts.append(raw.strip().upper().startswith("OUI"))
        except LLMError:
            continue
    return sum(verdicts) / len(verdicts) if verdicts else None


def evaluate_citations(retriever, items, config, max_questions=10, use_llm_judge=False):
    ev = config.get("evaluation", {})
    threshold = ev.get("support_threshold", 0.5)
    per_question = []
    for it in items[:max_questions]:
        text_results = retriever.retrieve_text(it["question"])
        try:
            res = answer_question(it["question"], text_results, config)
        except LLMError as e:
            print(f"[eval] LLM indisponible : {e}")
            break
        n_cited = len(res["cited"]) + len(res["invalid"])
        relevant_cited = [n for n in res["cited"] if is_relevant(text_results[n - 1], it)]
        support, details = citation_support(res["body"], text_results,
                                            retriever.text_model_name, threshold)
        row = {
            "question": it["question"],
            "repondu": not res["no_answer"],
            "nb_citations": n_cited,
            "validite": len(res["cited"]) / n_cited if n_cited else None,
            "precision_citations": len(relevant_cited) / len(res["cited"]) if res["cited"] else None,
            "succes_citation": 1.0 if relevant_cited else 0.0,
            "support": support,
        }
        if use_llm_judge and details:
            row["juge_llm"] = llm_judge(details, text_results, config)
        per_question.append(row)
        print(f"[eval] {it['question'][:60]}... -> {row}")

    def mean(key):
        vals = [r[key] for r in per_question if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "nb_questions": len(per_question),
        "Taux de réponse": mean("repondu"),
        "Validité des citations": mean("validite"),
        "Précision des citations": mean("precision_citations"),
        "Taux de succès des citations": mean("succes_citation"),
        "Support sémantique des phrases citées": mean("support"),
    }
    if use_llm_judge:
        summary["Juge LLM (phrases soutenues)"] = mean("juge_llm")
    return summary, per_question


# ---------------------------------------------------------------------------
# Rapport complet
# ---------------------------------------------------------------------------
def run_full_evaluation(retriever, config, generate=False, n_questions=15,
                        max_answer_questions=None, use_llm_judge=None):
    ev = config.get("evaluation", {})
    max_answer_questions = max_answer_questions or ev.get("max_answer_questions", 10)
    use_llm_judge = ev.get("llm_judge", False) if use_llm_judge is None else use_llm_judge

    if generate or not os.path.exists(EVAL_SET_PATH):
        items = generate_eval_set(retriever, config, n=n_questions)
    else:
        items = load_eval_set(doc_names=set(retriever.doc_names))
    if not items:
        raise ValueError("Jeu de test vide : générez-le avec --generate.")

    retrieval = evaluate_retrieval(retriever, items)
    citations, per_question = evaluate_citations(
        retriever, items, config, max_questions=max_answer_questions, use_llm_judge=use_llm_judge)

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "documents": retriever.doc_names,
        "modele_embedding": retriever.text_model_name,
        "nb_questions_recherche": len(items),
        "recherche": retrieval,
        "citations": citations,
        "detail_citations": per_question,
    }
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(report_to_markdown(report))
    print(f"[eval] Rapport -> {REPORT_PATH} et {REPORT_MD_PATH}")
    return report


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.2%}" if isinstance(v, float) and v <= 1 else str(v)


def report_to_markdown(report):
    lines = [f"# Rapport d'évaluation ({report['date']})", "",
             f"- Documents : {', '.join(report['documents'])}",
             f"- Modèle d'embedding : `{report['modele_embedding']}`",
             f"- Questions de test : {report['nb_questions_recherche']}", "",
             "## Qualité de la recherche (retrieval)", ""]
    metrics = list(next(iter(report["recherche"].values())).keys())
    lines.append("| Configuration | " + " | ".join(metrics) + " |")
    lines.append("|---" * (len(metrics) + 1) + "|")
    for name, m in report["recherche"].items():
        cells = [f"{m[k]} ms" if k == "temps_moyen_ms" else
                 f"{m[k]:.3f}" if k == "MRR" else _fmt(m[k]) for k in metrics]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Pertinence des citations", "", "| Métrique | Valeur |", "|---|---|"]
    for k, v in report["citations"].items():
        lines.append(f"| {k} | {_fmt(v) if k != 'nb_questions' else v} |")
    return "\n".join(lines) + "\n"


def main():
    from src.ingest import load_config
    from src.corpus import load_selection, list_documents, load_corpus_retriever

    parser = argparse.ArgumentParser(description="Évaluation du RAG")
    parser.add_argument("--generate", action="store_true", help="(re)générer le jeu de test")
    parser.add_argument("--n", type=int, default=15, help="nombre de questions à générer")
    parser.add_argument("--max-answers", type=int, default=None,
                        help="nombre de questions pour l'évaluation des citations (appels LLM)")
    parser.add_argument("--judge", action="store_true", help="activer le juge LLM")
    args = parser.parse_args()

    cfg = load_config()
    hashes = load_selection() or [d["hash"] for d in list_documents()]
    retriever = load_corpus_retriever(cfg, hashes)
    report = run_full_evaluation(retriever, cfg, generate=args.generate, n_questions=args.n,
                                 max_answer_questions=args.max_answers,
                                 use_llm_judge=args.judge or None)
    print(report_to_markdown(report))


if __name__ == "__main__":
    main()
