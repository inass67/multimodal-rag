"""
Résumé des sources (exigence du sujet : « Résume les sources »).

Trois niveaux :
1. Résumé de chaque source utilisée dans une réponse : produit directement
   dans la réponse (voir generate.py, section « Résumé des sources utilisées »).
2. Résumé d'un document entier : approche MAP-REDUCE (comme la chaîne
   « map_reduce » de LangChain) :
     - MAP    : le document est découpé en lots de passages, chaque lot est
                résumé séparément avec les numéros de pages ;
     - REDUCE : les résumés partiels sont fusionnés en un résumé final structuré.
   Le résultat est mis en cache (processed/summary.md) : on ne paie le coût
   qu'une seule fois par document.
3. Synthèse du corpus : compare les résumés de tous les documents sélectionnés
   (thèmes communs, différences, complémentarités).
"""
import os

from src.llm import run_chain
from src.retrieve import location_label

MAP_TEMPLATE = """Résume l'extrait suivant du document « {doc} » en 3 à 6 puces factuelles.
Termine chaque puce par la localisation entre parenthèses, par exemple (page 4) ou (section 2).
N'ajoute aucune information absente de l'extrait. Écris en français.

Extrait :
{text}
"""

REDUCE_TEMPLATE = """Voici des résumés partiels du document « {doc} ».
Fusionne-les en UN résumé final en Markdown, sans répétition, avec exactement ces rubriques :

**Sujet** : une phrase.
**Points clés** : 4 à 8 puces, en gardant les localisations (page/section) entre parenthèses.
**À retenir** : une ou deux phrases.

N'invente rien. Écris en français.

Résumés partiels :
{partials}
"""

CORPUS_TEMPLATE = """Voici les résumés de plusieurs documents d'un même corpus.
Rédige une synthèse globale en Markdown avec ces rubriques :

**Vue d'ensemble** : de quoi parle le corpus (2-3 phrases).
**Thèmes communs** : puces, en citant les documents concernés entre crochets, ex. [nom du document].
**Différences et points spécifiques** : puces, en citant les documents.
**Contradictions éventuelles** : ou « Aucune contradiction relevée ».

N'invente rien. Écris en français.

{summaries}
"""


def _make_batches(chunks, max_words):
    """Regroupe les passages (triés par page) en lots d'au plus `max_words` mots."""
    text_chunks = [c for c in chunks if c.get("type", "text") == "text"] or chunks
    text_chunks = sorted(text_chunks, key=lambda c: (c["page"], c.get("id", 0)))
    batches, current, count = [], [], 0
    for c in text_chunks:
        n = len(c["content"].split())
        if current and count + n > max_words:
            batches.append(current)
            current, count = [], 0
        current.append(c)
        count += n
    if current:
        batches.append(current)
    return batches


def _sample_evenly(items, k):
    """Garde k éléments répartis sur tout le document (début, milieu, fin)."""
    if len(items) <= k:
        return items
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


def summarize_document(doc_name, chunks, config):
    s_cfg = config.get("summary", {})
    max_words = s_cfg.get("max_words_per_batch", 2500)
    max_batches = s_cfg.get("max_batches", 6)

    batches = _sample_evenly(_make_batches(chunks, max_words), max_batches)
    partials = []
    for batch in batches:
        text = "\n\n".join(f"({location_label(c)}) {c['content']}" for c in batch)
        partials.append(run_chain(MAP_TEMPLATE, {"doc": doc_name, "text": text}, config))

    if len(partials) == 1:
        return run_chain(REDUCE_TEMPLATE, {"doc": doc_name, "partials": partials[0]}, config)
    return run_chain(REDUCE_TEMPLATE, {
        "doc": doc_name,
        "partials": "\n\n---\n\n".join(partials),
    }, config)


def get_or_create_summary(doc_name, chunks, config, cache_path=None, force=False):
    """Retourne le résumé d'un document, depuis le cache si possible."""
    if cache_path and os.path.exists(cache_path) and not force:
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    summary = summarize_document(doc_name, chunks, config)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(summary)
    return summary


def summarize_corpus(summaries_by_doc, config):
    """Synthèse comparative de plusieurs documents à partir de leurs résumés."""
    if len(summaries_by_doc) == 1:
        return next(iter(summaries_by_doc.values()))
    text = "\n\n".join(f"### Document : {doc}\n{s}" for doc, s in summaries_by_doc.items())
    return run_chain(CORPUS_TEMPLATE, {"summaries": text}, config)
