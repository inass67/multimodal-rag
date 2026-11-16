"""
Étape 6 - Génération d'une réponse SOURCÉE (avec références numérotées).

1. Les passages retrouvés sont numérotés [1], [2], ... avec leur document et
   leur page/section.
2. Une chaîne LangChain (prompt | LLM | parser) demande au LLM de répondre
   UNIQUEMENT à partir de ces passages et de citer le numéro de la source
   après chaque affirmation.
3. La réponse est analysée : on extrait les numéros cités, on détecte les
   citations inventées (numéro inexistant) et on construit la liste des
   références réellement utilisées. Cela limite les hallucinations et permet
   d'évaluer la pertinence des citations (src/evaluate.py).
4. Le LLM produit aussi un court résumé de chaque source utilisée
   (exigence du sujet : « Résume les sources »).
"""
import re

from src.llm import run_chain
from src.retrieve import location_label

NO_ANSWER = "Je ne sais pas"

ANSWER_TEMPLATE = """Tu es un agent de recherche documentaire. Tu réponds à la question
en t'appuyant UNIQUEMENT sur les sources numérotées ci-dessous.

Règles obligatoires :
1. Chaque affirmation doit être suivie du numéro de la ou des sources qui la justifient,
   entre crochets : [1], [2] ou [1][3]. N'utilise que les numéros présents ci-dessous.
2. N'invente rien. Si les sources ne permettent pas de répondre, écris exactement :
   "{no_answer}" puis explique en une phrase ce qui manque.
3. Si plusieurs sources se contredisent, signale-le en citant chacune d'elles.
4. Si la question demande de "montrer" ou "afficher" une image et que les sources parlent
   du sujet, réponds avec les informations textuelles : les images sont affichées
   séparément par l'application.
5. Réponds dans la langue de la question, de façon claire et structurée (Markdown).

Format de sortie EXACT :
### Réponse
<ta réponse avec les citations [n]>

### Résumé des sources utilisées
- [n] <une phrase qui résume ce que cette source apporte à la réponse>

Sources :
{context}

Question : {question}
"""


def format_context(text_results):
    blocks = []
    for i, r in enumerate(text_results, start=1):
        loc = r.get("loc") or location_label(r)
        blocks.append(f"[{i}] (Document : {r['doc']}, {loc})\n{r['content']}")
    return "\n\n".join(blocks)


def extract_citations(answer, n_sources):
    """Retourne (citations_valides, citations_invalides) trouvées dans la réponse."""
    found = set()
    for group in re.findall(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", answer):
        for num in re.split(r"[,;]", group):
            found.add(int(num.strip()))
    valid = sorted(n for n in found if 1 <= n <= n_sources)
    invalid = sorted(n for n in found if not 1 <= n <= n_sources)
    return valid, invalid


def remove_citations(text, numbers):
    """Supprime les citations [n] dont le numéro est dans `numbers`."""
    bad = set(numbers)

    def fix(m):
        kept = [x.strip() for x in re.split(r"[,;]", m.group(1)) if int(x.strip()) not in bad]
        return f"[{', '.join(kept)}]" if kept else ""
    return re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", fix, text)


def split_answer_sections(answer):
    """Sépare la partie 'Réponse' de la partie 'Résumé des sources'."""
    m = re.search(r"#+\s*R[ée]sum[ée] des sources.*", answer, flags=re.IGNORECASE)
    if not m:
        return answer.replace("### Réponse", "").strip(), ""
    body = answer[:m.start()].replace("### Réponse", "").strip()
    summary = answer[m.end():].strip()
    return body, summary


def is_no_answer(text):
    return NO_ANSWER.lower() in text.lower()[:200]


def answer_question(query, text_results, config, extra_instruction=""):
    """Génère une réponse sourcée. Retourne un dictionnaire détaillé."""
    if not text_results:
        return {"answer": f"{NO_ANSWER} : aucun passage pertinent n'a été trouvé.",
                "body": f"{NO_ANSWER} : aucun passage pertinent n'a été trouvé.",
                "sources_summary": "", "cited": [], "invalid": [], "references": [],
                "no_answer": True}

    template = ANSWER_TEMPLATE
    if extra_instruction:
        template = template.replace("Sources :", extra_instruction + "\n\nSources :")
    raw = run_chain(template, {
        "no_answer": NO_ANSWER,
        "context": format_context(text_results),
        "question": query,
    }, config)

    body, sources_summary = split_answer_sections(raw)
    cited, invalid = extract_citations(body, len(text_results))
    if invalid:  # on retire les numéros de sources inventés par le LLM
        body = remove_citations(body, invalid)
        sources_summary = remove_citations(sources_summary, invalid)
    references = [{"num": n, **text_results[n - 1]} for n in cited]
    return {
        "answer": raw,
        "body": body,
        "sources_summary": sources_summary,
        "cited": cited,
        "invalid": invalid,
        "references": references,
        "no_answer": is_no_answer(body),
    }


def generate_answer(query, text_results, config):
    """Compatibilité avec l'ancienne API : renvoie seulement le texte."""
    return answer_question(query, text_results, config)["answer"]
