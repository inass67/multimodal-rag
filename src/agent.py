"""
Agent de recherche documentaire.

Le système n'est pas un simple pipeline figé : un AGENT décide, pour chaque
demande de l'utilisateur, quel OUTIL utiliser, l'exécute, vérifie le résultat
et se corrige si nécessaire.

Boucle de l'agent :
    1. PLANIFIER : le LLM lit la demande + la liste des outils + la liste des
       documents, et choisit un outil avec ses arguments (réponse JSON).
       En cas d'échec du LLM, un routeur à base de règles prend le relais.
    2. AGIR      : l'outil choisi est exécuté (recherche, résumé, comparaison...).
    3. VÉRIFIER  : pour une recherche, l'agent contrôle les citations :
         - réponse sans aucune citation ou avec un numéro de source inventé
           -> régénération avec une consigne plus stricte ;
         - « Je ne sais pas » -> nouvelle recherche élargie (plus de passages).
    4. RÉPONDRE  : réponse finale + références + trace des étapes (affichée
       dans l'interface pour la transparence).

Les outils sont aussi exportables au format LangChain (StructuredTool) via
`as_langchain_tools()`, pour être branchés sur un agent LangChain.
"""
import json
import re
from dataclasses import dataclass
from typing import Callable

from src.llm import run_chain, LLMError
from src.generate import answer_question, is_no_answer
from src.retrieve import strip_accents
from src.summarize import get_or_create_summary, summarize_corpus


@dataclass
class Tool:
    name: str
    description: str
    func: Callable


PLANNER_TEMPLATE = """Tu es le planificateur d'un agent de recherche documentaire.
Choisis l'outil le plus adapté à la demande de l'utilisateur.

Outils disponibles :
{tools}

Documents du corpus :
{docs}

Demande de l'utilisateur : {query}

Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, de la forme :
{{"outil": "<nom de l'outil>", "arguments": {{"question": "...", "document": "..."}}, "raison": "<une phrase>"}}
Les arguments inutiles pour l'outil choisi peuvent être omis.
Pour "document", recopie exactement un nom de la liste des documents.
"""


GENERIC_WORDS = {"document", "documents", "pdf", "docx", "txt", "html", "fichier", "fichiers",
                 "resume", "resumer", "resumes", "les", "des", "the", "sur", "test", "page", "article"}


def _norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", strip_accents(text.lower()))


def _keywords(text):
    return set(w for w in _norm(text).split() if len(w) > 2) - GENERIC_WORDS


def match_document(name, doc_names):
    """Retrouve le document le plus proche d'un nom approximatif (ex: 'le guide
    des papillons' -> 'Guide-papillons.pdf')."""
    if not name:
        return None
    if name in doc_names:
        return name
    q = _keywords(name)
    best, best_score = None, 0
    for d in doc_names:
        score = len(q & _keywords(d))
        if score > best_score:
            best, best_score = d, score
    return best


class ResearchAgent:
    def __init__(self, retriever, config, summary_path_fn=None):
        self.retriever = retriever
        self.config = config
        self.summary_path_fn = summary_path_fn or (lambda doc: None)
        self.doc_names = retriever.doc_names
        self.trace = []
        self.tools = {t.name: t for t in [
            Tool("rechercher",
                 "Répond à une question précise à partir des documents, avec citations [n]. "
                 "Arguments : question, document (optionnel, pour limiter à un document).",
                 self.tool_search),
            Tool("resumer_document",
                 "Résume un document entier du corpus. Argument : document.",
                 self.tool_summarize_document),
            Tool("resumer_corpus",
                 "Résume et synthétise TOUS les documents du corpus (thèmes communs, différences).",
                 self.tool_summarize_corpus),
            Tool("comparer_documents",
                 "Compare ce que disent les différents documents sur un sujet. Argument : question.",
                 self.tool_compare),
            Tool("lister_documents",
                 "Liste les documents disponibles dans le corpus.",
                 self.tool_list),
        ]}

    # ------------------------------------------------------------------
    # Journal des étapes (affiché dans l'interface)
    # ------------------------------------------------------------------
    def log(self, step, detail):
        self.trace.append({"etape": step, "detail": detail})

    # ------------------------------------------------------------------
    # 1. PLANIFICATION
    # ------------------------------------------------------------------
    def plan(self, query):
        mode = self.config.get("agent", {}).get("planner", "llm")
        if mode == "llm":
            try:
                tools_desc = "\n".join(f"- {t.name} : {t.description}" for t in self.tools.values())
                raw = run_chain(PLANNER_TEMPLATE, {
                    "tools": tools_desc,
                    "docs": "\n".join(f"- {d}" for d in self.doc_names),
                    "query": query,
                }, self.config, temperature=0.0)
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                decision = json.loads(m.group(0)) if m else {}
                if decision.get("outil") in self.tools:
                    decision.setdefault("arguments", {})
                    decision["arguments"].setdefault("question", query)
                    decision["planificateur"] = "LLM"
                    return decision
            except (LLMError, ValueError, json.JSONDecodeError) as e:
                self.log("Planification", f"LLM indisponible ({e}), routeur à règles utilisé")
        return self.rule_based_plan(query)

    def rule_based_plan(self, query):
        q = _norm(query)
        doc = match_document(query, self.doc_names)
        if re.search(r"\b(quels? documents?|liste|lister|corpus contient)\b", q):
            tool = "lister_documents"
        elif re.search(r"\b(compar\w*|difference\w*|versus|vs)\b", q):
            tool = "comparer_documents"
        elif re.search(r"\b(resum\w*|synthe\w*|summar\w*)\b", q):
            tool = "resumer_document" if doc else "resumer_corpus"
        else:
            tool = "rechercher"
        return {"outil": tool, "arguments": {"question": query, "document": doc},
                "raison": "Choix par mots-clés (routeur à règles).", "planificateur": "règles"}

    # ------------------------------------------------------------------
    # 2. OUTILS
    # ------------------------------------------------------------------
    def tool_search(self, question, document=None, **_):
        doc_filter = None
        if document:
            matched = match_document(document, self.doc_names)
            if matched:
                doc_filter = [matched]
                self.log("Filtre", f"Recherche limitée au document « {matched} »")

        k = self.config["retrieval"]["top_k_text"]
        text_results, image_results = self.retriever.retrieve(question, doc_filter=doc_filter, k=k)
        self.log("Recherche", f"{len(text_results)} passages retrouvés (hybride sémantique + BM25)")
        result = answer_question(question, text_results, self.config)

        # 3. VÉRIFICATION et auto-correction (au plus `max_retries` fois)
        max_retries = self.config.get("agent", {}).get("max_retries", 1)
        for _ in range(max_retries):
            if result["no_answer"]:
                k = k * 2
                text_results, image_results = self.retriever.retrieve(question, doc_filter=doc_filter, k=k)
                self.log("Vérification", f"Pas de réponse : nouvelle recherche élargie à {k} passages")
                result = answer_question(question, text_results, self.config)
            elif not result["cited"] or result["invalid"]:
                problem = ("aucune citation" if not result["cited"]
                           else f"citations inexistantes {result['invalid']}")
                self.log("Vérification", f"Réponse rejetée ({problem}) : régénération stricte")
                result = answer_question(question, text_results, self.config, extra_instruction=(
                    "ATTENTION : ta réponse précédente était mal sourcée. Chaque phrase DOIT se "
                    f"terminer par une citation valide entre [1] et [{len(text_results)}]."))
            else:
                break

        self.log("Vérification", (
            f"Réponse finale : {len(result['cited'])} source(s) citée(s)"
            + (f", citations invalides retirées : {result['invalid']}" if result["invalid"] else "")
        ))
        return {"type": "reponse", "result": result,
                "text_results": text_results, "image_results": image_results}

    def _summary(self, doc):
        chunks = [c for c in self.retriever.text_meta if c["doc"] == doc]
        return get_or_create_summary(doc, chunks, self.config, cache_path=self.summary_path_fn(doc))

    def tool_summarize_document(self, document=None, question="", **_):
        doc = match_document(document or question, self.doc_names) or (
            self.doc_names[0] if len(self.doc_names) == 1 else None)
        if not doc:
            self.log("Résumé", "Document non identifié : synthèse du corpus à la place")
            return self.tool_summarize_corpus()
        self.log("Résumé", f"Résumé map-reduce du document « {doc} »")
        return {"type": "resume", "titre": f"Résumé — {doc}", "texte": self._summary(doc)}

    def tool_summarize_corpus(self, **_):
        summaries = {}
        for doc in self.doc_names:
            self.log("Résumé", f"Résumé du document « {doc} »")
            summaries[doc] = self._summary(doc)
        self.log("Synthèse", f"Synthèse comparative de {len(summaries)} document(s)")
        return {"type": "resume", "titre": "Synthèse du corpus",
                "texte": summarize_corpus(summaries, self.config), "details": summaries}

    def tool_compare(self, question, **_):
        per_doc = max(2, self.config["retrieval"]["top_k_text"] // max(len(self.doc_names), 1))
        text_results = []
        for doc in self.doc_names:
            res = self.retriever.retrieve_text(question, k=per_doc, doc_filter=[doc])
            text_results.extend(res)
        self.log("Comparaison", f"{len(text_results)} passages retrouvés dans {len(self.doc_names)} documents")
        result = answer_question(question, text_results, self.config, extra_instruction=(
            "Organise la réponse document par document, puis termine par les points communs "
            "et les différences entre documents."))
        return {"type": "reponse", "result": result, "text_results": text_results, "image_results": []}

    def tool_list(self, **_):
        counts = {}
        for c in self.retriever.text_meta:
            counts[c["doc"]] = counts.get(c["doc"], 0) + 1
        lines = [f"- **{d}** — {n} passages indexés" for d, n in counts.items()]
        return {"type": "resume", "titre": "Documents du corpus", "texte": "\n".join(lines)}

    # ------------------------------------------------------------------
    # Point d'entrée
    # ------------------------------------------------------------------
    def run(self, query):
        self.trace = []
        decision = self.plan(query)
        tool = self.tools[decision["outil"]]
        args = {k: v for k, v in decision.get("arguments", {}).items() if v}
        args.setdefault("question", query)
        self.log("Planification", f"Outil choisi : {tool.name} ({decision.get('planificateur')}) — "
                                  f"{decision.get('raison', '')}")
        output = tool.func(**args)
        output["outil"] = tool.name
        output["trace"] = list(self.trace)
        return output

    def as_langchain_tools(self):
        """Exporte les outils au format LangChain (pour un agent LangChain)."""
        from langchain_core.tools import StructuredTool
        return [StructuredTool.from_function(func=t.func, name=t.name, description=t.description)
                for t in self.tools.values()]
