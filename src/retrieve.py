"""
Etape 5 - Retrieval HYBRIDE + recherche déterministe par nom propre

Trois mécanismes complémentaires, utilisés dans cet ordre de priorité :

1. RECHERCHE TEXTE HYBRIDE (sémantique + BM25) : cf. TP3-RAG Partie 3 du cours.
   Fonctionne sur un CORPUS de plusieurs documents (index fusionnés).

2. RECHERCHE IMAGE DÉTERMINISTE PAR NOM PROPRE (nouveau, PRIORITAIRE) :
   si la question contient un nom propre (ex: "Pierre Bernard") qui apparaît
   textuellement dans le document, on retourne DIRECTEMENT l'image présente
   sur cette page précise. Aucune similarité visuelle floue n'intervient :
   c'est une correspondance texte exacte, donc fiable à 100%. C'est
   indispensable car un avatar générique (icône silhouette colorée) n'a
   aucun lien visuel réel avec un nom — CLIP ne peut pas deviner ce lien de
   façon fiable, seule une correspondance textuelle exacte le peut.

3. RECHERCHE IMAGE SÉMANTIQUE (CLIP) : utilisée uniquement en secours,
   quand aucun nom propre n'est détecté dans la question (ex: "montre-moi
   un papillon avec des ocelles" — là, CLIP est pertinent car les vraies
   photos ont un contenu visuel réellement identifiable).
"""
import json
import os
import re
import unicodedata
import numpy as np
from src.ingest import load_config
from src.embed_index import encode_texts, get_embedding_model, read_index_info

# Mots vides français/anglais retirés pour le BM25 (sinon "le", "de", "the"...
# dominent le score lexical sans apporter d'information).
STOPWORDS = set("""
le la les un une des du de d l au aux et ou en dans sur pour par avec sans ce
cet cette ces qui que quoi dont est sont etre a ont il elle ils elles on nous
vous je tu me te se y ne pas plus quel quelle quels quelles comment pourquoi
quand the of and or to in on for with is are be what which how why when
""".split())


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def tokenize(text):
    """Tokenisation pour BM25 : minuscules, accents retirés, mots vides retirés.
    (Correction : l'ancienne version [a-z0-9]+ coupait les mots accentués,
    ex. "été" devenait "t", ce qui faussait la recherche sur des textes français.)"""
    tokens = re.findall(r"\w+", strip_accents(text.lower()))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def location_label(meta):
    """'page 12' pour un PDF, 'section 3' pour les autres sources."""
    unit = "page" if meta.get("source_type", "pdf") == "pdf" else "section"
    return f"{unit} {meta['page']}"


# Détecte les séquences de mots commençant par une majuscule (noms propres
# type "Pierre Bernard", "Monique Thomas"). Fonctionne pour le français avec
# accents sur les majuscules.
NAME_PATTERN = re.compile(
    r"\b[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][a-zà-ÿ'\-]+(?:\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][a-zà-ÿ'\-]+)+\b"
)


SINGLE_WORD_PATTERN = re.compile(r"\b[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][a-zà-ÿ'\-]+\b")


def extract_name_candidates(query):
    return NAME_PATTERN.findall(query)


def extract_single_word_candidates(query):
    return SINGLE_WORD_PATTERN.findall(query)


class Retriever:
    """Moteur de recherche sur UN ou PLUSIEURS documents (corpus).

    `index_dirs` : liste des dossiers d'index à interroger ensemble. Les
    vecteurs de chaque document sont fusionnés dans un seul index FAISS en
    mémoire, ce qui permet d'interroger tout le corpus d'un coup et de
    comparer équitablement les passages de documents différents.
    """

    def __init__(self, config, index_dirs=None):
        import faiss
        from rank_bm25 import BM25Okapi

        self.config = config
        index_dirs = index_dirs or [config["paths"]["index_dir"]]
        self.text_model_name = config["embeddings"]["text_model"]

        # --- Fusion des index TEXTE ---
        vectors, self.text_meta = [], []
        for d in index_dirs:
            info = read_index_info(d)
            if info.get("text_model") != self.text_model_name:
                raise ValueError(
                    f"L'index {d} a été construit avec {info.get('text_model')} "
                    f"mais la config utilise {self.text_model_name} : réindexer ce document."
                )
            idx = faiss.read_index(os.path.join(d, "text.index"))
            with open(os.path.join(d, "text_meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            if idx.ntotal:
                vectors.append(idx.reconstruct_n(0, idx.ntotal))
            self.text_meta.extend(meta)
        for m in self.text_meta:
            m.setdefault("source_type", "pdf")
            m.setdefault("source", m["doc"])

        dim = vectors[0].shape[1] if vectors else 384
        self.text_index = faiss.IndexFlatIP(dim)
        if vectors:
            self.text_index.add(np.vstack(vectors).astype("float32"))

        corpus_tokens = [tokenize(c["content"]) for c in self.text_meta]
        self.bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
        self.doc_names = sorted({c["doc"] for c in self.text_meta})

        # Registre des noms propres complets présents dans le corpus
        # (ex: "Monique Thomas") et association mot -> nom(s) complet(s).
        self.known_full_names = set()
        self.name_token_map = {}
        for c in self.text_meta:
            for full_name in NAME_PATTERN.findall(c["content"]):
                self.known_full_names.add(full_name)
                for word in full_name.split():
                    self.name_token_map.setdefault(word.lower(), set()).add(full_name)

        # --- Fusion des index IMAGE ---
        self.image_index = None
        self.image_meta = []
        img_vectors = []
        for d in index_dirs:
            p = os.path.join(d, "image.index")
            if os.path.exists(p):
                idx = faiss.read_index(p)
                with open(os.path.join(d, "image_meta.json"), "r", encoding="utf-8") as f:
                    self.image_meta.extend(json.load(f))
                if idx.ntotal:
                    img_vectors.append(idx.reconstruct_n(0, idx.ntotal))
        self.clip_model = None
        if img_vectors:
            self.clip_model = get_embedding_model(config["embeddings"]["clip_model"])
            all_img = np.vstack(img_vectors).astype("float32")
            self.image_index = faiss.IndexFlatIP(all_img.shape[1])
            self.image_index.add(all_img)

    # -----------------------------------------------------------------
    # Recherche texte hybride (sémantique + BM25)
    # -----------------------------------------------------------------
    def retrieve_text(self, query, k=None, candidate_k=20, alpha=None, doc_filter=None):
        """Recherche hybride : score = alpha * sémantique + (1 - alpha) * BM25.
        `doc_filter` : liste de noms de documents pour restreindre la recherche."""
        k = k or self.config["retrieval"]["top_k_text"]
        alpha = self.config["retrieval"].get("alpha", 0.5) if alpha is None else alpha
        n_docs = len(self.text_meta)
        if n_docs == 0:
            return []
        if doc_filter:
            candidate_k = n_docs  # on doit pouvoir trouver des passages du document ciblé

        q_emb = encode_texts(self.text_model_name, [query], is_query=True)
        sem_k = min(max(candidate_k, k), n_docs)
        sem_scores, sem_idxs = self.text_index.search(q_emb, sem_k)
        sem_score_map = {}
        for score, idx in zip(sem_scores[0], sem_idxs[0]):
            if idx != -1:
                sem_score_map[int(idx)] = float(score)

        bm25_score_map = {}
        if self.bm25 is not None:
            bm25_scores_all = self.bm25.get_scores(tokenize(query))
            top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:max(candidate_k, k)]
            for idx in top_bm25_idx:
                bm25_score_map[int(idx)] = float(bm25_scores_all[idx])

        candidates = set(sem_score_map.keys()) | set(bm25_score_map.keys())
        max_sem = max(sem_score_map.values(), default=0.0) or 1.0
        max_bm25 = max(bm25_score_map.values(), default=0.0) or 1.0

        results = []
        for idx in candidates:
            meta = self.text_meta[idx]
            if doc_filter and meta["doc"] not in doc_filter:
                continue
            sem_norm = sem_score_map.get(idx, 0.0) / max_sem
            bm25_norm = bm25_score_map.get(idx, 0.0) / max_bm25
            combined = alpha * sem_norm + (1 - alpha) * bm25_norm
            results.append({**meta, "score": float(combined),
                            "semantic_score": sem_score_map.get(idx, 0.0),
                            "loc": location_label(meta)})

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        return results

    # -----------------------------------------------------------------
    # Recherche image DÉTERMINISTE par nom propre exact (prioritaire)
    # -----------------------------------------------------------------
    def find_pages_by_exact_name(self, query):
        """Retourne les (doc, page) où un nom propre de la question apparaît
        TEXTUELLEMENT dans le document, classées par PERTINENCE réelle par
        rapport à la question complète (pas par ordre de page arbitraire).

        Important si le nom est mentionné sur de nombreuses pages (ex: nom
        d'une organisation citée partout dans un rapport de stage) : sans ce
        classement, on risquait de choisir une page au hasard plutôt que la
        plus pertinente (ex: la page avec le logo/photo officiel plutôt
        qu'une page de texte qui mentionne juste le nom en passant)."""
        candidates = extract_name_candidates(query)

        # Si aucun nom complet (2+ mots) n'est détecté dans la question, on
        # regarde si un mot capitalisé isolé (ex: "Thomas") correspond à un
        # nom complet connu du document (ex: "Monique Thomas") -> on utilise
        # alors ce nom complet pour la recherche exacte.
        if not candidates:
            for word in extract_single_word_candidates(query):
                matched_names = self.name_token_map.get(word.lower())
                if matched_names:
                    candidates.extend(matched_names)

        if not candidates:
            return []
        candidates = sorted(set(candidates), key=len, reverse=True)

        for name in candidates:
            normalized_name = re.sub(r"\s+", " ", name.lower())
            matched_indices = [
                i for i, c in enumerate(self.text_meta)
                if normalized_name in re.sub(r"\s+", " ", c["content"].lower())
            ]
            if not matched_indices:
                continue

            if len(matched_indices) == 1:
                c = self.text_meta[matched_indices[0]]
                return [(c["doc"], c["page"])]

            # Plusieurs pages mentionnent ce nom -> on les classe par
            # similarité sémantique avec la question complète, pour
            # privilégier la page la plus probablement recherchée.
            q_emb = encode_texts(self.text_model_name, [query], is_query=True)[0]
            scores = np.array([float(self.text_index.reconstruct(int(i)) @ q_emb)
                               for i in matched_indices])

            ranked = sorted(zip(matched_indices, scores), key=lambda x: x[1], reverse=True)
            ordered_pages = []
            seen = set()
            for idx, _ in ranked:
                c = self.text_meta[idx]
                key = (c["doc"], c["page"])
                if key not in seen:
                    ordered_pages.append(key)
                    seen.add(key)
            return ordered_pages
        return []

    def get_images_by_pages(self, ordered_pages):
        """Retourne les images des pages données, dans l'ORDRE DE PERTINENCE
        fourni (pas triées par numéro de page)."""
        order_index = {p: i for i, p in enumerate(ordered_pages)}
        pages_set = set(ordered_pages)
        matches = [m for m in self.image_meta if (m["doc"], m["page"]) in pages_set]
        matches.sort(key=lambda m: order_index.get((m["doc"], m["page"]), 999999))
        return matches

    # -----------------------------------------------------------------
    # Recherche image sémantique (CLIP) — secours uniquement
    # -----------------------------------------------------------------
    def retrieve_images_semantic(self, query):
        if self.image_index is None:
            return []
        candidate_k = min(30, self.image_index.ntotal) if self.image_index.ntotal else 0
        if candidate_k == 0:
            return []
        q_emb = np.asarray(self.clip_model.encode([query], normalize_embeddings=True), dtype="float32")
        scores, idxs = self.image_index.search(q_emb, candidate_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.image_meta[idx]
            results.append({**meta, "score": float(score)})
        return results

    # -----------------------------------------------------------------
    # Point d'entrée principal
    # -----------------------------------------------------------------
    def retrieve(self, query, doc_filter=None, k=None):
        text_results = self.retrieve_text(query, k=k, doc_filter=doc_filter)
        top_k_images = self.config["retrieval"]["top_k_images"]

        # PRIORITÉ 1 : nom propre détecté et trouvé textuellement -> réponse
        # déterministe, classée par pertinence réelle, aucune ambiguïté.
        exact_pages_ordered = self.find_pages_by_exact_name(query)
        if exact_pages_ordered:
            image_results = self.get_images_by_pages(exact_pages_ordered)
            if doc_filter:
                image_results = [m for m in image_results if m["doc"] in doc_filter]
            image_results = image_results[:top_k_images]
            return text_results, image_results

        # PRIORITÉ 2 (secours) : pas de nom propre -> recherche visuelle CLIP.
        # Important : CLIP renvoie TOUJOURS "les plus proches" candidats,
        # même si aucun n'est réellement pertinent (ex: un montant en euros
        # n'a aucun rapport avec une photo). On filtre donc par un score de
        # confiance minimum AVANT tout boost, pour ne rien afficher plutôt
        # que d'afficher des images sans rapport.
        anchor_text_results = self.retrieve_text(query, k=15, doc_filter=doc_filter)
        image_candidates = self.retrieve_images_semantic(query)

        min_score = self.config["retrieval"].get("min_image_score", 0.18)
        image_candidates = [img for img in image_candidates if img["score"] >= min_score
                            and (not doc_filter or img["doc"] in doc_filter)]

        top_pages_by_doc = {}
        for r in anchor_text_results:
            top_pages_by_doc.setdefault(r["doc"], []).append(r["page"])

        for img in image_candidates:
            same_doc_pages = top_pages_by_doc.get(img["doc"], [])
            if same_doc_pages:
                min_distance = min(abs(img["page"] - p) for p in same_doc_pages)
                boost = max(0.0, 0.3 - 0.1 * min_distance)
            else:
                boost = 0.0
            img["score"] += boost

        image_results = sorted(image_candidates, key=lambda x: x["score"], reverse=True)[:top_k_images]
        return text_results, image_results


if __name__ == "__main__":
    cfg = load_config()
    r = Retriever(cfg)
    t, i = r.retrieve("Donne-moi l'image de Pierre Bernard")
    print(json.dumps(i, ensure_ascii=False, indent=2))