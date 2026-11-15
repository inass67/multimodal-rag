"""
Etape 5 - Retrieval HYBRIDE + recherche déterministe par nom propre

Trois mécanismes complémentaires, utilisés dans cet ordre de priorité :

1. RECHERCHE TEXTE HYBRIDE (sémantique + BM25) : cf. TP3-RAG Partie 3 du cours.

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
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.ingest import load_config


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


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
    def __init__(self, config):
        self.config = config
        index_dir = config["paths"]["index_dir"]

        self.text_model = SentenceTransformer(config["embeddings"]["text_model"])
        self.text_index = faiss.read_index(os.path.join(index_dir, "text.index"))
        with open(os.path.join(index_dir, "text_meta.json"), "r", encoding="utf-8") as f:
            self.text_meta = json.load(f)

        corpus_tokens = [tokenize(c["content"]) for c in self.text_meta]
        self.bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

        # Registre des noms complets réellement présents dans le document
        # (ex: "Monique Thomas"), et association de chaque mot du nom
        # ("monique", "thomas") vers le(s) nom(s) complet(s) correspondant(s).
        # Permet de reconnaître une question tapée avec un seul mot du nom
        # (ex: juste "Thomas") et de la relier au bon nom complet.
        self.known_full_names = set()
        self.name_token_map = {}
        for c in self.text_meta:
            for full_name in NAME_PATTERN.findall(c["content"]):
                self.known_full_names.add(full_name)
                for word in full_name.split():
                    self.name_token_map.setdefault(word.lower(), set()).add(full_name)

        self.image_index = None
        self.image_meta = []
        self.clip_model = None
        img_index_path = os.path.join(index_dir, "image.index")
        if os.path.exists(img_index_path):
            self.clip_model = SentenceTransformer(config["embeddings"]["clip_model"])
            self.image_index = faiss.read_index(img_index_path)
            with open(os.path.join(index_dir, "image_meta.json"), "r", encoding="utf-8") as f:
                self.image_meta = json.load(f)

    # -----------------------------------------------------------------
    # Recherche texte hybride (sémantique + BM25)
    # -----------------------------------------------------------------
    def retrieve_text(self, query, k=None, candidate_k=20, alpha=0.5):
        k = k or self.config["retrieval"]["top_k_text"]
        n_docs = len(self.text_meta)
        if n_docs == 0:
            return []

        q_emb = self.text_model.encode([query], normalize_embeddings=True).astype("float32")
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
            sem_norm = sem_score_map.get(idx, 0.0) / max_sem
            bm25_norm = bm25_score_map.get(idx, 0.0) / max_bm25
            combined = alpha * sem_norm + (1 - alpha) * bm25_norm
            meta = self.text_meta[idx]
            results.append({**meta, "score": float(combined)})

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
            q_emb = self.text_model.encode([query], normalize_embeddings=True)[0]
            contents = [self.text_meta[i]["content"] for i in matched_indices]
            chunk_embs = self.text_model.encode(contents, normalize_embeddings=True)
            scores = chunk_embs @ q_emb

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
        q_emb = self.clip_model.encode([query], normalize_embeddings=True).astype("float32")
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
    def retrieve(self, query):
        text_results = self.retrieve_text(query)
        top_k_images = self.config["retrieval"]["top_k_images"]

        # PRIORITÉ 1 : nom propre détecté et trouvé textuellement -> réponse
        # déterministe, classée par pertinence réelle, aucune ambiguïté.
        exact_pages_ordered = self.find_pages_by_exact_name(query)
        if exact_pages_ordered:
            image_results = self.get_images_by_pages(exact_pages_ordered)[:top_k_images]
            return text_results, image_results

        # PRIORITÉ 2 (secours) : pas de nom propre -> recherche visuelle CLIP.
        # Important : CLIP renvoie TOUJOURS "les plus proches" candidats,
        # même si aucun n'est réellement pertinent (ex: un montant en euros
        # n'a aucun rapport avec une photo). On filtre donc par un score de
        # confiance minimum AVANT tout boost, pour ne rien afficher plutôt
        # que d'afficher des images sans rapport.
        anchor_text_results = self.retrieve_text(query, k=15)
        image_candidates = self.retrieve_images_semantic(query)

        min_score = self.config["retrieval"].get("min_image_score", 0.18)
        image_candidates = [img for img in image_candidates if img["score"] >= min_score]

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