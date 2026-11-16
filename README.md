# Agent de recherche documentaire (RAG multimodal)

**Projet n°1 : Agent de recherche documentaire** (catégorie Connaissance et RAG).

> **Objectif du sujet :** permettre à un utilisateur d'interroger en langage naturel un corpus
> de documents hétérogènes et d'obtenir des réponses sourcées.

---

## 1. Couverture du cahier des charges

| Exigence du sujet | Où c'est fait dans le projet |
|---|---|
| **Cherche dans des PDF, articles et bases internes** | `src/loaders.py`, `src/ingest.py`, `src/corpus.py` : PDF (texte + images + OCR), Word `.docx`, `.txt`/`.md`, pages internes `.html`, **articles web par URL**. Plusieurs documents sont interrogés **ensemble** (corpus). |
| **Résume les sources** | `src/summarize.py` : résumé *map-reduce* d'un document, synthèse comparative du corpus, et résumé de chaque source citée dans chaque réponse (`src/generate.py`). |
| **Répond avec références** | `src/generate.py` : sources numérotées `[1]`, `[2]`… citées après chaque affirmation, liste des références (document + page/section + lien), citations inventées détectées et retirées. |
| **LLM** | Gemini (par défaut) ou OpenAI, `src/llm.py`, avec modèles de secours automatiques. |
| **RAG** | Découpage → embeddings → base vectorielle → recherche hybride → génération ancrée. |
| **LangChain** | Chaînes LCEL `PromptTemplate \| LLM \| StrOutputParser` (`src/llm.py`), `RecursiveCharacterTextSplitter` (`src/chunk_compare.py`), chargeurs `WebBaseLoader` / `Docx2txtLoader` (`src/loaders.py`), outils de l'agent exportables en `StructuredTool` (`src/agent.py`). |
| **Vector Database** | FAISS (`IndexFlatIP`, similarité cosinus), un index texte + un index image (CLIP), fusionnés en mémoire pour interroger tout le corpus. |
| **Agent** | `src/agent.py` : le LLM **choisit l'outil** (rechercher, résumer un document, résumer le corpus, comparer, lister), **vérifie** les citations et **se corrige** (régénération, recherche élargie). Les étapes sont affichées dans l'interface. |
| **Point clé : qualité du découpage** | `src/chunk_compare.py` : 5 méthodes comparées automatiquement pour chaque document (taille fixe courte/longue, par phrases, récursif maison, LangChain), la meilleure est retenue. |
| **Point clé : choix de l'embedding** | `src/compare_embeddings.py` : 3 modèles comparés (Recall@k, MRR, coût) ; modèle multilingue `multilingual-e5-small` retenu. |
| **Point clé : évaluation de la pertinence des citations** | `src/evaluate.py` : jeu de test, Recall@k / Precision@k / MRR (sémantique vs BM25 vs hybride), validité, précision, taux de succès et support sémantique des citations, juge LLM en option. |
| **Limiter les hallucinations** | Réponse uniquement à partir des sources, « Je ne sais pas » si l'information est absente, contrôle des citations. |

---

## 2. Architecture

```
 Sources hétérogènes : PDF · DOCX · TXT/MD · HTML interne · article web (URL)
        │
        ▼
 [1] ingest.py / loaders.py  → texte par page (PDF) ou par section (autres) + images
        │                       + provenance : document, page/section, type, URL
        ▼
 [2] chunk_compare.py        → 5 méthodes de découpage comparées, la meilleure est retenue
        │
        ▼
 [3] ocr_utils.py            → OCR (Tesseract) du texte contenu dans les images
        │
        ▼
 [4] embed_index.py          → embeddings multilingual-e5-small → FAISS (texte)
        │                       + CLIP → FAISS (images)
        ▼
 [5] retrieve.py             → recherche HYBRIDE (sémantique + BM25) sur tout le corpus
        │                       + images (nom propre exact puis CLIP)
        ▼
 [6] agent.py                → PLANIFIER (choix de l'outil) → AGIR → VÉRIFIER → RÉPONDRE
        │        ├─ generate.py   réponse avec citations [n] + résumé des sources
        │        └─ summarize.py  résumé map-reduce d'un document / du corpus
        ▼
 app.py (Streamlit)          → Questions · Résumés · Évaluation  (+ voix, comptes utilisateurs)

 evaluate.py / compare_embeddings.py → rapports dans eval/
```

| Fichier | Rôle |
|---|---|
| `app.py` | Interface : gestion du corpus, questions, résumés, évaluation |
| `src/corpus.py` | Cache des documents, sélection du corpus, pipeline de traitement |
| `src/loaders.py` | Chargeurs PDF / DOCX / TXT / MD / HTML / URL |
| `src/ingest.py` | Extraction texte + images avec provenance |
| `src/chunk_compare.py` | Comparaison et choix automatique du découpage |
| `src/chunk_text.py` | Découpage simple (ligne de commande) |
| `src/ocr_utils.py` | OCR des images |
| `src/embed_index.py` | Embeddings et index FAISS |
| `src/retrieve.py` | Recherche hybride multi-documents |
| `src/llm.py` | Appels LLM + chaînes LangChain |
| `src/generate.py` | Réponse sourcée et analyse des citations |
| `src/summarize.py` | Résumés map-reduce |
| `src/agent.py` | Agent : planification, outils, vérification |
| `src/evaluate.py` | Jeu de test et métriques |
| `src/compare_embeddings.py` | Comparaison des modèles d'embedding |
| `src/auth.py` | Comptes utilisateurs (mots de passe salés et hachés SHA-256) |

---

## 3. Installation

### Tesseract OCR (pour lire le texte dans les images)
- Windows : installeur sur https://github.com/UB-Mannheim/tesseract/wiki, puis ajouter au PATH.
- Mac : `brew install tesseract tesseract-lang`
- Linux : `sudo apt-get install tesseract-ocr tesseract-ocr-fra`

Sans Tesseract, l'application fonctionne quand même : seule l'étape d'OCR est sautée.

### Dépendances Python
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate      Mac/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

### Clé API du LLM
Créer une clé sur https://aistudio.google.com/apikey puis :
```bash
# Windows (PowerShell)
$env:GOOGLE_API_KEY="ta_cle_ici"
# Mac/Linux
export GOOGLE_API_KEY="ta_cle_ici"
```
Pour OpenAI : `provider: "openai"`, `model` et `api_key_env: "OPENAI_API_KEY"` dans `config.yaml`.

---

## 4. Utilisation

```bash
streamlit run app.py
```

1. **Créer un compte**, puis se connecter.
2. **Barre latérale** : ajouter des fichiers (PDF, Word, texte, HTML) ou coller l'URL d'un article,
   puis cocher les documents à interroger ensemble.
3. **Onglet « Poser une question »** (au clavier ou à la voix). Exemples :
   - « Quelles sont les conditions pour demander un congé ? » → recherche avec citations
   - « Résume le guide des papillons » → résumé du document
   - « Compare ce que disent les documents sur la sécurité » → comparaison
   - « Quels documents sont disponibles ? » → liste du corpus
   La réponse affiche les **références**, le **résumé des sources utilisées**, les **images**,
   tous les passages retrouvés (cités ou non) et les **étapes suivies par l'agent**.
4. **Onglet « Résumer les sources »** : résumé d'un document ou synthèse de tout le corpus
   (téléchargeable en `.md`).
5. **Onglet « Évaluation »** : voir section 5.

> Les documents déjà traités avec l'ancien modèle d'embedding sont **réindexés automatiquement**
> au premier chargement (quelques secondes).

---

## 5. Évaluation (à montrer à la soutenance)

À lancer depuis l'onglet **Évaluation** ou en ligne de commande, sur les documents sélectionnés :

```bash
python -m src.evaluate --generate --n 15     # crée le jeu de test puis évalue
python -m src.compare_embeddings             # compare les modèles d'embedding
python -m src.evaluate --judge               # ré-évalue avec le juge LLM en plus
```

Résultats dans `eval/` : `evaluation_report.md`, `embeddings_report.md` (+ versions JSON).

**Jeu de test :** le LLM écrit une question à partir de passages tirés au hasard (répartis entre
les documents). La vérité terrain est la page d'où vient le passage. Le fichier
`eval/eval_set.json` peut être relu et corrigé à la main.

| Métrique | Ce qu'elle mesure |
|---|---|
| Recall@k | La bonne page est-elle parmi les k premiers passages retrouvés ? |
| Precision@k | Part des k passages qui viennent de la bonne page |
| MRR | 1 / rang du premier bon passage (1 = toujours en tête) |
| Taux de réponse | Questions auxquelles l'agent répond (pas « Je ne sais pas ») |
| Validité des citations | Part des numéros `[n]` qui correspondent à une vraie source |
| Précision des citations | Part des sources citées qui viennent de la bonne page |
| Taux de succès des citations | Part des réponses qui citent au moins la bonne source |
| Support sémantique | Part des phrases citées réellement proches de leur source (similarité ≥ 0,5) |
| Juge LLM (option) | Le LLM confirme que la source soutient la phrase |

La recherche est évaluée dans 3 configurations (sémantique seule, BM25 seul, hybride) pour
**justifier le choix de la recherche hybride**.

**Économie du quota LLM :** les métriques de recherche et la comparaison des embeddings ne font
**aucun** appel au LLM. Seules la génération du jeu de test (1 appel pour 5 questions) et
l'évaluation des citations (1 appel par question) en font.

---

## 6. Choix techniques à justifier à l'oral

- **Découpage :** il n'existe pas de taille idéale universelle. Le système compare 5 méthodes sur
  chaque nouveau document avec des métriques sans vérité terrain : cohérence interne d'un passage
  (similarité entre ses deux moitiés, à maximiser) et redondance entre passages voisins (à minimiser),
  avec une pénalité pour les passages trop courts ou irréguliers. La meilleure est retenue automatiquement.
- **Embedding :** `all-MiniLM-L6-v2` (ancien choix) est entraîné surtout sur de l'anglais et
  `paraphrase-multilingual-MiniLM-L12-v2` tronque les passages à 128 tokens. `multilingual-e5-small`
  est multilingue, accepte 512 tokens et reste léger (384 dimensions, fonctionne sur CPU, gratuit).
  Le choix est vérifié par `compare_embeddings.py`.
- **Recherche hybride :** la recherche sémantique comprend les reformulations, BM25 retrouve les
  termes exacts (noms propres, codes, chiffres). Les scores normalisés sont combinés
  (`alpha = 0.5`). Correction apportée : la tokenisation BM25 gérait mal les accents
  (« été » devenait « t ») ; les accents et les mots vides sont désormais traités.
- **Base vectorielle FAISS :** rapide, locale, sans serveur. `IndexFlatIP` sur vecteurs normalisés
  = similarité cosinus exacte. Les index des documents sélectionnés sont fusionnés en mémoire.
- **Citations :** sources numérotées, consigne stricte de citer après chaque affirmation, numéros
  inventés détectés puis retirés, régénération si aucune citation. C'est la réponse au risque
  d'hallucination.
- **Agent :** boucle planifier → agir → vérifier → répondre. Le planificateur LLM répond en JSON ;
  un routeur par mots-clés prend le relais si le LLM est indisponible (robustesse).
- **Multimodal :** images indexées avec CLIP, OCR des images, recherche d'image exacte par nom propre.
- **Généricité :** aucun mot-clé métier dans le code ; changer de domaine = changer de documents.

## 7. Limites et perspectives

- Le jeu de test synthétique favorise les questions proches du texte : un jeu annoté à la main
  donnerait une mesure plus exigeante.
- La précision des citations ne compte comme correcte que la page d'origine de la question ; une
  autre page contenant la même information est comptée fausse (mesure prudente). Le support
  sémantique et le juge LLM compensent ce biais.
- Perspectives : base vectorielle persistante avec filtres (Chroma, Qdrant), reranking par
  cross-encoder, mémoire de conversation, Graph RAG.
