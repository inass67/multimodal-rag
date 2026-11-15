# RAG Multimodal Générique (Texte + Images + OCR)

Architecture pensée pour être **indépendante du domaine** : pour changer de sujet
(papillons → oiseaux → documents administratifs...), tu ne touches à **aucun fichier de code**,
tu remplaces juste le(s) PDF dans `data/raw/`.

## Vue d'ensemble du pipeline

```
PDF (data/raw)
   │
   ▼
[1] ingest.py        → texte par page + images extraites (data/processed)
   │
   ▼
[2] chunk_text.py     → chunks de texte avec source (doc, page)
   │
   ▼
[3] ocr_utils.py       → OCR sur les images → chunks supplémentaires
   │
   ▼
[4] embed_index.py    → index FAISS texte (MiniLM) + index FAISS image (CLIP)
   │
   ▼
[5] retrieve.py        → top-k texte + top-k images pour une question
   │
   ▼
[6] generate.py        → réponse LLM (Gemini / OpenAI) à partir du contexte
   │
   ▼
app.py (Streamlit)     → interface question / réponse + images sources
```

## Plan sur 3 heures (ordre à respecter)

| Temps | Étape |
|---|---|
| 0:00–0:15 | Installation des dépendances (voir ci-dessous) |
| 0:15–0:30 | Déposer le PDF (ta thèse) dans `data/raw/`, lancer `ingest.py` |
| 0:30–0:45 | Lancer `chunk_text.py` puis `ocr_utils.py`, vérifier `chunks.json` |
| 0:45–1:30 | Lancer `embed_index.py` (le plus long : téléchargement des modèles + calcul des embeddings) |
| 1:30–2:00 | Lancer `retrieve.py` en test, ajuster `top_k` si besoin |
| 2:00–2:30 | Configurer la clé API LLM, tester `generate.py`, lancer `streamlit run app.py` |
| 2:30–3:00 | Tester 3-5 questions réelles, marge de sécurité pour bugs / slides |

## 1) Installation

### Dépendance système : Tesseract OCR (obligatoire pour l'OCR)

- Windows : installe le binaire depuis https://github.com/UB-Mannheim/tesseract/wiki
  puis ajoute-le au PATH (ou renseigne `pytesseract.pytesseract.tesseract_cmd` dans `ocr_utils.py`).
- Mac : `brew install tesseract`
- Linux : `sudo apt-get install tesseract-ocr tesseract-ocr-fra`

### Dépendances Python

Dans Antigravity IDE, ouvre un terminal sur ce dossier puis :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# Mac/Linux : source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Clé API LLM

Ce projet utilise **Gemini** par défaut (rapide, quota gratuit généreux, cohérent avec un usage
dans Antigravity qui est l'IDE de Google).

1. Récupère une clé sur https://aistudio.google.com/apikey
2. Définis la variable d'environnement :

```bash
# Windows (PowerShell)
$env:GOOGLE_API_KEY="ta_cle_ici"

# Mac/Linux
export GOOGLE_API_KEY="ta_cle_ici"
```

Si tu préfères OpenAI, change `provider: "openai"` et `model` dans `config.yaml`,
et définis `OPENAI_API_KEY` à la place.

## 3) Exécution du pipeline (dans l'ordre, une seule fois par nouveau document)

```bash
# 1. Dépose ta thèse PDF dans data/raw/  (ex: data/raw/these_papillons.pdf)

# 2. Extraction texte + images
python -m src.ingest

# 3. Chunking
python -m src.chunk_text

# 4. OCR sur les images extraites
python -m src.ocr_utils

# 5. Embeddings + index FAISS (texte et images)
python -m src.embed_index
```

Vérifie après chaque étape que les fichiers attendus existent bien :
- `data/processed/pages_raw.json`
- `data/processed/chunks.json`
- `index/text.index`, `index/text_meta.json`
- `index/image.index`, `index/image_meta.json` (si le PDF contient des images)

## 4) Lancer la démo

```bash
streamlit run app.py
```

Pose une question dans l'interface : tu obtiens une réponse générée, les images
pertinentes avec leur page source, et les extraits de texte utilisés (transparence/traçabilité,
utile pour la soutenance).

## 5) Pour changer de domaine plus tard (oiseaux, plantes, docs admin...)

1. Vide ou complète `data/raw/` avec les nouveaux PDF.
2. Relance uniquement les étapes 2 à 5 ci-dessus (`ingest` → `chunk_text` → `ocr_utils` → `embed_index`).
3. Aucune ligne de code à modifier : la genericité vient du fait que rien n'est codé "en dur"
   (pas de mot-clé "papillon" nulle part), tout passe par les métadonnées extraites automatiquement.

## Points à mentionner à l'oral (架构 / justification)

- **Double index** : un index texte classique (recherche sémantique sur le contenu + l'OCR)
  et un index image CLIP (recherche cross-modale : la question texte retrouve directement
  une image sans dépendre uniquement de l'OCR).
- **Traçabilité** : chaque chunk et chaque image gardent leur document et numéro de page d'origine
  → réponses avec sources, comme demandé dans le cahier des charges du projet PFA.
- **Généricité** : le pipeline ne connaît aucun concept métier ; il ingère n'importe quel PDF
  texte+images. C'est la définition même d'une architecture "domain-agnostic".
- **Limites assumées (à mentionner honnêtement)** : pas de Neo4j / Graph RAG dans cette version
  (hors scope du délai de 3h) — c'est une extension possible évoquée dans le rapport comme
  travail futur, en lien avec le volet Agentic Graph-Vector RAG déjà vu en cours.
