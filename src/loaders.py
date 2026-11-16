"""
Chargeurs de sources HÉTÉROGÈNES (exigence du sujet : « PDF, articles et
bases internes »).

Formats pris en charge :
- PDF            -> traité dans ingest.py (texte + images + OCR)
- .txt / .md     -> notes, pages de wiki internes exportées, bases de connaissances
- .docx          -> documents Word internes (procédures, rapports...)
- .html / .htm   -> pages internes (intranet) enregistrées
- URL            -> articles web, téléchargés au moment de l'ajout

Quand LangChain est installé, on utilise ses chargeurs officiels
(WebBaseLoader, Docx2txtLoader). Sinon, repli sur requests/BeautifulSoup et
python-docx : le résultat est le même pour la suite du pipeline.

Chaque document non-PDF est découpé en « sections » d'environ 350 mots en
respectant les paragraphes. Une section joue le rôle d'une page, ce qui permet
de citer précisément l'endroit d'où vient l'information (ex. « section 3 »).
"""
import os
import re

TEXT_EXT = {".txt", ".md", ".markdown"}
DOCX_EXT = {".docx"}
HTML_EXT = {".html", ".htm"}
PDF_EXT = {".pdf"}
SUPPORTED_EXT = TEXT_EXT | DOCX_EXT | HTML_EXT | PDF_EXT
UPLOAD_TYPES = sorted(e.lstrip(".") for e in SUPPORTED_EXT)


def source_type_from_ext(ext):
    ext = ext.lower()
    if ext in PDF_EXT:
        return "pdf"
    if ext in DOCX_EXT:
        return "docx"
    if ext in HTML_EXT:
        return "web"
    return "texte"


def split_into_sections(text, target_words=350):
    """Regroupe les paragraphes en sections d'environ `target_words` mots."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sections, current, count = [], [], 0
    for p in paragraphs:
        n = len(p.split())
        if current and count + n > target_words:
            sections.append("\n".join(current))
            current, count = [], 0
        current.append(p)
        count += n
    if current:
        sections.append("\n".join(current))
    return sections


def read_text_file(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


def read_docx(path):
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        docs = Docx2txtLoader(path).load()
        return "\n\n".join(d.page_content for d in docs)
    except Exception:
        pass
    import docx  # python-docx
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:  # les tableaux contiennent souvent l'essentiel
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def html_to_text(html):
    """Extrait le texte utile d'une page HTML (sans menus, scripts, pieds de page)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "form", "noscript", "svg", "iframe"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "pre", "blockquote"]):
        t = el.get_text(" ", strip=True)
        if len(t) > 1:
            blocks.append(t)
    text = "\n\n".join(blocks) if blocks else root.get_text("\n", strip=True)
    return title, text


def read_html_file(path):
    return html_to_text(read_text_file(path))


def fetch_url(url, timeout=20):
    """Télécharge un article web. Retourne (titre, html_brut, texte)."""
    if not re.match(r"^https?://", url.strip()):
        raise ValueError("L'adresse doit commencer par http:// ou https://")
    url = url.strip()
    try:
        from langchain_community.document_loaders import WebBaseLoader
        docs = WebBaseLoader(url).load()
        if docs and docs[0].page_content.strip():
            title = docs[0].metadata.get("title", "") or url
            text = re.sub(r"\n{3,}", "\n\n", docs[0].page_content)
            return title, None, text
    except Exception:
        pass
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (agent-recherche-documentaire)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding
    title, text = html_to_text(resp.text)
    return title or url, resp.text, text


def load_non_pdf(path):
    """Retourne (titre, liste_de_sections) pour un fichier non-PDF."""
    ext = os.path.splitext(path)[1].lower()
    title = os.path.basename(path)
    if ext in TEXT_EXT:
        text = read_text_file(path)
    elif ext in DOCX_EXT:
        text = read_docx(path)
    elif ext in HTML_EXT:
        page_title, text = read_html_file(path)
        title = page_title or title
    else:
        raise ValueError(f"Format non pris en charge : {ext}")
    return title, split_into_sections(text)


def safe_filename(name, max_len=80):
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip(" ._")
    return (name or "document")[:max_len]
