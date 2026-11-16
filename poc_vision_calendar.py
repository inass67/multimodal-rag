"""
Preuve de concept — Correction du bug "calendrier encodé visuellement"

Contexte : les fiches espèces du guide contiennent une frise J F M A M J J A
S O N D où les mois d'activité sont indiqués par un surlignage bleu (pas par
du texte). L'extraction texte classique perd cette information.

Ce script teste la solution proposée sur UN cas réel (Vulcain, page 37) :
1. Rendu de la page en image (pypdfium2, déjà utilisé dans le pipeline)
2. Envoi de l'image à Gemini Vision avec un prompt ciblé
3. Extraction de la réponse : quels mois sont surlignés ?

Ne touche à AUCUN fichier du pipeline principal — script isolé, sans risque
pour la démo déjà fonctionnelle.

Utilisation :
    $env:GOOGLE_API_KEY="ta_clé"
    python poc_vision_calendar.py
"""
import os
import sys
import pypdfium2 as pdfium

PDF_PATH = "cache/documents/ce70510d4eeaba06/raw/Guide-papillons.pdf"  # chemin réel dans le cache
PAGE_NUMBER = 37  # numéro de page tel qu'affiché dans le guide (1-indexé)
ESPECE = "Vulcain"


def render_page_as_image(pdf_path, page_number, scale=3):
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_number - 1]  # pypdfium2 est 0-indexé
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def ask_gemini_vision(image, espece):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERREUR : définis d'abord la variable d'environnement GOOGLE_API_KEY")
        sys.exit(1)

    from vision_analysis import make_vision_model
    model = make_vision_model("gemini-3.6-flash", api_key)

    prompt = f"""Cette image est une page d'un guide sur les papillons, consacrée à l'espèce "{espece}".

Elle contient une frise calendaire avec les 12 lettres J F M A M J J A S O N D
(une par mois, de janvier à décembre). Une partie de cette frise est
surlignée par une barre de couleur (généralement bleue), indiquant les mois
où le papillon est actif.

Identifie précisément :
1. Quels mois (nom complet en français) sont couverts par la barre colorée ?
2. Combien de mois cela représente-t-il ?

Réponds uniquement avec les mois de début et de fin, et le nombre total de
mois, sous la forme : "Actif de [mois début] à [mois fin] ([nombre] mois)."
"""

    return model(prompt, image)


if __name__ == "__main__":
    print(f"Rendu de la page {PAGE_NUMBER} en image...")
    img = render_page_as_image(PDF_PATH, PAGE_NUMBER)

    print("Envoi à Gemini Vision pour analyse du calendrier...")
    result = ask_gemini_vision(img, ESPECE)

    print("\n" + "=" * 60)
    print(f"RÉSULTAT pour {ESPECE} (page {PAGE_NUMBER})")
    print("=" * 60)
    print(result)
    print("\nRéponse attendue (vérifiée visuellement) : avril à octobre (7 mois)")
