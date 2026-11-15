"""
Etape 6 - Génération
Construit un prompt à partir des chunks récupérés et interroge le LLM configuré.
Le prompt est neutre / générique : il ne mentionne aucun domaine particulier.
"""
import os


def build_prompt(query, text_results):
    context = "\n\n".join(
        f"[Source: {r['doc']}, page {r['page']}]\n{r['content']}" for r in text_results
    )
    prompt = f"""Tu es un assistant qui répond STRICTEMENT à partir du contexte fourni ci-dessous.
Si l'information n'est pas présente dans le contexte, réponds "Je ne sais pas".

Important : si la question demande de "montrer", "donner l'image de" ou "afficher" quelque chose,
et que le contexte ci-dessous contient bien des informations sur ce sujet (même sans image),
ne réponds PAS "Je ne sais pas" — réponds avec les informations textuelles trouvées et précise
que l'image correspondante est affichée séparément ci-dessous (elle est gérée par un autre
composant du système, pas par toi). Ne dis "Je ne sais pas" que si le SUJET lui-même est absent
du contexte, pas simplement parce que tu ne peux pas générer d'image toi-même.

Contexte :
{context}

Question : {query}

Consignes :
- Réponds de façon claire, concise et directement utile.
- Cite la page source entre parenthèses quand c'est pertinent, ex: (page 12).
"""
    return prompt


def generate_answer(query, text_results, config):
    prompt = build_prompt(query, text_results)
    provider = config["llm"]["provider"]

    if provider == "gemini":
        import google.generativeai as genai
        api_key = os.environ.get(config["llm"]["api_key_env"])
        if not api_key:
            raise ValueError(f"Clé API manquante : définis la variable d'env {config['llm']['api_key_env']}")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config["llm"]["model"])
        response = model.generate_content(prompt)
        return response.text

    elif provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get(config["llm"]["api_key_env"])
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Provider LLM non supporté : {provider}")