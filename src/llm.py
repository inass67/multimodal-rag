"""
Couche LLM unique du projet (utilisée par la génération, les résumés, l'agent
et l'évaluation).

- Fournisseurs : Gemini (SDK `google-genai`, repli sur l'ancien
  `google-generativeai`) ou OpenAI.
- Modèles de secours : si le modèle configuré n'existe plus côté fournisseur
  (les noms Gemini changent souvent), on essaie automatiquement les modèles
  listés dans `llm.fallback_models` de config.yaml.
- Intégration LangChain : `run_chain()` construit une chaîne LCEL
  `PromptTemplate | LLM | StrOutputParser` (LangChain Expression Language).
  Si LangChain n'est pas installé, on retombe sur un simple str.format,
  le résultat est identique.
- Réponse en direct (streaming) : `stream_llm()` renvoie la réponse morceau
  par morceau, comme ChatGPT.
"""
import os
import time

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    HAS_LANGCHAIN = True
except Exception:  # LangChain absent -> mode dégradé transparent
    HAS_LANGCHAIN = False


class LLMError(RuntimeError):
    pass


def _get_api_key(config):
    env_name = config["llm"].get("api_key_env", "GOOGLE_API_KEY")
    key = os.environ.get(env_name)
    if not key and config["llm"].get("provider") == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError(
            f"Clé API manquante : définis la variable d'environnement {env_name} "
            f"(PowerShell : $env:{env_name}=\"ta_cle\")"
        )
    return key


_THINKING_UNSUPPORTED = set()  # modèles qui refusent le réglage de réflexion


def _call_gemini(prompt, model, api_key, temperature, thinking="low"):
    try:
        from google import genai  # nouveau SDK officiel
        from google.genai import types
    except ImportError:
        import google.generativeai as genai_old  # ancien SDK (repli)
        genai_old.configure(api_key=api_key)
        m = genai_old.GenerativeModel(model)
        resp = m.generate_content(prompt, generation_config={"temperature": temperature})
        return resp.text or ""

    client = genai.Client(api_key=api_key)
    cfg_kwargs = {"temperature": temperature}
    # Les modèles Gemini récents « réfléchissent » avant de répondre, ce qui
    # peut prendre de longues secondes. Pour un RAG, la réponse est dans les
    # passages fournis : une réflexion courte suffit et la réponse est bien
    # plus rapide.
    if thinking and model not in _THINKING_UNSUPPORTED:
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
        except Exception:
            pass
    try:
        resp = client.models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(**cfg_kwargs))
    except Exception as e:
        if "thinking_config" in cfg_kwargs and "think" in str(e).lower():
            _THINKING_UNSUPPORTED.add(model)  # réglage refusé : on réessaie sans
            cfg_kwargs.pop("thinking_config")
            resp = client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg_kwargs))
        else:
            raise
    return resp.text or ""


def _call_openai(prompt, model, api_key, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _is_model_not_found(err):
    msg = str(err).lower()
    return ("not found" in msg or "404" in msg or "is not supported" in msg
            or "does not exist" in msg)


def call_llm(prompt, config, temperature=None, retries=2):
    """Envoie un prompt texte au LLM configuré et renvoie le texte de réponse."""
    llm_cfg = config["llm"]
    provider = llm_cfg.get("provider", "gemini")
    temperature = llm_cfg.get("temperature", 0.2) if temperature is None else temperature
    api_key = _get_api_key(config)
    models = [llm_cfg["model"]] + list(llm_cfg.get("fallback_models", []))

    last_err = None
    for model in models:
        for attempt in range(retries + 1):
            try:
                if provider == "gemini":
                    return _call_gemini(prompt, model, api_key, temperature,
                                        thinking=llm_cfg.get("thinking", "low"))
                if provider == "openai":
                    return _call_openai(prompt, model, api_key, temperature)
                raise LLMError(f"Fournisseur LLM non supporté : {provider}")
            except LLMError:
                raise
            except Exception as e:  # quota, réseau, modèle inconnu...
                last_err = e
                if _is_model_not_found(e):
                    break  # inutile de réessayer ce modèle -> modèle suivant
                msg = str(e).lower()
                if attempt < retries and ("429" in msg or "quota" in msg
                                          or "unavailable" in msg or "503" in msg):
                    time.sleep(2 * (attempt + 1))
                    continue
                break
    raise LLMError(f"Échec de l'appel LLM : {last_err}")


def stream_llm(prompt, config, temperature=None):
    """Version « en direct » de call_llm : renvoie la réponse morceau par morceau,
    comme ChatGPT. L'utilisateur voit les premiers mots après 1 à 2 secondes au
    lieu d'attendre la réponse complète."""
    llm_cfg = config["llm"]
    temperature = llm_cfg.get("temperature", 0.2) if temperature is None else temperature
    if llm_cfg.get("provider", "gemini") != "gemini":
        yield call_llm(prompt, config, temperature=temperature)
        return
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        yield call_llm(prompt, config, temperature=temperature)
        return

    client = genai.Client(api_key=_get_api_key(config))
    thinking = llm_cfg.get("thinking", "low")
    last_err = None
    for model in [llm_cfg["model"]] + list(llm_cfg.get("fallback_models", [])):
        for use_thinking in (True, False):
            if use_thinking and (not thinking or model in _THINKING_UNSUPPORTED):
                continue
            cfg_kwargs = {"temperature": temperature}
            if use_thinking:
                try:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
                except Exception:
                    pass
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                        model=model, contents=prompt, config=types.GenerateContentConfig(**cfg_kwargs)):
                    text = getattr(chunk, "text", None)
                    if text:
                        started = True
                        yield text
                return
            except Exception as e:
                if started:
                    raise LLMError(f"Réponse interrompue : {e}")
                last_err = e
                if use_thinking and "think" in str(e).lower():
                    _THINKING_UNSUPPORTED.add(model)
                    continue  # même modèle, sans le réglage de réflexion
                break  # modèle suivant
    raise LLMError(f"Échec de l'appel LLM : {last_err}")


def as_langchain_llm(config, temperature=None):
    """Expose notre LLM comme un Runnable LangChain (utilisable dans une chaîne LCEL)."""
    if not HAS_LANGCHAIN:
        raise ImportError("langchain-core n'est pas installé")

    def _invoke(prompt_value):
        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        return call_llm(text, config, temperature=temperature)

    return RunnableLambda(_invoke)


def run_chain(template, variables, config, temperature=None):
    """Exécute une chaîne LangChain `prompt | llm | parser`.

    `template` utilise la syntaxe {variable}. Les accolades littérales doivent
    être doublées ({{ }}), comme avec str.format.
    """
    if HAS_LANGCHAIN:
        chain = (PromptTemplate.from_template(template)
                 | as_langchain_llm(config, temperature)
                 | StrOutputParser())
        return chain.invoke(variables)
    return call_llm(template.format(**variables), config, temperature=temperature)