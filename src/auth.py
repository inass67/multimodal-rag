"""
Authentification simple pour l'application RAG (login / signup).

Stockage local dans users.json (à la racine du projet), mots de passe
JAMAIS stockés en clair : hashés avec un sel unique par utilisateur
(sha256(sel + mot_de_passe)).

Adapté à un usage démo/interne (PFE) : pas de gestion d'email de
confirmation, pas de récupération de mot de passe par email — juste
l'essentiel pour distinguer les utilisateurs et protéger l'accès.
"""
import json
import os
import hashlib
import secrets

USERS_FILE = "users.json"


def _hash_password(password, salt):
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def register_user(username, password, full_name=""):
    """Crée un nouveau compte. Retourne (succès: bool, message: str)."""
    username = (username or "").strip().lower()
    password = password or ""

    if not username or not password:
        return False, "Nom d'utilisateur et mot de passe requis."
    if len(username) < 3:
        return False, "Le nom d'utilisateur doit contenir au moins 3 caractères."
    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères."

    users = load_users()
    if username in users:
        return False, "Ce nom d'utilisateur existe déjà."

    salt = secrets.token_hex(8)
    users[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "full_name": full_name.strip() or username,
    }
    save_users(users)
    return True, "Compte créé avec succès. Vous pouvez maintenant vous connecter."


def verify_login(username, password):
    """Vérifie les identifiants. Retourne (succès: bool, nom_complet: str|None)."""
    users = load_users()
    username = (username or "").strip().lower()
    if username not in users:
        return False, None

    user = users[username]
    if _hash_password(password or "", user["salt"]) == user["password_hash"]:
        return True, user.get("full_name", username)
    return False, None
