"""
api_key.py
==========
Petit module partage pour stocker et relire la cle API GFW une seule fois.
La cle est ecrite dans api_key.txt a la racine du projet.
"""

from pathlib import Path

# Fichier de stockage, a la racine du projet (a cote de ce module)
_KEY_FILE = Path(__file__).parent / "api_key.txt"


def get_api_key():
    """Renvoie la cle stockee, ou None si aucune cle enregistree."""
    if not _KEY_FILE.exists():
        return None
    key = _KEY_FILE.read_text(encoding="utf-8").strip()
    return key or None


def save_api_key(key):
    """Enregistre la cle (ecrase l'ancienne). Renvoie la cle nettoyee."""
    key = (key or "").strip()
    if key:
        _KEY_FILE.write_text(key, encoding="utf-8")
    return key


def has_api_key():
    """True si une cle non vide est enregistree."""
    return get_api_key() is not None