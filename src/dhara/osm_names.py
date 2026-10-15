"""Which OpenStreetMap name to show. OSM's `name` is in the local script (Urdu in Pakistan,
Devanagari in Nepal, Chinese across the border); the English interface shows `name:en` or another
Latin-script name first, and the Nepali interface shows `name:ne` first."""

import re

_LATIN = re.compile(r"[A-Za-z]")
_SCRIPT_OK = re.compile(r"^[\sA-Za-z0-9À-ɏऀ-ॿ.,'()/&-]+$")  # Latin, Devanagari


def english_name(tags: dict, *keys: str) -> str:
    """Display name for English views: `name:en`, then any Latin-script name variant, then the
    original if it is Latin or Devanagari, else the original as the last resort. `keys` are tried
    first (e.g. "bridge:name"). A name only in another script is romanised."""
    for k in (*keys, "name:en", "int_name", "name:latin", "official_name:en", "alt_name:en"):
        v = str(tags.get(k) or "").strip()
        if v:
            return v
    for k, v in tags.items():
        v = str(v or "").strip()
        if k.startswith("name:") and v and _LATIN.search(v) and _SCRIPT_OK.match(v):
            return v
    name = str(tags.get("name") or "").strip()
    if not name or _SCRIPT_OK.match(name):
        return name
    # Only a non-Latin, non-Devanagari name (e.g. Urdu, Chinese): show a romanisation.
    from anyascii import anyascii

    roman = " ".join(anyascii(name).split())
    return roman.title() if roman else name


def nepali_name(tags: dict) -> str:
    """Display name for Nepali views: `name:ne`, then the original name."""
    return str(tags.get("name:ne") or tags.get("name") or "").strip()


def shown_name(name):
    """A stored name as shown in either interface: Latin and Devanagari names unchanged, any other
    script (e.g. Urdu, Chinese) romanised. Layers written before `english_name` existed still hold
    the raw local-script name. Empty values pass through unchanged."""
    if not name or not isinstance(name, str):
        return name
    s = name.strip()
    if not s or _SCRIPT_OK.match(s):
        return name
    from anyascii import anyascii

    roman = " ".join(anyascii(s).split())
    return roman.title() if roman else name
