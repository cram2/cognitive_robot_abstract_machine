from __future__ import annotations

import re

import inflect

inflect_engine = inflect.engine()


def _camel_to_words(name: str) -> str:
    """Convert a CamelCase class name to space-separated lowercase words.

    Examples: ``"HasRole"`` → ``"has role"``, ``"IsReachable"`` → ``"is reachable"``.
    """
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()


def _ordinal(n: int) -> str:
    return inflect_engine.ordinal(inflect_engine.number_to_words(n + 1))


def _ensure_plural(word: str) -> str:
    """Return *word* in plural form, without double-pluralising already-plural words."""
    return word if inflect_engine.singular_noun(word) else inflect_engine.plural(word)


def _apply_binding_aliases(text: str, alias_map: dict[str, str]) -> str:
    """Replace each verbalized binding value in *text* with its established field reference.

    Longer aliases are tried first to avoid partial replacements.
    """
    for value, field_ref in sorted(alias_map.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(value, field_ref)
    return text
