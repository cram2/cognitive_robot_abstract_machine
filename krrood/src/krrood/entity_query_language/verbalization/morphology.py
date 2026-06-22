from __future__ import annotations

import inflect
from lemminflect import getInflection, getLemma

from typing_extensions import Dict

#: The one shared ``inflect`` engine for the whole verbalization subsystem.
_engine = inflect.engine()

#: Domain-specific exceptions consulted *before* ``inflect`` — the override hook for
#: terms the general engine gets wrong (irregular/invariant plurals, acronym articles).
_plural_overrides: Dict[str, str] = {}
_article_overrides: Dict[str, str] = {}


def register_plural(singular: str, plural_form: str) -> None:
    """Register a domain-specific plural form for *singular*.

    For an invariant noun, register the word as its own plural
    (``register_plural("sheep", "sheep")``).

    :param singular: The singular surface form.
    :param plural_form: The plural surface form to emit.
    """
    _plural_overrides[singular] = plural_form


def register_indefinite_article(word: str, article: str) -> None:
    """Register a domain-specific indefinite article (``"a"`` / ``"an"``) for *word* — e.g.
    ``register_indefinite_article("FBI", "an")`` for an initialism ``inflect`` mis-reads.

    :param word: The word the article precedes.
    :param article: ``"a"`` or ``"an"``.
    """
    _article_overrides[word] = article


def clear_overrides() -> None:
    """Drop all registered morphology overrides (chiefly for test isolation)."""
    _plural_overrides.clear()
    _article_overrides.clear()


def plural(word: str) -> str:
    """
    :param word: An English noun (assumed singular).
    :return: The plural form of *word*, unconditionally (e.g. ``"Robot"`` → ``"Robots"``).
    """
    return _plural_overrides.get(word) or _engine.plural(word)


def ensure_plural(word: str) -> str:
    """
    :param word: An English noun in either number.
    :return: The plural form of *word*, without double-pluralising an already-plural word.
    """
    if word in _plural_overrides:
        return _plural_overrides[word]
    if word in _plural_overrides.values():  # already a registered plural form
        return word
    return word if _engine.singular_noun(word) else _engine.plural(word)


def is_plural(word: str) -> bool:
    """
    :param word: An English noun.
    :return: ``True`` when *word* is already in plural form.
    """
    if word in _plural_overrides.values():
        return True
    if word in _plural_overrides:
        return False
    return bool(_engine.singular_noun(word))


def is_past_participle(word: str) -> bool:
    """
    :param word: A single English word.
    :return: ``True`` when *word* is a past-participle verb form — regular (*"assigned"*,
        *"located"*) or irregular (*"sent"*, *"given"*, *"written"*) — and ``False`` for a base form
        (*"assign"*), a noun (*"battery"*), or a past tense that differs from the participle
        (*"sang"* vs *"sung"*).

    Deterministic dictionary + rule lookup (``lemminflect``): the word's verb lemma is taken, and the
    word is checked against that lemma's generated ``VBN`` forms. No statistical model, no data
    download — so it is reproducible and offline, unlike a context-sensitive POS tagger.
    """
    lowered = word.lower()
    return any(
        lowered in {form.lower() for form in getInflection(lemma, tag="VBN")}
        for lemma in getLemma(lowered, upos="VERB")
    )


def indefinite_article(following_word: str) -> str:
    """
    :param following_word: The word the article precedes.
    :return: The indefinite article (``"a"`` / ``"an"``) for *following_word*, chosen
        phonologically (e.g. ``"hour"`` → ``"an"``, ``"robot"`` → ``"a"``).
    """
    if following_word in _article_overrides:
        return _article_overrides[following_word]
    return _engine.a(following_word).split()[0]


def ordinal(index: int) -> str:
    """
    :param index: Zero-based integer index.
    :return: The English ordinal word for a zero-based *index* (``0`` → ``"first"``).
    """
    return _engine.ordinal(_engine.number_to_words(index + 1))


def cardinal(n: int) -> str:
    """
    :param n: A positive integer.
    :return: The English cardinal word for *n* (``2`` → ``"two"``).
    """
    return _engine.number_to_words(n)
