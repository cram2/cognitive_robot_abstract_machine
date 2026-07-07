from __future__ import annotations

import inflect
from lemminflect import getInflection, getLemma

#: The one shared ``inflect`` engine for the whole verbalization subsystem.
_engine = inflect.engine()


def plural(word: str) -> str:
    """
    :param word: An English noun (assumed singular).
    :return: The plural form of *word*, unconditionally (e.g. ``"Robot"`` → ``"Robots"``).

    >>> plural("Robot")
    'Robots'
    >>> plural("battery")
    'batteries'
    """
    return _engine.plural(word)


def ensure_plural(word: str) -> str:
    """
    :param word: An English noun in either number.
    :return: The plural form of *word*, without double-pluralising an already-plural word.

    >>> ensure_plural("Robot")
    'Robots'
    >>> ensure_plural("Robots")
    'Robots'
    """
    return word if _engine.singular_noun(word) else _engine.plural(word)


def singular(word: str) -> str:
    """
    :param word: An English noun in either number.
    :return: The singular form of *word*, leaving an already-singular word unchanged
        (*"tasks"* → *"task"*, *"task"* → *"task"*).

    >>> singular("tasks")
    'task'
    >>> singular("task")
    'task'
    >>> singular("children")
    'child'
    """
    singularized = _engine.singular_noun(word)
    return singularized if singularized else word


def is_plural(word: str) -> bool:
    """
    :param word: An English noun.
    :return: ``True`` when *word* is already in plural form.

    >>> is_plural("Robots")
    True
    >>> is_plural("Robot")
    False
    """
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

    >>> is_past_participle("assigned")
    True
    >>> is_past_participle("written")
    True
    >>> is_past_participle("battery")
    False
    """
    lowered = word.lower()
    return any(
        lowered in {form.lower() for form in getInflection(lemma, tag="VBN")}
        for lemma in getLemma(lowered, upos="VERB")
    )


def third_person_singular(lemma: str) -> str:
    """
    :param lemma: A verb in its base (lemma) form.
    :return: The third-person-singular present form (*"work"* → *"works"*, *"have"* → *"has"*,
        *"go"* → *"goes"*), via ``lemminflect``'s ``VBZ`` inflection; falls back to ``lemma + "s"``
        for an unknown word.

    >>> third_person_singular("work")
    'works'
    >>> third_person_singular("have")
    'has'
    >>> third_person_singular("contain")
    'contains'
    """
    forms = getInflection(lemma, tag="VBZ")
    return forms[0] if forms else lemma + "s"


def verb_lemma(word: str) -> str:
    """
    :param word: An English verb surface form.
    :return: Its base (lemma) form (*"connects"* → *"connect"*, *"is"* → *"be"*), via
        ``lemminflect``; the word unchanged when no verb lemma is known.

    >>> verb_lemma("connects")
    'connect'
    >>> verb_lemma("is")
    'be'
    """
    lemmas = getLemma(word.lower(), upos="VERB")
    return lemmas[0] if lemmas else word


def indefinite_article(following_word: str) -> str:
    """
    :param following_word: The word the article precedes.
    :return: The indefinite article (``"a"`` / ``"an"``) for *following_word*, chosen
        phonologically (e.g. ``"hour"`` → ``"an"``, ``"robot"`` → ``"a"``).

    >>> indefinite_article("hour")
    'an'
    >>> indefinite_article("robot")
    'a'
    """
    return _engine.a(following_word).split()[0]


def ordinal(index: int) -> str:
    """
    :param index: Zero-based integer index.
    :return: The English ordinal word for a zero-based *index* (``0`` → ``"first"``).

    >>> ordinal(0)
    'first'
    >>> ordinal(2)
    'third'
    """
    return _engine.ordinal(_engine.number_to_words(index + 1))


def index_ordinal(index: int) -> str:
    """
    :param index: A sequence index — a non-negative index counts from the start, a negative one
        from the end (Python slicing semantics).
    :return: The ordinal phrase naming the position: a non-negative index reads from the start
        (``0`` → ``"first"``), ``-1`` reads ``"last"``, and a more-negative index reads
        *"<ordinal> to last"* (``-2`` → ``"second to last"``).

    >>> index_ordinal(0)
    'first'
    >>> index_ordinal(-1)
    'last'
    >>> index_ordinal(-2)
    'second to last'
    """
    if index >= 0:
        return ordinal(index)
    if index == -1:
        return "last"
    return f"{ordinal(abs(index) - 1)} to last"


def cardinal(n: int) -> str:
    """
    :param n: A positive integer.
    :return: The English cardinal word for *n* (``2`` → ``"two"``).

    >>> cardinal(2)
    'two'
    >>> cardinal(21)
    'twenty-one'
    """
    return _engine.number_to_words(n)
