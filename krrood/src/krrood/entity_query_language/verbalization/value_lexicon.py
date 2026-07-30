from __future__ import annotations

import datetime
import enum

from typing_extensions import Any, List, Optional

from krrood.entity_query_language.utils import is_iterable

#: Human-readable nouns for the primitive types, whose bare ``__name__`` reads as programmer jargon
#: (*"int"*, *"str"*). Every other type keeps its ``__name__``.
_PRIMITIVE_TYPE_NOUNS = {
    int: "Integer",
    str: "Text",
    float: "floating-point number",
    bool: "Boolean",
}


def type_noun(type_: type) -> str:
    """
    Render a type as the noun it reads as in prose — the single type-name-as-noun point.

    :param type_: The class to name.
    :return: A friendly noun for a primitive (``int`` → *"Integer"*, ``float`` →
        *"floating-point number"*), else the class ``__name__``.

    >>> type_noun(int)
    'Integer'
    >>> type_noun(float)
    'floating-point number'
    """
    return _PRIMITIVE_TYPE_NOUNS.get(type_, type_.__name__)


def type_members(value: Any) -> Optional[List[type]]:
    """
    :return: the classes of a non-empty iterable of types, or ``None`` when *value* is not
        such a collection.

    >>> type_members((int, str))
    [<class 'int'>, <class 'str'>]
    >>> sorted(type_members({int, str}), key=lambda t: t.__name__)
    [<class 'int'>, <class 'str'>]
    >>> type_members([1, 2]) is None
    True
    """
    if not is_iterable(value):
        return None
    members = list(value)
    if members and all(isinstance(member, type) for member in members):
        return members
    return None


def value_phrase(value: Any) -> str:
    """
    Render a Python value as a human-readable string — the single value-lexicalisation
    point.

    * ``None`` → ``"nothing"`` (a genuine value-slot absence; a top-level ``== None`` comparison
      is rendered as an absence predicate, not via this function).
    * A bare ``type`` → its ``__name__`` (``Apple`` → ``"Apple"``).
    * An ``enum`` member → its ``name`` (``OPTION_A`` rather than ``<TestEnum.OPTION_A: …>``).
    * A ``datetime`` with no time → ``"May 23, 2026"``; with a time → ``"May 23, 2026 at 14:30"``.
    * Anything else → ``repr(value)``.

    :param value: Python value from a literal node.
    :return: Human-readable string representation.

    >>> value_phrase(None)
    'nothing'
    >>> value_phrase(int)
    'Integer'
    >>> value_phrase(datetime.datetime(2026, 5, 23))
    'May 23, 2026'
    >>> value_phrase(42)
    '42'
    """
    if value is None:
        return "nothing"
    if isinstance(value, type):
        return type_noun(value)
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, datetime.datetime):
        if value.time() == datetime.time.min:
            return value.strftime("%B %-d, %Y")
        return value.strftime("%B %-d, %Y at %H:%M")
    return repr(value)
