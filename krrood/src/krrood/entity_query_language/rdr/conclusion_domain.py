"""
Resolve the set of values an RDR conclusion attribute may take, from its declared type.

When the expert labels a case with no ground truth (the ``ask_for_rule`` path), the RDR can
derive *what the conclusion is allowed to be* directly from the conclusion attribute's type
hint — without asking. For an :class:`enum.Enum` (or ``bool``) the allowable values are a
finite, enumerable set; for an open type (``str`` / an arbitrary class) the domain is not
enumerable and we fall back to a type check.

:class:`ConclusionDomain` is that resolved description; :func:`resolve_conclusion_domain`
builds one for ``owner_type.attribute_name``. Both are consumed by the conclusion validator
(reject out-of-domain answers) and by the interactive shell (show the allowable values, inject
enum members for tab-completion, source the example).
"""

from __future__ import annotations

import enum
import inspect
import types
import typing
from dataclasses import dataclass

from typing_extensions import Any, Dict, Optional, Tuple, get_args, get_origin

from krrood.class_diagrams.exceptions import CouldNotResolveType
from krrood.class_diagrams.utils import get_type_hints_of_object

#: The runtime type of ``None``, used to detect ``Optional`` / ``... | None`` annotations.
_NONE_TYPE = type(None)


@dataclass(frozen=True)
class ConclusionDomain:
    """The values an RDR conclusion attribute may take, resolved from its declared type."""

    expected_types: Tuple[type, ...]
    """The declared non-``None`` types of the attribute (``isinstance`` targets). Empty when
    the annotation could not be resolved (then any non-``None`` value is accepted)."""
    members: Tuple[Any, ...]
    """The enumerable allowable values (Enum members, or ``True``/``False``); empty when the
    domain is not enumerable."""
    is_enumerable: bool
    """Whether the domain is a finite, enumerable set (an Enum or ``bool``)."""
    allows_none: bool
    """Whether the declared type admits ``None`` (an ``Optional`` / ``... | None`` annotation)."""

    @property
    def type_display(self) -> str:
        """:return: A human label for the expected type(s) (e.g. ``Species`` or ``str or int``)."""
        if not self.expected_types:
            return "value"
        return " or ".join(t.__name__ for t in self.expected_types)

    def contains(self, value: Any) -> bool:
        """:return: Whether ``value`` is one of the enumerable members (identity-aware)."""
        return any(value is member or value == member for member in self.members)

    def display(self) -> str:
        """:return: The allowable values as a prose list, or the type label when not enumerable."""
        if self.is_enumerable:
            return ", ".join(repr(member) for member in self.members)
        return self.type_display

    def example_for(self, name: str) -> str:
        """:return: A copy-pasteable example assignment for the answer named ``name``."""
        if self.is_enumerable and self.members:
            return f"{name} = {self.members[0]!r}"
        return f"{name} = <{self.type_display}>"

    def namespace_bindings(self) -> Dict[str, Any]:
        """
        Names to inject into the expert's shell so the allowable values tab-complete.

        :return: ``{EnumType.__name__: EnumType}`` for each Enum among the expected types (so
            the expert can type ``Species.<tab>``); empty for non-enumerable / builtin domains.
        """
        bindings: Dict[str, Any] = {}
        for expected in self.expected_types:
            if inspect.isclass(expected) and issubclass(expected, enum.Enum):
                bindings[expected.__name__] = expected
        return bindings


def resolve_conclusion_domain(
    owner_type: type, attribute_name: str
) -> ConclusionDomain:
    """
    Resolve the conclusion domain for ``owner_type.attribute_name`` from its type hint.

    Unresolvable annotations degrade to a non-enumerable, no-expected-type domain (any
    non-``None`` value is then accepted), so callers never have to branch on failure.

    :param owner_type: The case type owning the attribute (e.g. ``Animal``).
    :param attribute_name: The conclusion attribute (e.g. ``"species"``).
    :return: The resolved :class:`ConclusionDomain`.
    """
    annotation = _annotation_of(owner_type, attribute_name)
    allows_none, expected_types = _split_optional(annotation)
    members = _enumerate_members(expected_types)
    return ConclusionDomain(
        expected_types=expected_types,
        members=members,
        is_enumerable=bool(members),
        allows_none=allows_none,
    )


def _annotation_of(owner_type: type, attribute_name: str) -> Any:
    """:return: The declared annotation for the attribute, or ``None`` if unresolvable."""
    try:
        return get_type_hints_of_object(owner_type).get(attribute_name)
    except (CouldNotResolveType, TypeError):
        return None


def _split_optional(annotation: Any) -> Tuple[bool, Tuple[type, ...]]:
    """:return: ``(allows_none, non_none_types)`` for a possibly-``Optional`` annotation."""
    if annotation is None:
        return False, ()
    if _is_union(annotation):
        args = get_args(annotation)
        allows_none = _NONE_TYPE in args
        non_none = tuple(
            arg for arg in args if arg is not _NONE_TYPE and isinstance(arg, type)
        )
        return allows_none, non_none
    if annotation is _NONE_TYPE:
        return True, ()
    if isinstance(annotation, type):
        return False, (annotation,)
    return False, ()


def _is_union(annotation: Any) -> bool:
    """:return: Whether ``annotation`` is a ``typing.Union`` or ``X | Y`` union."""
    origin = get_origin(annotation)
    return origin is typing.Union or origin is getattr(types, "UnionType", None)


def _enumerate_members(expected_types: Tuple[type, ...]) -> Tuple[Any, ...]:
    """:return: The enumerable members for a single Enum/``bool`` type, else ``()``."""
    if len(expected_types) != 1:
        return ()
    expected = expected_types[0]
    if inspect.isclass(expected) and issubclass(expected, enum.Enum):
        return tuple(expected)
    if expected is bool:
        return (False, True)
    return ()
