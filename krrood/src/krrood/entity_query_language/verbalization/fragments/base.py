"""
Fragment tree data model for verbalized output.

The fragment hierarchy forms the output IR that renderers traverse:

* :class:`WordFragment` — plain text (no semantic role).
* :class:`RoleFragment` — text with a :class:`SemanticRole` (drives colour / hyperlinks).
* :class:`PhraseFragment` — inline sequence of fragments joined by a separator.
* :class:`BlockFragment` — named structural block with header + bullet items.

Joining utilities (:func:`join_with`, :func:`oxford_and`) produce
:class:`PhraseFragment` trees from lists of fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Callable, List, Optional, TypeVar

from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef
from krrood.entity_query_language.verbalization.utils import _ensure_plural

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute

_T = TypeVar("_T")


@dataclass
class VerbFragment:
    """
    Abstract base for all verbalized output fragments.

    The fragment hierarchy forms a tree:

    * Leaf nodes: :class:`WordFragment`, :class:`RoleFragment`.
    * Inline composition: :class:`PhraseFragment`.
    * Block structure: :class:`BlockFragment`.

    Renderers traverse this tree to produce strings.
    """


@dataclass
class WordFragment(VerbFragment):
    """Plain neutral text with no semantic role: articles, connectives, punctuation."""

    text: str
    """The raw text string (e.g. ``"the"``, ``"and"``, ``","``)."""


@dataclass
class RoleFragment(VerbFragment):
    """
    Text carrying a :class:`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole`
    — drives colour markup and optional source hyperlinking.
    """

    text: str
    """Display text (e.g. ``"Robot"``, ``"is greater than"``)."""

    role: SemanticRole
    """Semantic role determining the colour applied by the formatter."""

    source_ref: Optional[SourceRef] = None
    """Optional reference to the Python class or attribute this fragment represents;
    used by
    :class:`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver`
    to build hyperlinks."""

    @classmethod
    def for_variable(cls, label: str, expression) -> RoleFragment:
        """
        Build a fragment for a
        :class:`~krrood.entity_query_language.core.variable.Variable`,
        :class:`~krrood.entity_query_language.core.variable.InstantiatedVariable`,
        or :class:`~krrood.entity_query_language.query.query.Entity`, linked to its type.

        :param label: Display text (type name or disambiguated label).
        :type label: str
        :param expression: Expression whose ``_type_`` attribute supplies the source reference.
        :return: :class:`RoleFragment` with :attr:`~SemanticRole.VARIABLE` role.
        :rtype: RoleFragment
        """
        return cls(
            text=label,
            role=SemanticRole.VARIABLE,
            source_ref=SourceRef.for_type(getattr(expression, "_type_", None)),
        )

    @classmethod
    def for_attribute(
        cls, owner, attribute_name: str, plural: bool = False
    ) -> RoleFragment:
        """
        Build a fragment for an attribute access, linked to its owner class.

        :param owner: Owner class of the attribute (used for source linking).
        :param attribute_name: Canonical attribute name on *owner*.
        :type attribute_name: str
        :param plural: Whether the attribute is pluralized in the display text.
        :type plural: bool
        :return: :class:`RoleFragment` with :attr:`~SemanticRole.ATTRIBUTE` role.
        :rtype: RoleFragment
        """
        label = attribute_name if not plural else _ensure_plural(attribute_name)
        return cls(
            text=label,
            role=SemanticRole.ATTRIBUTE,
            source_ref=SourceRef.for_attribute(owner, attribute_name),
        )

    @classmethod
    def for_operator(cls, label: str) -> RoleFragment:
        """
        Build a fragment for an operator or copula (no source link).

        :param label: Display text (e.g. ``"is"``, ``"not"``, ``"greater than"``).
        :type label: str
        :return: :class:`RoleFragment` with :attr:`~SemanticRole.OPERATOR` role.
        :rtype: RoleFragment
        """
        return cls(text=label, role=SemanticRole.OPERATOR)


@dataclass
class PhraseFragment(VerbFragment):
    """An inline sequence of fragments joined by a separator."""

    parts: list[VerbFragment]
    """Ordered list of child fragments."""

    separator: str = " "
    """String inserted between adjacent parts."""


@dataclass
class BlockFragment(VerbFragment):
    """
    A named structural block with an optional header and a list of sub-items.

    * :class:`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer`
      flattens header + items into a single comma-separated prose string.
    * :class:`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer`
      renders the header on one line, then each item as a bullet at the next indent level.
    """

    header: Optional[VerbFragment]
    """Optional lead fragment (e.g. ``"Find Robot"`` or ``"If"``)."""

    items: list[VerbFragment] = field(default_factory=list)
    """Ordered list of sub-item fragments."""


# ── Fragment catamorphism ──────────────────────────────────────────────────────


def fold_fragment(
    fragment: VerbFragment,
    *,
    word: Callable[[str], _T],
    role: Callable[[str, SemanticRole, Optional[SourceRef]], _T],
    phrase: Callable[[List[_T], str], _T],
    block: Callable[[BlockFragment], _T],
) -> _T:
    """
    Fold a :class:`VerbFragment` tree into a value of type ``_T`` by supplying one
    handler per node kind — the single, shared structural recursion over the IR.

    This is the *catamorphism* (the unique homomorphism from the fragment tree into
    a target algebra): the recursion scheme lives here once; each caller provides an
    *algebra* (the four handlers) describing how to combine results. Every consumer
    of the IR — plain-text flattening and each
    :class:`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer`
    — is expressed as one such fold, so the Word/Role/Phrase traversal is written
    exactly once instead of being copied per consumer.

    ``word``, ``role`` and ``phrase`` receive already-folded children; ``block``
    receives the raw :class:`BlockFragment` because block layout is genuinely
    consumer-specific (flat prose vs. indented bullets) and must control its own
    recursion (e.g. with depth).

    Concept references:

    * Catamorphism / F-algebra — Meijer, Fokkinga & Paterson (1991), "Functional
      Programming with Bananas, Lenses, Envelopes and Barbed Wire", FPCA; Bird & de
      Moor (1997), "Algebra of Programming".
    * Phrase specification traversed by realisation processors — Gatt & Reiter
      (2009), "SimpleNLG: A realisation engine for practical applications", ENLG.

    :param fragment: Root of the fragment tree.
    :param word: Handler for :class:`WordFragment` text.
    :param role: Handler for :class:`RoleFragment` ``(text, role, source_ref)``.
    :param phrase: Handler for :class:`PhraseFragment` ``(folded_parts, separator)``.
    :param block: Handler for a raw :class:`BlockFragment` (controls its own recursion).
    :return: The folded value.
    :rtype: _T
    """
    match fragment:
        case WordFragment(text=text):
            return word(text)
        case RoleFragment(text=text, role=semantic_role, source_ref=ref):
            return role(text, semantic_role, ref)
        case PhraseFragment(parts=parts, separator=separator):
            folded = [
                fold_fragment(p, word=word, role=role, phrase=phrase, block=block)
                for p in parts
            ]
            return phrase(folded, separator)
        case BlockFragment():
            return block(fragment)
        case _:
            return word("")


# ── Fragment flattening ────────────────────────────────────────────────────────


def flatten_fragment_to_plain_text(fragment: VerbFragment) -> str:
    """
    Flatten a :class:`VerbFragment` tree to a plain string (no colour markup).

    Used for internal comparisons, logging, and plain-text verbalization output.
    Expressed as a :func:`fold_fragment` over the plain-text algebra.

    :param fragment: Root of the fragment tree to flatten.
    :type fragment: VerbFragment
    :return: Plain-text representation with spaces between tokens.
    :rtype: str
    """

    def _block(b: BlockFragment) -> str:
        items = ", ".join(flatten_fragment_to_plain_text(i) for i in b.items)
        if b.header is None:
            return items
        header = flatten_fragment_to_plain_text(b.header)
        return f"{header} {items}" if items else header

    return fold_fragment(
        fragment,
        word=lambda text: text,
        role=lambda text, _role, _ref: text,
        phrase=lambda parts, separator: separator.join(parts),
        block=_block,
    )


# ── Fragment joining utilities ─────────────────────────────────────────────────


def join_with(parts: list[VerbFragment], separator: VerbFragment) -> VerbFragment:
    """
    Interleave *parts* with *separator* between each adjacent pair.

    :param parts: Fragments to join.
    :type parts: list[VerbFragment]
    :param separator: Separator fragment inserted between adjacent items.
    :type separator: VerbFragment
    :return: A single fragment (or the sole item when ``len(parts) == 1``).
    :rtype: VerbFragment
    """
    if not parts:
        return WordFragment(text="")
    if len(parts) == 1:
        return parts[0]
    result: list[VerbFragment] = []
    for i, fragment in enumerate(parts):
        result.append(fragment)
        if i < len(parts) - 1:
            result.append(separator)
    return PhraseFragment(parts=result, separator="")


def oxford_and(parts: list[VerbFragment], conjunction: VerbFragment) -> VerbFragment:
    """
    Join *parts* with Oxford-comma style: ``f1, f2, conj f3``.

    :param parts: Fragments to join.
    :type parts: list[VerbFragment]
    :param conjunction: Conjunction fragment (e.g. *"and"*, *"or"*).
    :type conjunction: VerbFragment
    :return: A single fragment representing the joined sequence.
    :rtype: VerbFragment
    """
    if not parts:
        return WordFragment(text="")
    if len(parts) == 1:
        return parts[0]
    head = parts[:-1]
    tail = parts[-1]
    result: list[VerbFragment] = []
    for fragment in head:
        result.append(fragment)
        result.append(WordFragment(text=", "))
    result.append(PhraseFragment(parts=[conjunction, tail]))
    return PhraseFragment(parts=result, separator="")
