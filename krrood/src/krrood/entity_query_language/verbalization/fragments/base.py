from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing_extensions import Any, Callable, List, Optional, TypeVar

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Spacing,
    Number,
    Separator,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.exceptions import UnloweredFragmentError
from krrood.entity_query_language.verbalization.value_lexicon import value_phrase

_T = TypeVar("_T")


@dataclass
class Fragment:
    """
    Abstract base for all verbalized output fragments.

    The hierarchy forms a tree: leaf nodes (``WordFragment``, ``RoleFragment``), inline
    composition (``PhraseFragment``), and block structure (``BlockFragment``).
    """

    source: Optional[SymbolicExpression] = field(
        default=None, kw_only=True, compare=False, repr=False
    )
    """Provenance: the EQL node this fragment was built from, stamped by the fold. A lossless
    side-channel a later pass can follow back to the read model (e.g. coreference reads the focus
    of a query-sourced fragment). Never participates in equality or rendering."""


@dataclass
class HasText:
    """Mixin contributing the display ``text`` field shared by text-bearing fragments."""

    text: str
    """The display text as a string (e.g. ``"the"``, ``"Robot"``)."""


@dataclass
class HasNumber:
    """Mixin contributing the grammatical ``number`` field shared by fragments whose own
    surface text is inflected for number."""

    number: Number = field(default=Number.SINGULAR, kw_only=True)
    """Grammatical number of this fragment's surface text."""


@dataclass
class WordFragment(HasText, HasNumber, Fragment):
    """
    Plain neutral text with no semantic role: articles, connectives, punctuation.

    May also carry a role-less noun (e.g. a group-key root), which can be tagged with a
    grammatical ``number`` for inflection.
    """

    spacing: Spacing = Spacing.NONE
    """Orthographic spacing of this token relative to its neighbours."""


@dataclass
class RoleFragment(HasText, HasNumber, Fragment):
    """
    Text carrying a semantic role — drives colour markup and optional source hyperlinking.
    """

    role: SemanticRole
    """Semantic role that determines this token's colour markup."""

    source_reference: Optional[SourceReference] = None
    """Optional reference to the Python class or attribute this fragment represents."""

    @classmethod
    def for_variable(
        cls,
        label: str,
        expression: SymbolicExpression,
        number: Number = Number.SINGULAR,
    ) -> RoleFragment:
        """
        Build a fragment for a variable, instantiated variable, or entity, linked to its type.

        :param label: Display text (type name or disambiguated label).
        :param expression: Expression whose ``_type_`` attribute supplies the source reference.
        :param number: Grammatical-number feature.
        :return: A fragment with the ``VARIABLE`` role.
        """
        return cls(
            text=label,
            role=SemanticRole.VARIABLE,
            source_reference=SourceReference.for_type(
                getattr(expression, "_type_", None)
            ),
            number=number,
        )

    @classmethod
    def for_attribute(
        cls,
        owner: Optional[type],
        attribute_name: str,
        number: Number = Number.SINGULAR,
        *,
        text: Optional[str] = None,
    ) -> RoleFragment:
        """
        Build a fragment for an attribute access, linked to its owner class.

        Inflection is not applied here; a plural *number* is a feature realised later, and the
        source link always uses the canonical name.

        :param owner: Owner class of the attribute (for source linking), or ``None`` for no link
            (when the access is referred to by name only, without a navigable owner in scope).
        :param attribute_name: Canonical attribute name on *owner* (always used for the link).
        :param number: Grammatical-number feature.
        :param text: Surface word override — a display name or a relational verb phrase
            (*"assigned to"*); defaults to *attribute_name*.
        :return: A fragment with the ``ATTRIBUTE`` role.
        """
        return cls(
            text=text if text is not None else attribute_name,
            role=SemanticRole.ATTRIBUTE,
            source_reference=(
                SourceReference.for_attribute(owner, attribute_name)
                if owner is not None
                else None
            ),
            number=number,
        )

    @classmethod
    def for_type(
        cls,
        type_: object,
        number: Number = Number.SINGULAR,
        *,
        text: Optional[str] = None,
    ) -> RoleFragment:
        """
        Build a fragment for a type rendered as a noun (*"Robot"*), with the ``VARIABLE`` role.

        :param type_: The class to name; linked to its source when it is a class (a non-class —
            a typing generic or unknown — yields no link).
        :param number: Grammatical-number feature.
        :param text: Surface word override; defaults to the class ``__name__``.
        :return: A fragment with the ``VARIABLE`` role.
        """
        is_class = isinstance(type_, type)
        return cls(
            text=(
                text
                if text is not None
                else (type_.__name__ if is_class else str(type_))
            ),
            role=SemanticRole.VARIABLE,
            source_reference=SourceReference.for_type(type_) if is_class else None,
            number=number,
        )

    @classmethod
    def for_literal(cls, value: Any) -> RoleFragment:
        """
        Build a fragment for a literal value (*42*, *"hello"*, an enum member), with the ``LITERAL``
        role; the surface text is the shared value lexicalisation.

        :param value: The Python value to render.
        :return: A fragment with the ``LITERAL`` role.
        """
        return cls(text=value_phrase(value), role=SemanticRole.LITERAL)

    @classmethod
    def for_operator(cls, label: str) -> RoleFragment:
        """
        Build a fragment for an operator or copula (no source link).

        :param label: Display text (e.g. ``"is"``, ``"not"``, ``"greater than"``).
        :return: A fragment with the ``OPERATOR`` role.
        """
        return cls(text=label, role=SemanticRole.OPERATOR)


@dataclass
class PhraseFragment(Fragment):
    """An inline sequence of fragments joined by a separator."""

    parts: list[Fragment]
    """Ordered list of child fragments."""

    separator: Separator = Separator.SPACE
    """Separator inserted between adjacent parts."""


@dataclass
class NounPhrase(HasNumber, Fragment):
    """
    A noun-phrase specification (a determiner phrase) — the grammatical features of a noun
    phrase, *not* its surface determiner.

    The determiner is chosen later from its definiteness and number, so that decision lives in
    exactly one place.

    Reference: Gatt & Reiter (2009), SimpleNLG — ``NPPhraseSpec``.
    """

    head: Fragment
    """The noun leaf or sub-phrase the determiner attaches to."""

    definiteness: Definiteness = Definiteness.INDEFINITE
    """Determiner-system feature — selects *"a/an"* / *"the"* / no determiner."""

    modifiers: List[Fragment] = field(default_factory=list)
    """Post-modifiers following the head (e.g. *"of the Root"*, *"where … such that …"*)."""

    modifier_separator: Separator = Separator.SPACE
    """Separator between the determiner-and-head group and the modifiers.  Default
    :attr:`Separator.SPACE` (*"drawers of Cabinets"*); :attr:`Separator.NONE` lets an appositive
    clause attach without a spurious space (*"a Robot, where …"*)."""

    referent_id: Optional[uuid.UUID] = None
    """When set, this noun phrase is a referring expression for that entity, and its
    definiteness holds the first-mention form."""

    subject_of_modifiers: bool = True
    """Whether this referring noun phrase is the discourse subject *of its own modifiers* — true for
    an entity with restrictive modifiers (*"a Robot whose battery exceeds its threshold"*), false
    for a quantity whose modifiers are a complement rooted elsewhere (*"the sum of the amount of its
    revenue"* — the complement keeps the outer subject, so it must not be re-scoped to the sum)."""

    pre_head: Optional[Fragment] = None
    """A qualifier placed between the determiner and the head (*"the [first two] Robots"*), e.g. a
    ``limit`` ranking phrase. Pre-nominal, so distinct from the post-nominal :attr:`modifiers`."""


@dataclass
class PossessiveChain(Fragment):
    """
    A navigation chain whose pronominal-vs-possessive surface form is decided by coreference
    (e.g. *"the amount of its amount_details"* vs. *"the amount of the amount_details of the
    BankTransaction"*).
    """

    parts: List[PathStep]
    """The chain's navigation path (:class:`PathStep` hops), innermost-last."""

    root_fragment: Fragment
    """The referring noun phrase for the chain root."""

    root_referent_id: Optional[uuid.UUID] = None
    """The root variable's referent id — the chain pronominalises only when this is the current
    subject (and the root is not a numbered label)."""

    node_id: Optional[uuid.UUID] = None
    """The chain's own terminal node id. When the chain is a query's selected / measured quantity,
    a later mention of that same quantity (a WHERE on the very attribute being aggregated) reduces
    to a bare *"the <attribute>"* instead of repeating the whole possessive."""


@dataclass
class BlockFragment(Fragment):
    """
    A named structural block with an optional header and a list of sub-items.
    """

    header: Optional[Fragment]
    """Optional lead fragment (e.g. ``"Find Robot"`` or ``"If"``)."""

    items: list[Fragment] = field(default_factory=list)
    """Ordered list of sub-item fragments."""

    conjunction: Optional[Fragment] = None
    """When set, the items are coordinated by this conjunction: paragraph rendering joins them
    Oxford-style (*"a, b, and c"*) and hierarchical rendering prefixes the last bullet with it
    (*"… and c"*). ``None`` ⇒ the items are an uncoordinated list (comma-joined / plain bullets)."""

    bulleted_header: bool = False
    """When this block is itself a list item, ``True`` renders its header *as a bullet* (a list
    entry that owns a nested sub-list — e.g. an inference antecedent), while ``False`` renders the
    header as a plain label above its items (e.g. a *"whose"* / *"given that"* section)."""


# ── Fragment catamorphism ──────────────────────────────────────────────────────


def fold_fragment(
    fragment: Fragment,
    *,
    word: Callable[[str], _T],
    role: Callable[[str, SemanticRole, Optional[SourceReference]], _T],
    phrase: Callable[[List[_T], str], _T],
    block: Callable[[BlockFragment], _T],
) -> _T:
    """
    Fold a fragment tree into a value of type ``_T`` by supplying one handler per node kind —
    the single, shared structural recursion over the fragment tree.

    This is the catamorphism over the fragment tree: the recursion scheme lives here once, and
    each caller provides an algebra (the four handlers) describing how to combine results.
    ``word``, ``role`` and ``phrase`` receive already-folded children; ``block`` receives the
    raw ``BlockFragment`` because block layout is consumer-specific and controls its own
    recursion.

    Concept references:

    * Catamorphism / F-algebra — Meijer, Fokkinga & Paterson (1991), "Functional Programming
      with Bananas, Lenses, Envelopes and Barbed Wire", FPCA; Bird & de Moor (1997), "Algebra
      of Programming".
    * Phrase specification traversed by realisation processors — Gatt & Reiter (2009),
      "SimpleNLG: A realisation engine for practical applications", ENLG.

    :param fragment: Root of the fragment tree.
    :param word: Handler for ``WordFragment`` text.
    :param role: Handler for ``RoleFragment`` ``(text, role, source_reference)``.
    :param phrase: Handler for ``PhraseFragment`` ``(folded_parts, separator)``.
    :param block: Handler for a raw ``BlockFragment`` (controls its own recursion).
    :return: The folded value.
    """
    match fragment:
        case WordFragment(text=text):
            return word(text)
        case RoleFragment(text=text, role=semantic_role, source_reference=reference):
            return role(text, semantic_role, reference)
        case PhraseFragment(parts=parts, separator=separator):
            folded = [
                fold_fragment(p, word=word, role=role, phrase=phrase, block=block)
                for p in parts
            ]
            return phrase(folded, separator)
        case BlockFragment():
            return block(fragment)
        case _:
            raise UnloweredFragmentError(fragment=fragment)


# ── Fragment transform (tree → tree) ────────────────────────────────────────────


def map_structural_children(
    fragment: Fragment, recurse: Callable[[Fragment], Fragment]
) -> Optional[Fragment]:
    """
    Rebuild a structural container (``PhraseFragment``, ``BlockFragment`` — the nodes that merely
    hold children) by applying *recurse* to each child, or return ``None`` for anything else (a
    leaf, or a node the caller treats specially).

    :param fragment: Node to rebuild.
    :param recurse: Transform applied to each child.
    :return: The rebuilt container, or ``None`` when *fragment* is not a structural container.
    """
    match fragment:
        case PhraseFragment(parts=parts):
            # replace() preserves the separator (and any future fields); only children change.
            return replace(fragment, parts=[recurse(p) for p in parts])
        case BlockFragment(header=header, items=items):
            # replace() preserves conjunction / bulleted_header (and source); only children change.
            return replace(
                fragment,
                header=None if header is None else recurse(header),
                items=[recurse(i) for i in items],
            )
        case _:
            return None


def map_fragment(fragment: Fragment, leaf: Callable[[Fragment], Fragment]) -> Fragment:
    """
    Rebuild a fragment tree, replacing each leaf (``WordFragment`` / ``RoleFragment``) by
    ``leaf(node)`` and reconstructing the structural containers around the transformed children.

    The structural dual of ``fold_fragment``: it maps a tree to a tree rather than to a value.
    New nodes are returned and the input is left untouched, so shared sub-trees are safe.

    :param fragment: Root of the tree to transform.
    :param leaf: Transform applied to each leaf fragment (identity for unaffected leaves).
    :return: The rebuilt tree.
    """
    rebuilt = map_structural_children(fragment, lambda f: map_fragment(f, leaf))
    return rebuilt if rebuilt is not None else leaf(fragment)


# ── Fragment flattening ────────────────────────────────────────────────────────


def flatten_fragment_to_plain_text(fragment: Fragment) -> str:
    """
    Flatten a fragment tree to a plain string (no colour markup).

    :param fragment: Root of the fragment tree to flatten.
    :return: Plain-text representation with spaces between tokens.
    """

    def _block(b: BlockFragment) -> str:
        rendered = [flatten_fragment_to_plain_text(i) for i in b.items]
        if b.conjunction is not None and len(rendered) > 1:
            conjunction = flatten_fragment_to_plain_text(b.conjunction)
            rendered[-1] = f"{conjunction} {rendered[-1]}"
        items = ", ".join(rendered)
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


def oxford_comma(
    parts: list[Fragment], conjunction: Fragment, *, pair_comma: bool = False
) -> Fragment:
    """
    Join *parts* with Oxford-comma style: ``f1, f2, conj f3``.

    :param parts: Fragments to join.
    :param conjunction: Conjunction fragment (e.g. *"and"*, *"or"*).
    :param pair_comma: Whether a two-element list keeps a comma before the conjunction. Defaults to
        ``False`` — *"a and b"*, standard English (the serial comma is defined only for a series of
        three or more, so a pair takes none); pass ``True`` only to join two independent clauses.
    :return: A single fragment representing the joined sequence.
    """
    if not parts:
        return WordFragment(text="")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2 and not pair_comma:
        return PhraseFragment(parts=[parts[0], conjunction, parts[1]])
    head = parts[:-1]
    tail = parts[-1]
    result: list[Fragment] = []
    for fragment in head:
        result.append(fragment)
        result.append(WordFragment(text=Separator.COMMA))
    result.append(PhraseFragment(parts=[conjunction, tail]))
    return PhraseFragment(parts=result, separator=Separator.NONE)
