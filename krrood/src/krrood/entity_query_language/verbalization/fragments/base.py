from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    TypeVar,
)

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Spacing,
    GrammaticalNumber,
    Separator,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.exceptions import UnloweredFragmentError
from krrood.entity_query_language.verbalization.value_lexicon import (
    type_noun,
    value_phrase,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.microplanning.coordination import (
        FoldNode,
    )

_T = TypeVar("_T")


@dataclass
class VerbalizationFragment:
    """
    Abstract base for all verbalized output fragments.

    The hierarchy forms a tree: leaf nodes (``WordFragment``, ``RoleFragment``), inline
    composition (``PhraseFragment``), and block structure (``BlockFragment``).
    """

    source: Optional[FoldNode] = field(
        default=None, kw_only=True, compare=False, repr=False
    )
    """The EQL node (or synthetic coordination artifact) this fragment was built from, recorded so a
    later pass can trace the fragment back to the query it came from (for example, coreference looks
    up the focus of a query-sourced fragment). It is never used for equality or rendering."""

    def as_fragment(self) -> VerbalizationFragment:
        """:return: this fragment itself — the identity that lets an already-rendered fragment be a
        clause constituent on equal footing with the typed part-of-speech elements (see
        ``vocabulary.parts_of_speech.ClauseConstituent``)."""
        return self


@dataclass
class HasText:
    """Mixin contributing the display ``text`` field shared by text-bearing fragments."""

    text: str
    """The display text as a string (e.g. ``"the"``, ``"Robot"``)."""


@dataclass
class HasNumber:
    """Mixin contributing the grammatical ``number`` field shared by fragments whose own
    surface text is inflected for number."""

    number: GrammaticalNumber = field(default=GrammaticalNumber.SINGULAR, kw_only=True)
    """Grammatical number of this fragment's surface text."""


@dataclass
class HasPolarity:
    """Mixin contributing the grammatical ``negated`` field shared by fragments whose surface is
    realised in the negative (a verb with do-support, a copula with its negative suppletion).
    """

    negated: bool = field(default=False, kw_only=True)
    """Whether this leaf is realised negative — the morphology pass turns a ``VERB`` lemma into
    *"does not <lemma>"* and a copula into *"is not"*. Affirmative by default."""


@dataclass
class WordFragment(HasText, HasNumber, VerbalizationFragment):
    """
    Plain neutral text with no semantic role: articles, connectives, punctuation.

    May also carry a role-less noun (e.g. a group-key root), which can be tagged with a
    grammatical ``number`` for inflection.
    """

    spacing: Spacing = Spacing.NONE
    """Orthographic spacing of this token relative to its neighbours."""


@dataclass
class RoleFragment(HasText, HasNumber, HasPolarity, VerbalizationFragment):
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
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    ) -> RoleFragment:
        """
        Build a fragment for a variable, instantiated variable, or entity, linked to its type.

        :param label: Display text (type name or disambiguated label).
        :param expression: Expression whose ``_type_`` attribute supplies the source reference.
        :param number: Grammatical-number feature.
        :return: A fragment with the ``VARIABLE`` role.

        >>> RoleFragment.for_variable("Robot", variable(Robot, [])).text
        'Robot'
        """
        return cls(
            text=label,
            role=SemanticRole.VARIABLE,
            source_reference=SourceReference.for_type(
                expression._type_ if isinstance(expression, Selectable) else None
            ),
            number=number,
        )

    @classmethod
    def for_attribute(
        cls,
        owner: Optional[type],
        attribute_name: str,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
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

        >>> RoleFragment.for_attribute(Robot, "battery").text
        'battery'
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
        type_: Any,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
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

        >>> RoleFragment.for_type(Robot).text
        'Robot'
        """
        is_class = isinstance(type_, type)
        return cls(
            text=(
                text
                if text is not None
                else (type_noun(type_) if is_class else str(type_))
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

        >>> RoleFragment.for_literal(42).text
        '42'
        """
        return cls(text=value_phrase(value), role=SemanticRole.LITERAL)

    @classmethod
    def for_operator(cls, label: str) -> RoleFragment:
        """
        Build a fragment for an operator or copula (no source link).

        :param label: Display text (e.g. ``"is"``, ``"not"``, ``"greater than"``).
        :return: A fragment with the ``OPERATOR`` role.

        >>> RoleFragment.for_operator("greater than").text
        'greater than'
        """
        return cls(text=label, role=SemanticRole.OPERATOR)


@dataclass
class PhraseFragment(VerbalizationFragment):
    """An inline sequence of fragments joined by a separator."""

    parts: list[VerbalizationFragment]
    """Ordered list of child fragments."""

    separator: Separator = Separator.SPACE
    """Separator inserted between adjacent parts."""


@dataclass
class Clause(PhraseFragment):
    """A subject-led predicate clause — *"<subject> <verb/copula> …"* — built from typed
    part-of-speech constituents.

    Marking the clause (rather than inferring it from its parts) lets coreference recognise the
    leading constituent as the clause's grammatical subject: when that subject is the current
    discourse subject it pronominalises (*"it"* / *"they"*) and the finite verb or copula agrees
    with it (*"is"* → *"are"*, *"works"* → *"work"*). A plain :class:`PhraseFragment` carries no
    such promise about its first part.
    """


@dataclass
class NounPhrase(HasNumber, VerbalizationFragment):
    """
    A noun-phrase specification (a determiner phrase) — the grammatical features of a noun
    phrase, *not* its surface determiner.

    The determiner is chosen later from its definiteness and number, so that decision lives in
    exactly one place.

    Reference: :cite:t:`gatt2009simplenlg` — ``NPPhraseSpec``.
    """

    head: VerbalizationFragment
    """The noun leaf or sub-phrase the determiner attaches to."""

    definiteness: Definiteness = Definiteness.INDEFINITE
    """Determiner-system feature — selects *"a/an"* / *"the"* / no determiner."""

    modifiers: List[VerbalizationFragment] = field(default_factory=list)
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

    pre_head: Optional[VerbalizationFragment] = None
    """A qualifier placed between the determiner and the head (*"the [first two] Robots"*), e.g. a
    ``limit`` ranking phrase. Pre-nominal, so distinct from the post-nominal :attr:`modifiers`."""

    relative_clause: bool = False
    """``True`` when the :attr:`modifiers` form a relative clause (*"to which its primary is
    assigned"*). Such modifiers are set off by commas when the head is independently identified — a
    disambiguation number (*"Robot 1, to which …"*) makes the clause non-restrictive — but stay
    comma-less while the clause itself identifies the head (*"the Robot to which a Mission is
    assigned"*)."""


@dataclass
class PossessiveChain(VerbalizationFragment):
    """
    A navigation chain whose pronominal-vs-possessive surface form is decided by coreference
    (e.g. *"the amount of its amount_details"* vs. *"the amount of the amount_details of the
    BankTransaction"*).
    """

    parts: List[PathStep]
    """The chain's navigation path (:class:`PathStep` hops), innermost-last."""

    root_fragment: VerbalizationFragment
    """The referring noun phrase for the chain root."""

    root_referent_id: Optional[uuid.UUID] = None
    """The root variable's referent id — the chain pronominalises only when this is the current
    subject (and the root is not a numbered label)."""

    node_id: Optional[uuid.UUID] = None
    """The chain's own terminal node id. When the chain is a query's selected / measured quantity,
    a later mention of that same quantity (a WHERE on the very attribute being aggregated) reduces
    to a bare *"the <attribute>"* instead of repeating the whole possessive."""


@dataclass
class OwnedAttributes(VerbalizationFragment):
    """One or more attributes bound to an owner, whose possessive-vs-genitive surface coreference
    decides — *"its x, y, and z"* when the owner is the current discourse subject, else the genitive
    *"the x, y, and z of <owner>"*. Lets a caller name an owner's attributes without itself choosing
    the pronoun (which is the coreference pass's concern)."""

    attributes: VerbalizationFragment
    """The possessed attribute(s), pre-joined (*"x, y, and z"*)."""

    owner_fragment: VerbalizationFragment
    """The referring phrase for the owner, rendered by the normal recursion; used in the genitive
    (non-subject) branch (*"… of the position"* / *"… of its position"*)."""

    owner_referent_id: Optional[uuid.UUID] = None
    """The owner's referent id — the surface pronominalises only when this is the current subject
    (and the owner is not a numbered label)."""


@dataclass
class BlockFragment(VerbalizationFragment):
    """
    A named structural block with an optional header and a list of sub-items.
    """

    header: Optional[VerbalizationFragment]
    """Optional lead fragment (e.g. ``"Find Robot"`` or ``"If"``)."""

    items: list[VerbalizationFragment] = field(default_factory=list)
    """Ordered list of sub-item fragments."""

    conjunction: Optional[VerbalizationFragment] = None
    """When set, the items are coordinated by this conjunction: paragraph rendering joins them
    Oxford-style (*"a, b, and c"*) and hierarchical rendering prefixes the last bullet with it
    (*"… and c"*). ``None`` ⇒ the items are an uncoordinated list (comma-joined / plain bullets)."""

    bulleted_header: bool = False
    """When this block is itself a list item, ``True`` renders its header *as a bullet* (a list
    entry that owns a nested sub-list — e.g. an inference antecedent), while ``False`` renders the
    header as a plain label above its items (e.g. a *"whose"* / *"given that"* section)."""


# %% VerbalizationFragment catamorphism


def fold_fragment(
    fragment: VerbalizationFragment,
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

    A *catamorphism* is the unique fold that collapses a recursive structure to a single value by
    replacing each node with a handler; an *F-algebra* is that bundle of per-node handlers — here the
    four ``word`` / ``role`` / ``phrase`` / ``block`` functions. References: :cite:t:`meijer1991bananas`;
    :cite:t:`bird1997algebra`; and :cite:t:`gatt2009simplenlg` for the phrase specification traversed
    by realisation processors.

    :param fragment: Root of the fragment tree.
    :param word: Handler for ``WordFragment`` text.
    :param role: Handler for ``RoleFragment`` ``(text, role, source_reference)``.
    :param phrase: Handler for ``PhraseFragment`` ``(folded_parts, separator)``.
    :param block: Handler for a raw ``BlockFragment`` (controls its own recursion).
    :return: The folded value.

    >>> phrase = PhraseFragment(parts=[RoleFragment.for_operator("is"), RoleFragment.for_literal(42)])
    >>> fold_fragment(phrase, word=lambda text: [text], role=lambda text, role, ref: [text],
    ...               phrase=lambda parts, separator: [w for part in parts for w in part],
    ...               block=lambda block: [])
    ['is', '42']
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


# %% VerbalizationFragment transform (tree → tree)


def map_structural_children(
    fragment: VerbalizationFragment,
    recurse: Callable[[VerbalizationFragment], VerbalizationFragment],
) -> Optional[VerbalizationFragment]:
    """
    Rebuild a structural container (``PhraseFragment``, ``BlockFragment`` — the nodes that merely
    hold children) by applying *recurse* to each child, or return ``None`` for anything else (a
    leaf, or a node the caller treats specially).

    :param fragment: Node to rebuild.
    :param recurse: Transform applied to each child.
    :return: The rebuilt container, or ``None`` when *fragment* is not a structural container.

    >>> map_structural_children(RoleFragment.for_operator("is"), lambda f: f) is None
    True
    """
    match fragment:
        case PhraseFragment(parts=parts):
            return replace(fragment, parts=[recurse(p) for p in parts])
        case BlockFragment(header=header, items=items):
            return replace(
                fragment,
                header=None if header is None else recurse(header),
                items=[recurse(i) for i in items],
            )
        case _:
            return None


def map_fragment(
    fragment: VerbalizationFragment,
    leaf: Callable[[VerbalizationFragment], VerbalizationFragment],
) -> VerbalizationFragment:
    """
    Rebuild a fragment tree, replacing each leaf (``WordFragment`` / ``RoleFragment``) by
    ``leaf(node)`` and reconstructing the structural containers around the transformed children.

    The structural dual of ``fold_fragment``: it maps a tree to a tree rather than to a value.
    New nodes are returned and the input is left untouched, so shared sub-trees are safe.

    :param fragment: Root of the tree to transform.
    :param leaf: Transform applied to each leaf fragment (identity for unaffected leaves).
    :return: The rebuilt tree.

    >>> phrase = PhraseFragment(parts=[RoleFragment.for_operator("is"), RoleFragment.for_literal(42)])
    >>> shouted = map_fragment(phrase, lambda leaf: RoleFragment.for_operator(leaf.text.upper()))
    >>> flatten_fragment_to_plain_text(shouted)
    'IS 42'
    """
    rebuilt = map_structural_children(fragment, lambda f: map_fragment(f, leaf))
    return rebuilt if rebuilt is not None else leaf(fragment)


# %% VerbalizationFragment flattening


def flatten_fragment_to_plain_text(fragment: VerbalizationFragment) -> str:
    """
    Flatten a fragment tree to a plain string (no colour markup).

    :param fragment: Root of the fragment tree to flatten.
    :return: Plain-text representation with spaces between tokens.

    >>> flatten_fragment_to_plain_text(
    ...     PhraseFragment(parts=[RoleFragment.for_operator("is"), RoleFragment.for_literal(42)]))
    'is 42'
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


# %% VerbalizationFragment joining utilities


def oxford_comma(
    parts: list[VerbalizationFragment],
    conjunction: VerbalizationFragment,
    *,
    pair_comma: bool = False,
) -> VerbalizationFragment:
    """
    Join *parts* with Oxford-comma style: ``f1, f2, conj f3``.

    :param parts: Fragments to join.
    :param conjunction: Conjunction fragment (e.g. *"and"*, *"or"*).
    :param pair_comma: Whether a two-element list keeps a comma before the conjunction. Defaults to
        ``False`` — *"a and b"*, standard English (the serial comma is defined only for a series of
        three or more, so a pair takes none); pass ``True`` only to join two independent clauses.
    :return: A single fragment representing the joined sequence.

    >>> words = [RoleFragment.for_operator(w) for w in ("a", "b", "c")]
    >>> flatten_fragment_to_plain_text(oxford_comma(words, RoleFragment.for_operator("and")))
    'a, b, and c'
    >>> flatten_fragment_to_plain_text(oxford_comma(words[:2], RoleFragment.for_operator("and")))
    'a and b'
    """
    if not parts:
        return WordFragment(text="")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2 and not pair_comma:
        return PhraseFragment(parts=[parts[0], conjunction, parts[1]])
    head = parts[:-1]
    tail = parts[-1]
    result: list[VerbalizationFragment] = []
    for fragment in head:
        result.append(fragment)
        result.append(WordFragment(text=Separator.COMMA))
    result.append(PhraseFragment(parts=[conjunction, tail]))
    return PhraseFragment(parts=result, separator=Separator.NONE)
