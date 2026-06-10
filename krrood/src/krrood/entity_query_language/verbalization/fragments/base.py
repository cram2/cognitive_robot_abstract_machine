"""
Fragment tree data model for verbalized output.

The fragment hierarchy forms the output IR that renderers traverse:

* :class:`WordFragment` — plain text (no semantic role).
* :class:`RoleFragment` — text with a :class:`SemanticRole` (drives colour / hyperlinks).
* :class:`PhraseFragment` — inline sequence of fragments joined by a separator.
* :class:`BlockFragment` — named structural block with header + bullet items.

The joining utility :func:`oxford_and` produces a :class:`PhraseFragment` tree from a list
of fragments (Oxford-comma style).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Callable, List, Optional, Tuple, TypeVar

from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_ref import SourceRef

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
    """Plain neutral text with no semantic role: articles, connectives, punctuation.

    May also carry a noun rendered role-lessly (e.g. a group-key root); such a leaf can be
    tagged :attr:`number` for the morphology pass to pluralise, like a ``RoleFragment``.
    """

    text: str
    """The raw text string (e.g. ``"the"``, ``"and"``, ``","``)."""

    number: Number = Number.SINGULAR
    """Grammatical number *feature* — the morphology pass pluralises a ``PLURAL`` leaf's text."""


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

    number: Number = Number.SINGULAR
    """Grammatical number *feature* (not yet applied to :attr:`text`).  The
    :class:`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`
    pass pluralises the text of leaves tagged :attr:`Number.PLURAL`."""

    @classmethod
    def for_variable(
        cls, label: str, expression, number: Number = Number.SINGULAR
    ) -> RoleFragment:
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
            number=number,
        )

    @classmethod
    def for_attribute(
        cls, owner, attribute_name: str, number: Number = Number.SINGULAR
    ) -> RoleFragment:
        """
        Build a fragment for an attribute access, linked to its owner class.

        Inflection is **not** applied here — a ``PLURAL`` *number* is a feature the
        morphology pass realises later (the source link always uses the canonical name).

        :param owner: Owner class of the attribute (used for source linking).
        :param attribute_name: Canonical attribute name on *owner*.
        :type attribute_name: str
        :param number: Grammatical-number feature (pluralised by the morphology pass).
        :return: :class:`RoleFragment` with :attr:`~SemanticRole.ATTRIBUTE` role.
        :rtype: RoleFragment
        """
        return cls(
            text=attribute_name,
            role=SemanticRole.ATTRIBUTE,
            source_ref=SourceRef.for_attribute(owner, attribute_name),
            number=number,
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
class NounPhrase(VerbFragment):
    """
    A **noun-phrase specification** (a determiner phrase / DP) — carries the grammatical
    features of a noun phrase, *not* its surface determiner.

    A rule emits this whenever it means *"a noun phrase"*; the
    :class:`~krrood.entity_query_language.verbalization.rendering.determiner_processor.DeterminerProcessor`
    pass lowers it to a :class:`PhraseFragment`, choosing the determiner from
    :attr:`definiteness` × :attr:`number` (the concord table) and tagging the head's
    :class:`Number` so the morphology pass inflects it.  Centralising the determiner decision
    here means it lives in exactly one place instead of being re-decided at every NP site.

    Reference: Gatt & Reiter (2009), SimpleNLG — ``NPPhraseSpec`` (a phrase spec carrying
    number / definiteness features, realised by a downstream processor).
    """

    head: VerbFragment
    """The noun leaf/sub-phrase the determiner attaches to (e.g. a ``VARIABLE``-role noun)."""

    number: Number = Number.SINGULAR
    """Grammatical number — drives both head inflection and the determiner concord."""

    definiteness: Definiteness = Definiteness.INDEFINITE
    """Determiner-system feature — selects *"a/an"* / *"the"* / no determiner."""

    modifiers: List[VerbFragment] = field(default_factory=list)
    """Post-modifiers following the head (e.g. *"of the Root"*, *"where … such that …"*)."""

    modifier_separator: str = " "
    """Separator between the determiner+head group and the :attr:`modifiers`.  Default ``" "``
    (*"drawers of Cabinets"*); ``""`` lets an appositive clause attach without a spurious space
    (*"a Robot"* + *", where …"* → *"a Robot, where …"*)."""

    referent_id: Optional[uuid.UUID] = None
    """When set, this NP is a **referring expression** for that entity.  :attr:`definiteness`
    then holds the *first-mention* form; the
    :class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`
    pass downgrades a *repeat* mention to definite (dropping :attr:`modifiers`, keeping only
    :attr:`head` as the label) or to a pronoun — the discourse decision, made in one place."""


@dataclass
class SubjectScope(VerbFragment):
    """
    Marks the region in which :attr:`subject_id` is the pronoun-eligible discourse subject.

    A structural wrapper (the document-order replacement for the build-time
    ``push_subject``/``pop_subject`` stack): the
    :class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`
    pushes :attr:`subject_id` on entry and pops it on exit, so a referring NP whose referent is
    the current subject can be pronominalised.  After that pass it is replaced by its (resolved)
    :attr:`child`; all other passes recurse through it transparently.
    """

    subject_id: Optional[uuid.UUID]
    """The subject's referent id, or ``None`` for a scope with no single subject (e.g. ``SetOf``),
    which suppresses pronominalisation."""

    child: VerbFragment
    """The wrapped fragment the scope applies to."""


@dataclass
class PossessiveChain(VerbFragment):
    """
    A navigation chain whose **pronominal-vs-possessive** surface form is decided by coreference.

    Emitted by the chain rule instead of pre-rendering, because the choice between
    *"the amount of its amount_details"* (root is the current subject) and
    *"the amount of the amount_details of the BankTransaction"* (otherwise) is a discourse
    decision the :class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`
    makes — it then calls the
    :mod:`~krrood.entity_query_language.verbalization.rendering.possessive` builders to render
    the chosen form.  Bool-predicative chains never pronominalise and so render directly without
    this node.
    """

    parts: List[Tuple[str, Optional[SourceRef]]]
    """The chain's ``(attr_name, source_ref)`` path, innermost-last (the same shape the
    possessive/pronominal builders consume)."""

    root_fragment: VerbFragment
    """The referring noun phrase for the chain root, used by the *possessive* rendering (and
    resolved for first/subsequent mention by the same pass)."""

    root_referent_id: Optional[uuid.UUID] = None
    """The root variable's referent id — the chain pronominalises only when this is the current
    subject (and the root is not a numbered label)."""


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
        case SubjectScope(child=child):
            return fold_fragment(
                child, word=word, role=role, phrase=phrase, block=block
            )
        case _:
            return word("")


# ── Fragment transform (tree → tree) ────────────────────────────────────────────


def map_structural_children(
    fragment: VerbFragment, recurse: Callable[[VerbFragment], VerbFragment]
) -> Optional[VerbFragment]:
    """
    Rebuild a **structural container** (the nodes that merely hold children —
    :class:`PhraseFragment`, :class:`BlockFragment`, :class:`SubjectScope`) by applying *recurse*
    to each child, or return ``None`` for anything else (a leaf, or a node the caller treats
    specially).

    This is the one definition of *"how the recursive containers are rebuilt"*, shared by every
    tree→tree pass (:func:`map_fragment` and the stateful
    :class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`
    walk) so the container shapes are enumerated once.  Each pass supplies its own *recurse*
    (plain self-recursion for ``map_fragment``; a scope-tracking walk for coreference) and keeps
    its own handling of the *non*-container nodes it cares about.

    :param fragment: Node to rebuild.
    :param recurse: Transform applied to each child.
    :return: The rebuilt container, or ``None`` when *fragment* is not a structural container.
    """
    match fragment:
        case PhraseFragment(parts=parts, separator=separator):
            return PhraseFragment(
                parts=[recurse(p) for p in parts], separator=separator
            )
        case BlockFragment(header=header, items=items):
            return BlockFragment(
                header=None if header is None else recurse(header),
                items=[recurse(i) for i in items],
            )
        case SubjectScope(subject_id=subject_id, child=child):
            return SubjectScope(subject_id=subject_id, child=recurse(child))
        case _:
            return None


def map_fragment(
    fragment: VerbFragment, leaf: Callable[[VerbFragment], VerbFragment]
) -> VerbFragment:
    """
    Rebuild a :class:`VerbFragment` tree, replacing each **leaf** (``WordFragment`` /
    ``RoleFragment``) by ``leaf(node)`` and reconstructing the structural containers
    (:func:`map_structural_children`) around the transformed children.

    The structural dual of :func:`fold_fragment` (which folds *to a value*): this maps a
    tree *to a tree*, the recursion scheme a realisation pass (e.g. the
    :class:`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`)
    needs.  Fragments are immutable in spirit here — new nodes are returned, the input is
    left untouched, so shared sub-trees (e.g. a reused coreference label) are safe.

    :param fragment: Root of the tree to transform.
    :param leaf: Transform applied to each leaf fragment (identity for unaffected leaves).
    :return: The rebuilt tree.
    :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
    """
    rebuilt = map_structural_children(fragment, lambda f: map_fragment(f, leaf))
    return rebuilt if rebuilt is not None else leaf(fragment)


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
