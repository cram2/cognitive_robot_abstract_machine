from __future__ import annotations

import uuid
from dataclasses import dataclass, is_dataclass, replace

from typing_extensions import TYPE_CHECKING, Callable, List, Optional

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    FlatVariable,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.relational_attributes import (
        RelationVerb,
    )


@dataclass(frozen=True)
class RelationStep:
    """The data a *relational* hop renders from — the related type (the relative clause's head
    noun), the verb split into participle + preposition (so the preposition pied-pipes before
    *"which"*), and the owner class (for the verb's source link)."""

    value_type: Optional[type]
    """The hop's value type — the head noun of the relative clause (*"the Robot"*)."""

    owner_class: type
    """The attribute's owner class — supplies the verb's source link."""

    participle: str
    """The past-participle verb word(s) (*"assigned"*) — the *"… is assigned"* tail."""

    preposition: str
    """The trailing preposition (*"to"*) — pied-piped to *"to which …"*."""

    referent_id: Optional[uuid.UUID] = None
    """The navigated value node's id. The relative clause is a *referring* expression for that
    entity, so a repeat mention of the same navigation reduces to a bare *"the <Type>"* (coreference)
    rather than repeating the whole *"the Robot to which it is assigned"*; ``None`` disables it."""

    is_agentive: bool = False
    """``True`` for an agentive relation (preposition *"by"*), read in the active voice (*"the Person
    who owns a Book"*) instead of the passive relative clause (*"the Person by which a Book is
    owned"*)."""

    active_verb: str = ""
    """The active present verb (*"owns"*) used for the active-voice reading of an agentive relation;
    unused for a non-agentive relation."""


@dataclass(frozen=True)
class PathStep:
    """
    One hop of a navigation chain — a display name and the source reference it links to.

    Replaces a bare ``(name, source_reference)`` tuple so the two halves are named (``.name`` /
    ``.source_reference``) rather than positional.
    """

    name: str
    """The display text for this hop (e.g. ``"amount"``, ``"handle[0]"``, ``"()"``)."""

    source_reference: Optional[SourceReference] = None
    """The attribute's source reference, or ``None`` for composite / index / call hops."""

    relation: Optional[RelationStep] = None
    """When set, this hop is a *relation* named as a verb (``assigned_to``) and renders as a relative
    clause *"the <Type> which <owner> is <verb-phrase>"* instead of the genitive *"the <attr> of
    <owner>"*; ``None`` for a plain noun hop."""

    is_scalar_value: bool = False
    """``True`` when this hop's value is an atomic scalar (a number / string / date), not an entity.
    A scalar leaf possessed by a plural subject distributes (*"their salaries"*); an entity owner of
    further structure does not (*"the begin and end of their period"*)."""

    ordinal: Optional[str] = None
    """The ordinal phrase (*"first"*, *"last"*) when this attribute hop selects a single indexed
    element of a collection: the collection noun singularizes and takes the ordinal (*"the first
    task"* rather than *"the first of the tasks"*); ``None`` for a whole-collection hop."""

    @property
    def is_plain_attribute(self) -> bool:
        """:return: ``True`` when this hop is a plain genitive attribute — it links to a source
        attribute and is not a relative-clause relation, so an integer index can fold its ordinal
        into it (*"the first task"*)."""
        return self.source_reference is not None and self.relation is None

    @property
    def is_relation(self) -> bool:
        """:return: ``True`` when this hop renders as a relative clause rather than a genitive.

        >>> from krrood.entity_query_language.core.expression_structure import walk_chain
        >>> from krrood.entity_query_language.verbalization.relational_attributes import relational_verb
        >>> chain, _ = walk_chain(variable(Mission, []).assigned_to)
        >>> build_path_parts(chain, relational_verb)[0].is_relation
        True
        >>> build_path_parts(walk_chain(variable(Robot, []).battery)[0])[0].is_relation
        False
        """
        return self.relation is not None


def build_path_parts(
    chain: List[MappedVariable],
    relation_verb: Optional[Callable[[str], Optional["RelationVerb"]]] = None,
) -> List[PathStep]:
    """
    Convert a walked chain into :class:`PathStep` hops.

    Hop rules:

    * ``Attribute`` nodes appear as the attribute name, linked to their source reference. When
      *relation_verb* recognises the name as a verb relation (returns its split verb), the hop is
      flagged relational, so it renders as a relative clause instead of a genitive.
    * An integer ``Index`` node folds its ordinal (``0`` → ``"first"``) into the collection attribute
      it subscripts, which singularizes: *"the first task of …"* rather than *"the first of the tasks
      of …"* or a raw subscript (``"tasks[0]"``). An index that does not follow a plain attribute
      (e.g. on a call result) stays a standalone ordinal hop. Non-integer keys keep the ``"[key]"``
      bracket form.
    * ``Call`` nodes appear as ``"()"`` with no source reference.
    * ``FlatVariable`` nodes are skipped.

    :param chain: Innermost-first chain list (nearest the root first).
    :param relation_verb: Optional name → split-verb recogniser, injected so this module stays
        decoupled from the grammar's recognizers; ``None`` disables relational rendering.
    :return: Ordered list of :class:`PathStep`, innermost hop first.

    >>> from krrood.entity_query_language.core.expression_structure import walk_chain
    >>> chain, _ = walk_chain(variable(BankTransaction, []).amount_details.amount)
    >>> [step.name for step in build_path_parts(chain)]
    ['amount_details', 'amount']
    """
    parts: List[PathStep] = []
    for node in chain:
        if isinstance(node, Attribute):
            owner = node._owner_class_
            name = node._attribute_name_
            verb = relation_verb(name) if relation_verb is not None else None
            relation = (
                RelationStep(
                    node._type_,
                    owner,
                    verb.participle,
                    verb.preposition,
                    node._id_,
                    is_agentive=verb.is_agentive,
                    active_verb=verb.active_verb,
                )
                if verb is not None
                else None
            )
            parts.append(
                PathStep(
                    name,
                    SourceReference.for_attribute(owner, name),
                    relation=relation,
                    is_scalar_value=_is_scalar_value(node._type_),
                )
            )
        elif isinstance(node, Index):
            _append_index(parts, node)
        elif isinstance(node, Call):
            parts.append(PathStep("()", None))
        elif isinstance(node, FlatVariable):
            pass
    return parts


def _is_scalar_value(value_type: Optional[type]) -> bool:
    """:return: ``True`` when *value_type* is an atomic scalar — a primitive value, not an entity
    (an entity is a dataclass and reads as an owner, not a distributable value).

    >>> _is_scalar_value(int), _is_scalar_value(Robot), _is_scalar_value(None)
    (True, False, False)
    """
    return value_type is not None and not is_dataclass(value_type)


def _append_index(parts: List[PathStep], node: Index) -> None:
    """Fold an integer index into the plain attribute hop it subscripts (*"the first task"*), else
    append it as a standalone hop.

    An integer index selects one element of the preceding collection attribute, so its ordinal folds
    into that hop and the noun singularizes. When the index does not follow a plain attribute (e.g.
    a call result) it becomes a standalone ordinal hop; a non-integer key keeps the ``"[key]"`` form.
    """
    key = node._key_
    if not isinstance(key, int) or isinstance(key, bool):
        parts.append(PathStep(f"[{repr(key)}]", None))
        return
    ordinal_phrase = morphology.index_ordinal(key)
    if parts and parts[-1].is_plain_attribute and parts[-1].ordinal is None:
        parts[-1] = replace(
            parts[-1],
            ordinal=ordinal_phrase,
            is_scalar_value=_is_scalar_value(node._type_),
        )
        return
    parts.append(PathStep(ordinal_phrase, None))
