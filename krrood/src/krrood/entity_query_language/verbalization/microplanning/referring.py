from __future__ import annotations

import operator
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing_extensions import Dict, List, Optional, Set, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.value_lexicon import type_noun
from krrood.entity_query_language.verbalization.relational_attributes import (
    relational_verb,
)
from krrood.entity_query_language.query.aggregation_structure import (
    aggregation_source_root,
    selected_aggregator,
)


@dataclass
class _CanonicalReferentGrouping:
    """Numberable referent ids grouped under their canonical entity (``==``-unified referents
    collapse to one), recording each type's canonicals in encounter order so they can be numbered.
    """

    canonicals_by_type: Dict[str, List[uuid.UUID]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """Each type's canonical entities, in first-encounter order — a type with two or more is numbered."""

    members_by_canonical: Dict[uuid.UUID, List[uuid.UUID]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """Every original referent id that maps to a given canonical entity."""

    _type_of_canonical: Dict[uuid.UUID, str] = field(default_factory=dict)
    """The type a canonical was first registered under (guards against re-listing it)."""

    def add(self, referent_id: uuid.UUID, canonical: uuid.UUID, type_name: str) -> None:
        """Record *referent_id* as a member of *canonical*, registering *canonical* under *type_name*
        on first sight."""
        if canonical not in self._type_of_canonical:
            self._type_of_canonical[canonical] = type_name
            self.canonicals_by_type[type_name].append(canonical)
        self.members_by_canonical[canonical].append(referent_id)

    def labels(self) -> Tuple[Dict[uuid.UUID, str], Dict[uuid.UUID, str]]:
        """:return: ``(labels, numbered)`` — every referent id mapped to its display label, and the
        subset whose label was numbered (its type had several distinct canonicals)."""
        labels: Dict[uuid.UUID, str] = {}
        numbered: Dict[uuid.UUID, str] = {}
        for type_name, canonicals in self.canonicals_by_type.items():
            is_numbered = len(canonicals) > 1
            for ordinal, canonical in enumerate(canonicals, 1):
                label = f"{type_name} {ordinal}" if is_numbered else type_name
                for member in self.members_by_canonical[canonical]:
                    labels[member] = label
                    if is_numbered:
                        numbered[member] = label
        return labels, numbered


@dataclass(frozen=True)
class NumberedLabel:
    """A variable's disambiguation label and whether it was numbered (*"Robot"* vs *"Robot 2"*)."""

    text: str
    """The display label — the plain type name, or a numbered *"TypeName N"* on a collision."""

    is_numbered: bool
    """``True`` when the label was disambiguated with a number (so it renders ``BARE``)."""


@dataclass(frozen=True)
class NounForm:
    """The first-mention determiner form for a variable noun: its definiteness and label."""

    definiteness: Definiteness
    """``BARE`` for a numbered label, else ``INDEFINITE`` (*"a Robot"*)."""

    label: str
    """The display label for the noun head."""


@dataclass
class ReferringExpressions:
    """
    Pre-computed referring-expression state for a verbalization pass: the disambiguation labels
    (numbering colliding types) and the set of referents already introduced.

    This is the referring-expression generation subtask of microplanning: deciding between an
    indefinite first mention (*"a Robot"*), a definite subsequent mention (*"the Robot"*), a
    numbered form when one type occurs several times (*"Robot 1"* / *"Robot 2"*), and a pronoun
    (*"its …"*) when a chain is rooted at the current discourse subject.

    References:

    * :cite:t:`reiter2000building` — referring-expression generation as a microplanning subtask.
    * :cite:t:`dale1995gricean` — interpreting the Gricean maxims in referring-expression generation.
    """

    seen: Set[uuid.UUID] = field(default_factory=set)
    """Referent ``_id_`` s introduced so far, used to seed coreference across builds sharing this
    context, so verbalizing the same expression twice reads *"a Robot"* then *"the Robot"*."""

    disambiguation_map: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Maps variable ``_id_`` → display label, pre-computed before verbalization
    begins.  Single-type variables keep the plain type name; colliding types get
    ``"TypeName 1"``, ``"TypeName 2"`` labels."""

    numbered_labels: Dict[uuid.UUID, str] = field(default_factory=dict)
    """The subset of :attr:`disambiguation_map` whose label was actually numbered (the collisions).
    Relational referents carry no rule-resolved label — their relative-clause noun phrase is built
    deep in the microplanner — so the coreference pass applies these to number them *"Robot 1"*."""

    @classmethod
    def from_expression(cls, expression: SymbolicExpression) -> ReferringExpressions:
        """
        :param expression: Root EQL expression or query to scan.
        :return: An instance with the disambiguation map pre-built for *expression*.

        >>> first, second = variable(Robot, []), variable(Robot, [])
        >>> referring = ReferringExpressions.from_expression(an(set_of(first, second)))
        >>> referring.disambiguation_map[first._id_]
        'Robot 1'
        """
        labels, numbered = cls._build_disambiguation_map(expression)
        return cls(disambiguation_map=labels, numbered_labels=numbered)

    @classmethod
    def _build_disambiguation_map(
        cls, expression: SymbolicExpression
    ) -> Tuple[Dict[uuid.UUID, str], Dict[uuid.UUID, str]]:
        """
        Referents are counted by *canonical entity*: an ``==`` constraint that identifies two
        referents (``m.assigned_to == r``) collapses them to one, so the shared entity is named once
        (*"a Robot"*, not *"Robot 1"* / *"Robot 2"*). A canonical entity appearing once keeps the
        plain type name; a type with two or more distinct canonicals gets "TypeName 1", "TypeName 2",
        … in encounter order, and every original id of a canonical maps to its label. Both free
        variables and relational referents (the related entity a verb-named hop introduces) count;
        literal nodes are excluded, as are variables that only serve as an aggregation source.

        :param expression: Root expression to pre-scan.
        :return: ``(labels, numbered)`` — the full ``_id_`` → label map, and the subset whose label
            was actually numbered (the collisions).

        >>> labels, numbered = ReferringExpressions._build_disambiguation_map(
        ...     an(set_of(variable(Robot, []), variable(Robot, []))))
        >>> sorted(labels.values()), sorted(numbered.values())
        (['Robot 1', 'Robot 2'], ['Robot 1', 'Robot 2'])
        """
        if isinstance(expression, Query):
            expression.build()
        return cls._group_referents_by_canonical(expression).labels()

    @classmethod
    def _group_referents_by_canonical(
        cls, expression: SymbolicExpression
    ) -> _CanonicalReferentGrouping:
        """:return: Every numberable referent grouped under its canonical entity, in type-then-
        encounter order. Literal nodes, already-seen ids, and aggregation sources are skipped; an
        ``==``-unified pair shares one canonical (so it is named once)."""
        suppressed = cls._aggregation_source_ids(expression)
        aliases = referent_aliases(expression)
        grouping = _CanonicalReferentGrouping()
        seen: Set[uuid.UUID] = set()
        for node in expression._all_expressions_:
            type_name = cls._numberable_type_name(node)
            if type_name is None or node._id_ in suppressed or node._id_ in seen:
                continue
            seen.add(node._id_)
            grouping.add(node._id_, aliases.get(node._id_, node._id_), type_name)
        return grouping

    @staticmethod
    def _aggregation_source_ids(expression: SymbolicExpression) -> Set[uuid.UUID]:
        """
        Such a variable denotes a population to aggregate over, not a specific entity, so it must not
        consume an entity-disambiguation number — otherwise the outer subject would pick up a
        spurious *"1"* with no matching *"2"*, and a constrained aggregation scope would read *"among
        BankTransaction 2"* rather than *"among BankTransactions"*.

        :param expression: Root expression to scan.
        :return: The ``_id_`` of every variable that serves as the source population of an
            aggregation sub-query (e.g. the ``BankTransaction`` behind ``max(t.amount_details.amount)``),
            to exclude from numbering.

        >>> transaction = variable(BankTransaction, [])
        >>> source = an(entity(max(transaction.amount_details.amount)))
        >>> transaction._id_ in ReferringExpressions._aggregation_source_ids(source)
        True
        """
        ids: Set[uuid.UUID] = set()
        for node in expression._all_expressions_:
            if isinstance(node, Entity) and selected_aggregator(node) is not None:
                root = aggregation_source_root(node)
                if root is not None:
                    ids.add(root._id_)
        return ids

    @staticmethod
    def _numberable_type_name(node: SymbolicExpression) -> Optional[str]:
        """:return: The type name a node is disambiguated under — its own type for a (non-literal)
        variable, or the *value* type for a relational attribute (a verb-named hop such as
        ``assigned_to``, whose relative clause names a distinct entity that must be told apart from
        other same-type entities). ``None`` for anything that does not denote a numberable entity.

        >>> ReferringExpressions._numberable_type_name(variable(Robot, []))
        'Robot'
        """
        if isinstance(node, Variable) and not isinstance(node, Literal):
            return ReferringExpressions._variable_type_label(node)
        if (
            isinstance(node, Attribute)
            and relational_verb(node._attribute_name_) is not None
        ):
            value_type = node._type_
            return type_noun(value_type) if isinstance(value_type, type) else None
        return None

    @staticmethod
    def _variable_type_label(variable: Variable) -> str:
        """:return: The display noun for *variable*'s type — the friendly type noun when it carries a
        ``_type_`` (*"Integer"* for ``int``), else its own class name."""
        return (
            type_noun(variable._type_)
            if variable._type_
            else variable.__class__.__name__
        )

    def numbered_label(self, variable: Variable) -> NumberedLabel:
        """Records *variable* as introduced.

        The label is the pre-computed disambiguation label (*"Robot 2"* for a colliding type),
        else the plain type name; *is_numbered* is whether they differ.

        :param variable: A variable instance.
        :return: The :class:`NumberedLabel` for *variable*.

        >>> first, second = variable(Robot, []), variable(Robot, [])
        >>> referring = ReferringExpressions.from_expression(an(set_of(first, second)))
        >>> referring.numbered_label(second).text
        'Robot 2'
        """
        type_name = self._variable_type_label(variable)
        label = self.disambiguation_map.get(variable._id_, type_name)
        self.seen.add(variable._id_)
        return NumberedLabel(label, label != type_name)

    def noun_for_parts(self, variable: Variable) -> NounForm:
        """
        :param variable: A variable instance.
        :return: The first-mention :class:`NounForm` for *variable* — a numbered variable
            (*"Robot 2"*) is ``BARE``, any other is ``INDEFINITE`` (*"a Robot"*).

        >>> ReferringExpressions().noun_for_parts(variable(Robot, [])).label
        'Robot'
        """
        numbered = self.numbered_label(variable)
        definiteness = (
            Definiteness.BARE if numbered.is_numbered else Definiteness.INDEFINITE
        )
        return NounForm(definiteness, numbered.text)


def _entity_referent_id(node: SymbolicExpression) -> Optional[uuid.UUID]:
    """:return: the referent id of an entity-denoting node — a free variable, or a relational hop
    (the related entity it introduces) — else ``None``. The same notion of a numberable referent
    :meth:`ReferringExpressions._numberable_type_name` uses, so identity and numbering agree on what
    counts as an entity."""
    if ReferringExpressions._numberable_type_name(node) is None:
        return None
    return node._id_


def referent_aliases(expression: SymbolicExpression) -> Dict[uuid.UUID, uuid.UUID]:
    """
    :param expression: Root expression to scan.
    :return: A map from each entity referent id that participates in an identity to its canonical
        id. An ``==`` constraint between two entity referents (``m.assigned_to == r``) makes them one
        entity, so numbering and coreference can treat the pair as a single referent (*"a Robot"*,
        not *"Robot 1"* / *"Robot 2"*). Referents in no identity do not appear (each is its own
        canonical).

    >>> robot, mission = variable(Robot, []), variable(Mission, [])
    >>> aliases = referent_aliases(and_(mission.assigned_to == robot, mission.priority > 2))
    >>> len(set(aliases.values()))  # the relation and the variable collapse to one entity
    1
    """
    parent: Dict[uuid.UUID, uuid.UUID] = {}

    def find(node_id: uuid.UUID) -> uuid.UUID:
        parent.setdefault(node_id, node_id)
        root = node_id
        while parent[root] != root:
            root = parent[root]
        while parent[node_id] != root:
            parent[node_id], node_id = root, parent[node_id]
        return root

    def union(first: uuid.UUID, second: uuid.UUID) -> None:
        parent[find(second)] = find(first)

    for node in expression._all_expressions_:
        if isinstance(node, Comparator) and node.operation is operator.eq:
            left = _entity_referent_id(node.left)
            right = _entity_referent_id(node.right)
            if left is not None and right is not None:
                union(left, right)
    return {node_id: find(node_id) for node_id in parent}
