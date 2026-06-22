from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing_extensions import Dict, List, Optional, Set, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    relational_verb,
)
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_source_root,
    selected_aggregator,
)


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
    """
    ids: Set[uuid.UUID] = set()
    for node in expression._all_expressions_:
        if isinstance(node, Entity) and selected_aggregator(node) is not None:
            root = aggregation_source_root(node)
            if root is not None:
                ids.add(root._id_)
    return ids


def _numberable_type_name(node: SymbolicExpression) -> Optional[str]:
    """:return: The type name a node is disambiguated under — its own type for a (non-literal)
    variable, or the *value* type for a relational attribute (a verb-named hop such as
    ``assigned_to``, whose relative clause names a distinct entity that must be told apart from
    other same-type entities). ``None`` for anything that does not denote a numberable entity.
    """
    if isinstance(node, Variable) and not isinstance(node, Literal):
        return (
            node._type_.__name__
            if getattr(node, "_type_", None)
            else node.__class__.__name__
        )
    if (
        isinstance(node, Attribute)
        and relational_verb(node._attribute_name_) is not None
    ):
        value_type = getattr(node, "_type_", None)
        return value_type.__name__ if isinstance(value_type, type) else None
    return None


def _build_disambiguation_map(
    expression: SymbolicExpression,
) -> Tuple[Dict[uuid.UUID, str], Dict[uuid.UUID, str]]:
    """
    Types appearing once keep the plain type name; types appearing two or more times get
    "TypeName 1", "TypeName 2", … labels in encounter order. Both free variables and relational
    referents (the related entity a verb-named hop introduces) count toward a type's occurrences, so
    two distinct entities of one type are told apart wherever they appear. Literal nodes are
    excluded, as are variables that only serve as the source population of an aggregation sub-query.

    :param expression: Root expression to pre-scan.
    :return: ``(labels, numbered)`` — the full ``_id_`` → label map, and the subset whose label was
        actually numbered (the collisions).
    """
    if isinstance(expression, Query):
        expression.build()

    suppressed = _aggregation_source_ids(expression)

    type_to_ids: Dict[str, List[uuid.UUID]] = defaultdict(list)
    seen_ids: Set[uuid.UUID] = set()

    for node in expression._all_expressions_:
        type_name = _numberable_type_name(node)
        if type_name is None or node._id_ in suppressed or node._id_ in seen_ids:
            continue
        seen_ids.add(node._id_)
        type_to_ids[type_name].append(node._id_)

    labels: Dict[uuid.UUID, str] = {}
    numbered: Dict[uuid.UUID, str] = {}
    for type_name, ids in type_to_ids.items():
        if len(ids) == 1:
            labels[ids[0]] = type_name
            continue
        for ordinal, referent_id in enumerate(ids, 1):
            label = f"{type_name} {ordinal}"
            labels[referent_id] = label
            numbered[referent_id] = label
    return labels, numbered


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

    * Reiter, E. & Dale, R. (2000), "Building Natural Language Generation Systems", CUP —
      referring-expression generation as a microplanning subtask.
    * Dale, R. & Reiter, E. (1995), "Computational interpretations of the Gricean maxims in the
      generation of referring expressions", *Cognitive Science* 19(2).
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
        """
        labels, numbered = _build_disambiguation_map(expression)
        return cls(disambiguation_map=labels, numbered_labels=numbered)

    def numbered_label(self, variable: Variable) -> NumberedLabel:
        """Records *variable* as introduced.

        The label is the pre-computed disambiguation label (*"Robot 2"* for a colliding type),
        else the plain type name; *is_numbered* is whether they differ.

        :param variable: A variable instance.
        :return: The :class:`NumberedLabel` for *variable*.
        """
        type_name = (
            variable._type_.__name__
            if getattr(variable, "_type_", None)
            else variable.__class__.__name__
        )
        label = self.disambiguation_map.get(variable._id_, type_name)
        self.seen.add(variable._id_)
        return NumberedLabel(label, label != type_name)

    def noun_for_parts(self, variable: Variable) -> NounForm:
        """
        :param variable: A variable instance.
        :return: The first-mention :class:`NounForm` for *variable* — a numbered variable
            (*"Robot 2"*) is ``BARE``, any other is ``INDEFINITE`` (*"a Robot"*).
        """
        numbered = self.numbered_label(variable)
        definiteness = (
            Definiteness.BARE if numbered.is_numbered else Definiteness.INDEFINITE
        )
        return NounForm(definiteness, numbered.text)
