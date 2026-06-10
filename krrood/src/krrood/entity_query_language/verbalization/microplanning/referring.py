"""
Referring-expression generation — choosing how to *name* an entity each time it
is mentioned.

This is the **referring-expression generation** subtask of microplanning
(Reiter & Dale 2000): deciding between an indefinite first mention (*"a Robot"*),
a definite subsequent mention (*"the Robot"*), a numbered form when one type
occurs several times (*"Robot 1"* / *"Robot 2"*), and a pronoun (*"its …"*) when a
chain is rooted at the current discourse subject.  All of these facets share one
piece of state — what has been mentioned so far — and change for one reason
(referring-expression policy), so they form a single cohesive responsibility.

References:

* Reiter, E. & Dale, R. (2000), "Building Natural Language Generation Systems",
  CUP — referring-expression generation as a microplanning subtask.
* Dale, R. & Reiter, E. (1995), "Computational interpretations of the Gricean
  maxims in the generation of referring expressions", *Cognitive Science* 19(2).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Dict, List, Set

from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_source_root,
    selected_aggregator,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression


def _aggregation_source_ids(expression) -> set:
    """
    Return the ``_id_`` of every variable that serves as the source *population*
    of an aggregation sub-query (e.g. the ``BankTransaction`` behind
    ``max(t.amount_details.amount)``).

    Such a variable denotes a population to aggregate over, not a specific
    entity, so it must not consume an entity-disambiguation number — otherwise
    the outer subject would pick up a spurious *"1"* with no matching *"2"*, and
    a constrained aggregation scope would read *"among BankTransaction 2"* rather
    than *"among BankTransactions"*.

    :param expression: Root expression to scan.
    :return: Set of variable ids to exclude from numbering.
    :rtype: set
    """
    ids: set = set()
    for node in expression._all_expressions_:
        if isinstance(node, Entity) and selected_aggregator(node) is not None:
            root = aggregation_source_root(node)
            if root is not None:
                ids.add(root._id_)
    return ids


def _build_disambiguation_map(expression) -> Dict[uuid.UUID, str]:
    """
    Pre-scan *expression* and return a mapping of variable._id_ → display label.

    Types appearing once keep the plain type name; types appearing two or more
    times get "TypeName 1", "TypeName 2", … labels in encounter order.
    Literal nodes are excluded, as are variables that only serve as the source
    population of an aggregation sub-query (see :func:`_aggregation_source_ids`).
    """
    if isinstance(expression, Query):
        expression.build()

    suppressed = _aggregation_source_ids(expression)

    type_to_ids: Dict[str, List[uuid.UUID]] = defaultdict(list)
    seen_ids: set = set()

    for node in expression._all_expressions_:
        if isinstance(node, Variable) and not isinstance(node, Literal):
            if node._id_ in suppressed:
                continue
            type_name = (
                node._type_.__name__
                if getattr(node, "_type_", None)
                else node.__class__.__name__
            )
            if node._id_ not in seen_ids:
                seen_ids.add(node._id_)
                type_to_ids[type_name].append(node._id_)

    result: Dict[uuid.UUID, str] = {}
    for type_name, ids in type_to_ids.items():
        if len(ids) == 1:
            result[ids[0]] = type_name
        else:
            for n, vid in enumerate(ids, 1):
                result[vid] = f"{type_name} {n}"
    return result


@dataclass
class ReferringExpressions:
    """
    Pre-computed referring-expression state for a verbalization pass.

    Since coreference resolution moved into the document-order
    :class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`,
    this holds only *pre-computed* / *cross-build* state: the disambiguation labels
    (:attr:`disambiguation_map`, numbering colliding types) and the set of referents already
    introduced (:attr:`seen`) — the latter solely so a second verbalization sharing this context
    is seeded as already-mentioned (the within-build first/subsequent decision is the pass's).
    """

    seen: "Set[uuid.UUID]" = field(default_factory=set)
    """Referent ``_id_`` s introduced so far — used only to seed the coreference pass across
    builds sharing this context (see :meth:`EQLVerbalizer.build`), so verbalizing the same
    expression twice on one context reads *"a Robot"* then *"the Robot"*.  The within-build
    first/subsequent decision is the pass's."""

    disambiguation_map: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Maps variable ``_id_`` → display label, pre-computed before verbalization
    begins.  Single-type variables keep the plain type name; colliding types get
    ``"TypeName 1"``, ``"TypeName 2"`` labels."""

    @classmethod
    def from_expression(cls, expression) -> ReferringExpressions:
        """Create an instance with the disambiguation map pre-built for *expression*."""
        return cls(disambiguation_map=_build_disambiguation_map(expression))

    def mark_introduced(self, expression) -> None:
        """Record *expression* as introduced (so a later build sharing this context seeds it)."""
        self.seen.add(expression._id_)

    def noun_for_parts(self, var) -> tuple[Definiteness, str]:
        """
        Return the **first-mention** ``(Definiteness, label)`` for *var* — a numbered variable
        (*"Robot 2"*) is ``BARE``, any other is ``INDEFINITE`` (*"a Robot"*).

        The *subsequent*-mention downgrade to definite is no longer decided here: the caller
        stamps the resulting :class:`~krrood.entity_query_language.verbalization.fragments.base.NounPhrase`
        with a ``referent_id`` and the
        :class:`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`
        downgrades repeats in document order (the discourse decision, in one place).  The mention
        is still recorded in :attr:`seen` so a later build sharing this context seeds it.

        :param var: A :class:`~krrood.entity_query_language.core.variable.Variable` instance.
        :return: Tuple of ``(first-mention Definiteness, display_label)``.
        :rtype: tuple
        """
        type_name = (
            var._type_.__name__
            if getattr(var, "_type_", None)
            else var.__class__.__name__
        )
        label = self.disambiguation_map.get(var._id_, type_name)
        is_numbered = label != type_name
        self.seen.add(var._id_)  # recorded for cross-build seeding (see :attr:`seen`)
        return (Definiteness.BARE if is_numbered else Definiteness.INDEFINITE), label
