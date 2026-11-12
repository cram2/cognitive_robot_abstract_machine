"""
Corner-case provenance store for the EQL-native RDR.

Every rule in an RDR is created to handle one specific exception case — the *corner
case*. This module provides :class:`CornerCaseStore`, which records that case against
the rule's condition node and survives the save/load round-trip via a stable positional
index (see :func:`~krrood.entity_query_language.rdr.serialization.walk_rules_in_emission_order`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Any
from uuid import UUID

from krrood.code_generation.dataclass_source import (
    DataclassConstructorSerializer,
    ObjectSource,
    ObjectSourceSerializer,
)
from krrood.entity_query_language.core.base_expressions import SymbolicExpression

# %% CornerCaseStore


@dataclass
class CornerCaseStore:
    """
    Maps each rule's condition-node id to the case instance that triggered it.
    """

    cases: dict[UUID, Any] = field(default_factory=dict)
    """
    Live in-memory mapping from condition-node ``_id_`` to corner case instance.
    """

    serializer: ObjectSourceSerializer = field(
        default_factory=DataclassConstructorSerializer
    )
    """
    Pluggable serialization strategy.

    Default: :class:`~krrood.code_generation.dataclass_source.DataclassConstructorSerializer`.
    """

    def record_corner_case(self, node: SymbolicExpression, case: Any) -> None:
        """
        Record ``case`` as the corner case for the rule whose condition is ``node``.

        :param node: The condition node of the newly created rule.
        :param case: The concrete case instance that triggered the rule's creation.
        """
        self.cases[node._id_] = case

    def get_corner_case_for_node(self, node_id: UUID | None) -> Any | None:
        """
        Return the corner case recorded for ``node_id``, or ``None`` if absent.

        :param node_id: The ``_id_`` of a rule's condition node, or ``None``.
        :return: The recorded corner case, or ``None``.
        """
        if node_id is None:
            return None
        return self.cases.get(node_id)

    def to_ordered_sources(
        self,
        ordered_nodes: list[SymbolicExpression],
    ) -> dict[int, ObjectSource]:
        """
        Emit constructor source for every node that has a recorded corner case.

        Delegates to ``self.serializer.to_source`` for each recorded case.

        :param ordered_nodes: Rule condition nodes in the order returned by
            :func:`~krrood.entity_query_language.rdr.serialization.walk_rules_in_emission_order`.
        :return: Mapping from index to :class:`~krrood.code_generation.dataclass_source.ObjectSource` for nodes that have a
            recorded corner case; nodes without one are absent.
        """
        result: dict[int, ObjectSource] = {}
        for index, node in enumerate(ordered_nodes):
            case = self.cases.get(node._id_)
            if case is not None:
                result[index] = self.serializer.to_source(case)
        return result

    @classmethod
    def from_ordered_cases(
        cls,
        ordered_nodes: list[SymbolicExpression],
        cases_by_index: dict[int, Any],
    ) -> CornerCaseStore:
        """
        Rebuild a store from a positional index map loaded from a saved file.

        :param ordered_nodes: Rule condition nodes in the same emission order used at
            save time (from
            :func:`~krrood.entity_query_language.rdr.serialization.walk_rules_in_emission_order`
            over the freshly loaded rule tree).
        :param cases_by_index: Mapping from positional index to case instance (as loaded
            from the ``RDR_CORNER_CASES`` module-level dict).
        :return: A new :class:`CornerCaseStore` keyed by node ``_id_``.
        """
        store = cls()
        for index, node in enumerate(ordered_nodes):
            if index in cases_by_index:
                store.cases[node._id_] = cases_by_index[index]
        return store
