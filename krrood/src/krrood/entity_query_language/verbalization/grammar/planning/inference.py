"""
Inference-rule **planner** — pure structural analysis of an
:class:`~krrood.entity_query_language.query.query.Entity` whose selected variable is an
:class:`~krrood.entity_query_language.core.variable.InstantiatedVariable` (an inference
rule) into a :class:`RuleStructure` (the IF/THEN decomposition).

This is the analysis half of the planner/assembler split (see
:class:`~krrood.entity_query_language.verbalization.grammar.planning.base.Planner`):
it produces a plain data record of decisions and never builds fragments, touches the
context, or recurses.  The realisation lives in
:class:`~krrood.entity_query_language.verbalization.grammar.assembly.inference.InferenceAssembler`.

Reference: Reiter & Dale (2000) — content/structure determination (microplanning).
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import Any, FrozenSet, List, Optional, Tuple

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import AND
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.chain_utils import chain_root
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    attribute_names,
)
from krrood.entity_query_language.verbalization.grammar.planning.base import Planner


class AggregationStatus(Enum):
    """
    How a consequent binding or antecedent relates to the GROUP BY clause.

    :cvar GROUP_KEY: This expression is one of the ``grouped_by`` key variables.
    :cvar AGGREGATED: Present in the query but not a group key — rendered in plural form.
    :cvar NONE: No grouping context in this query.
    """

    GROUP_KEY = auto()
    AGGREGATED = auto()
    NONE = auto()


@dataclass
class ConditionPlan:
    """One antecedent WHERE condition, with the foldability decided up front.

    ``whose_attr`` is the (singular) attribute name when the condition is a single-hop
    attribute equality that folds into a *"whose <attr> is …"* modifier, else ``None``
    (render the condition normally).  Deciding this here keeps the assembler free of
    analysis.
    """

    expression: Any
    """The raw EQL condition expression."""

    whose_attr: Optional[str]
    """Singular attribute name when foldable to *whose*, else ``None``."""


@dataclass
class AntecedentInfo:
    """Descriptor for one antecedent variable in the IF clause."""

    root: Any
    """The underlying Variable/Entity (unwrapped from any ResultQuantifier)."""

    type_name: str
    """Human-readable Python type name of *root* (e.g. ``"Robot"``)."""

    aggregation_status: AggregationStatus
    """Whether this antecedent is a group key, aggregated, or neither."""

    conditions: List[ConditionPlan] = field(default_factory=list)
    """All WHERE conditions attributable to this antecedent (foldability pre-decided)."""


@dataclass
class ConsequentBinding:
    """Descriptor for one field binding in the THEN clause."""

    field_name: str
    """Python attribute name on the consequent type (e.g. ``"tasks"``)."""

    value_expression: Any
    """EQL expression providing the value for *field_name*."""

    is_plural_field: bool
    """``True`` when *field_name* is already plural."""

    aggregation_status: AggregationStatus
    """Whether the value is a group key, aggregated, or neither."""


@dataclass
class RuleStructure:
    """Complete decomposition of an inference-rule Entity query (the plan)."""

    primary_antecedents: List[AntecedentInfo]
    """Antecedents with at least one condition — items in the IF block."""

    secondary_antecedents: List[AntecedentInfo]
    """Antecedents with no conditions — only registered for coreference."""

    consequent_type: str
    """Python type name of the inferred variable (e.g. ``"Drawer"``)."""

    consequent_bindings: List[ConsequentBinding]
    """Ordered field bindings for the THEN clause."""

    unmatched_conditions: List[Any]
    """Outer WHERE conditions not attributable to any antecedent."""

    group_key_ids: FrozenSet[uuid.UUID]
    """``_id_`` values of the GROUP BY key variables."""


@dataclass
class InferencePlanner(Planner[Entity, RuleStructure]):
    """
    Decompose an inference-rule :class:`~krrood.entity_query_language.query.query.Entity`
    into a :class:`RuleStructure`.

    Algorithm: collect GROUP BY key ids; walk each consequent field binding to build
    :class:`ConsequentBinding` entries and discover antecedent roots; attribute outer
    WHERE conditions to antecedents by matching the left-hand root id; split antecedents
    into primary (have conditions) and secondary (none).
    """

    @staticmethod
    def can_handle(entity) -> bool:
        """Return ``True`` when *entity*'s selected variable is an InstantiatedVariable."""
        entity.build()
        return isinstance(entity.selected_variable, InstantiatedVariable)

    def plan(self) -> RuleStructure:
        self.node.build()
        group_key_ids = self._group_key_ids()
        antecedents, unmatched = self._plan_antecedents(group_key_ids)
        return RuleStructure(
            primary_antecedents=[a for a in antecedents if a.conditions],
            secondary_antecedents=[a for a in antecedents if not a.conditions],
            consequent_type=self._consequent_type(),
            consequent_bindings=self._plan_consequent(group_key_ids),
            unmatched_conditions=unmatched,
            group_key_ids=group_key_ids,
        )

    # ── shared analysis helpers ──────────────────────────────────────────────────

    @property
    def _inferred(self) -> InstantiatedVariable:
        """The InstantiatedVariable selected by the entity (after build)."""
        return self.node.selected_variable

    def _consequent_type(self) -> str:
        inferred = self._inferred
        return getattr(inferred._type_, "__name__", str(inferred._type_))

    def _group_key_ids(self) -> FrozenSet[uuid.UUID]:
        grouped = self.node._grouped_by_expression_
        if grouped is not None and grouped.variables_to_group_by:
            return frozenset(v._id_ for v in grouped.variables_to_group_by)
        return frozenset()

    @staticmethod
    def _aggregation_status(
        node_id, group_key_ids: FrozenSet[uuid.UUID]
    ) -> AggregationStatus:
        """GROUP_KEY if a group key, else AGGREGATED when grouping is present, else NONE."""
        if node_id in group_key_ids:
            return AggregationStatus.GROUP_KEY
        return AggregationStatus.AGGREGATED if group_key_ids else AggregationStatus.NONE

    # ── consequent (THEN bindings) ───────────────────────────────────────────────

    def _plan_consequent(
        self, group_key_ids: FrozenSet[uuid.UUID]
    ) -> List[ConsequentBinding]:
        return [
            ConsequentBinding(
                field_name=field_name,
                value_expression=child,
                is_plural_field=morphology.is_plural(field_name),
                aggregation_status=self._aggregation_status(child._id_, group_key_ids),
            )
            for field_name, child in self._inferred._child_vars_.items()
        ]

    # ── antecedents (IF roots + their conditions) ────────────────────────────────

    def _plan_antecedents(
        self, group_key_ids: FrozenSet[uuid.UUID]
    ) -> Tuple[List[AntecedentInfo], List[Any]]:
        """Discover antecedent roots, then attribute outer-WHERE conditions to them.

        Returns the antecedents (their ``conditions`` mutated in place) and the
        conditions that matched no antecedent.
        """
        antecedents = self._discover_antecedents(group_key_ids)
        unmatched = self._attribute_conditions(antecedents, self._outer_conditions())
        return antecedents, unmatched

    def _discover_antecedents(
        self, group_key_ids: FrozenSet[uuid.UUID]
    ) -> List[AntecedentInfo]:
        seen_root_ids: dict = {}
        for child in self._inferred._child_vars_.values():
            root = self._find_root(child)
            if root is None or root._id_ in seen_root_ids:
                continue
            type_name, own_conditions = self._extract_root_info(root)
            seen_root_ids[root._id_] = AntecedentInfo(
                root=root,
                type_name=type_name,
                aggregation_status=self._aggregation_status(root._id_, group_key_ids),
                conditions=own_conditions,
            )
        return list(seen_root_ids.values())

    def _outer_conditions(self) -> List[Any]:
        where = self.node._where_expression_
        return self._flatten_and(where.condition) if where is not None else []

    def _attribute_conditions(
        self, antecedents: List[AntecedentInfo], extra_conditions: List[Any]
    ) -> List[Any]:
        """Distribute outer-WHERE conditions to owning antecedents (in place); return unmatched."""
        id_to_antecedent = {self._antecedent_var_id(a): a for a in antecedents}
        unmatched: List[Any] = []
        for condition in extra_conditions:
            owner_id = self._condition_left_owner_id(condition)
            if owner_id is not None and owner_id in id_to_antecedent:
                id_to_antecedent[owner_id].conditions.append(self._planned(condition))
            else:
                unmatched.append(condition)
        return unmatched

    def _antecedent_var_id(self, antecedent: AntecedentInfo) -> Optional[object]:
        """Stable ``_id_`` of the underlying variable for an antecedent."""
        root = antecedent.root
        if isinstance(root, Entity):
            root.build()
            return getattr(root.selected_variable, "_id_", None)
        return getattr(root, "_id_", None)

    def _condition_left_owner_id(self, condition) -> Optional[object]:
        """``_id_`` of the root variable on the LHS of an equality condition, else ``None``."""
        if (
            not isinstance(condition, Comparator)
            or condition.operation is not operator.eq
        ):
            return None
        current = chain_root(condition.left)
        while isinstance(current, ResultQuantifier):
            current = current._child_
        return getattr(current, "_id_", None)

    def _find_root(self, expression) -> Optional[Any]:
        current = chain_root(expression)
        while isinstance(current, ResultQuantifier):
            current = current._child_
        if isinstance(current, (Variable, Entity)):
            return current
        return None

    def _extract_root_info(self, root) -> Tuple[str, List[ConditionPlan]]:
        """Return ``(type_name, own_conditions)`` for a root Variable or Entity."""
        if isinstance(root, Entity):
            root.build()
            var = root.selected_variable
            type_name = (
                var._type_.__name__
                if var and getattr(var, "_type_", None)
                else "entity"
            )
            conditions: List[ConditionPlan] = []
            if root._where_expression_ is not None:
                conditions = [
                    self._planned(e)
                    for e in self._flatten_and(root._where_expression_.condition)
                ]
            return type_name, conditions
        if isinstance(root, Variable):
            type_name = (
                root._type_.__name__ if getattr(root, "_type_", None) else "variable"
            )
            return type_name, []
        return "entity", []

    def _flatten_and(self, expression) -> List[Any]:
        """Recursively flatten a nested AND tree into a flat list of conjuncts."""
        if isinstance(expression, AND):
            return self._flatten_and(expression.left) + self._flatten_and(
                expression.right
            )
        return [expression]

    # ── condition foldability (the *whose* analysis — decided here, not in assembly) ──

    def _planned(self, condition) -> ConditionPlan:
        """Wrap a raw condition with its pre-decided *whose*-foldability."""
        return ConditionPlan(
            expression=condition, whose_attr=self._whose_attr(condition)
        )

    def _whose_attr(self, condition) -> Optional[str]:
        """Singular attribute name when *condition* folds to *"whose <attr> is …"*, else ``None``.

        Foldable iff it is an attribute equality (``Attribute == value``); the modifier
        attribute is the last hop of the attribute chain.
        """
        if (
            not isinstance(condition, Comparator)
            or condition.operation is not operator.eq
        ):
            return None
        if not isinstance(condition.left, Attribute):
            return None
        attr_names = attribute_names(condition.left)
        return attr_names[-1] if attr_names else None
