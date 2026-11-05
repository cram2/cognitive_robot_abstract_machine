"""
Adapter from an underspecified EQL ``Match`` to RDR attribute inference.

An underspecified query marks the attributes to infer with ``...`` (Ellipsis) and may
carry concrete attribute constraints plus a domain of instances, e.g.::

    an(Animal)(hair=True, species=...).from_(animals)

This adapter reads that query: it locates the ``...`` inference targets, keeps the
concrete attributes as an ordinary EQL filter (the ellipsis conditions stripped out), and
streams the domain instances that pass the filter — the cases an RDR backend then fills.

Single-class RDR fills a single, scalar attribute; an ``...`` slot whose declared type is
an unbounded iterable (``list``/``set``/...) is rejected here and left to a future
``MultiClassRDR``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from typing_extensions import (
    Any,
    Iterator,
    List,
    Type,
)

from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import entity
from krrood.entity_query_language.query.match import AttributeMatch, Match
from krrood.entity_query_language.rdr.exceptions import (
    MultipleInferenceTargets,
    NoInferenceTarget,
    UndeclaredInferenceTarget,
    UnsupportedInferenceTarget,
)
from krrood.symbol_graph.helpers import get_wrapped_field


@dataclass(kw_only=True)
class RippleDownRulesMatchParser:
    """
    Reads an underspecified :class:`Match` for RDR-based attribute inference.
    """

    match: Match
    """
    The underspecified query whose ``...`` attributes are inferred.
    """

    def __post_init__(self) -> None:
        # The properties below read attributes that exist only on a resolved match.
        self.match.resolve()

    @property
    def case_type(self) -> Type:
        """
        The type whose instances are being completed (e.g. ``Animal``).
        """
        return self.match._type_

    @property
    def variable(self) -> Variable:
        """
        The EQL variable the query ranges over.
        """
        return self.match._variable_

    @cached_property
    def inference_targets(self) -> List[AttributeMatch]:
        """
        The ``...`` attribute leaves to infer (each validated as single-valued).
        """
        targets = [
            attribute_match
            for attribute_match in self.match._matches_with_variables_
            if attribute_match.is_assigned_to_ellipsis
        ]
        for target in targets:
            self._guard_single_valued(target=target)
        return targets

    def single_target(self) -> AttributeMatch:
        """:return: The sole inference target, enforcing the single-class invariant."""
        targets = self.inference_targets
        if not targets:
            raise NoInferenceTarget(case_type=self.case_type)
        if len(targets) > 1:
            raise MultipleInferenceTargets(
                attribute_names=[target.attribute_name for target in targets]
            )
        return targets[0]

    @property
    def target_attribute_name(self) -> str:
        """
        The name of the single attribute this query asks to infer.
        """
        return self.single_target().attribute_name

    def filtered_cases(self) -> Iterator[Any]:
        """
        Lazily yield the domain instances that satisfy the concrete (non-``...``)
        constraints — ordinary EQL evaluation, with the ellipsis conditions stripped.
        """
        query = entity(self.variable)
        conditions = self._concrete_conditions()
        if conditions:
            query = query.where(*conditions)
        return query.evaluate()

    def _concrete_conditions(self) -> List[Any]:
        """
        Fresh comparator nodes for the concrete attribute constraints (``...`` dropped).
        """
        conditions: List[Any] = []
        for leaf in self.match._matches_with_variables_:
            if leaf.is_assigned_to_ellipsis:
                continue
            attribute = getattr(self.variable, leaf.attribute_name)
            conditions.append(attribute == leaf.assigned_variable._value_)
        return conditions

    def _guard_single_valued(self, *, target: AttributeMatch) -> None:
        """
        :raises UndeclaredInferenceTarget: If ``target`` is not a declared field of the
            case type.
        :raises UnsupportedInferenceTarget: If ``target``'s field holds a collection.
        """
        wrapped_field = get_wrapped_field(self.case_type, target.attribute_name)
        if wrapped_field is None:
            raise UndeclaredInferenceTarget(case_type=self.case_type, attribute_name=target.attribute_name)
        if wrapped_field.holds_collection:
            raise UnsupportedInferenceTarget(case_type=self.case_type, attribute_name=target.attribute_name)
