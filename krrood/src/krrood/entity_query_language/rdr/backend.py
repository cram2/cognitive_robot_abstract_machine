"""
RDR backend for underspecified EQL queries.

Given an underspecified ``Match`` — ``...`` marks the attribute to infer, concrete kwargs
filter, an optional domain supplies the instances — this backend infers the ``...``
attribute on each matching instance using a single-class RDR, **lazily, one case at a
time**, mirroring ordinary EQL evaluation.

It keeps one :class:`EQLSingleClassRDR` per ``(case type, attribute)``. Asked to infer an
attribute it has no model for, it first falls into *fit mode* (using ground-truth targets
when given, otherwise asking the expert for both the conclusion and its conditions).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
    Union,
)

from krrood.entity_query_language.core.base_expressions import UnificationDict
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.underspecified import UnderspecifiedMatch
from krrood.entity_query_language.rdr.utils import UNSET

#: Ground truth: either a single conclusion shared by every case, or a per-case callable.
GroundTruth = Union[Any, Callable[[Any], Any]]

#: Registry key identifying the RDR model that infers one attribute of one type.
ModelKey = Tuple[Type, str]


def key_from_attribute(attribute: Attribute) -> ModelKey:
    """:return: The registry key for an EQL attribute expression (e.g. ``animal.species``)."""
    return (attribute._child_._type_, attribute._attribute_name_)


@dataclass
class RDRBackend:
    """Infers underspecified (``...``) attributes on existing instances via RDR models."""

    expert: Optional[Expert] = None
    """Authors rule conditions (and, in fit mode without ground truth, conclusions too)."""
    models: Dict[ModelKey, EQLSingleClassRDR] = field(default_factory=dict)
    """One single-class RDR per ``(case type, attribute)`` the backend has learned."""

    def fit(
        self, query: Any, ground_truth: Optional[GroundTruth] = None
    ) -> "RDRBackend":
        """
        Train the model for ``query``'s ``...`` attribute over the filtered domain.

        :param query: An underspecified ``Match`` (with a domain) whose ``...`` slot to train.
        :param ground_truth: A single conclusion for every case, or a ``case -> conclusion``
            callable. When ``None`` the expert labels each case (via ``ask_for_rule``).
        :return: self.
        """
        statement = UnderspecifiedMatch(query)
        rdr = self._get_or_create(statement)
        for case in statement.filtered_cases():
            rdr.fit_case(case, self._target_for(case, ground_truth), expert=self.expert)
        return self

    def infer(
        self,
        query: Any,
        ground_truth: Optional[GroundTruth] = None,
        fill_in_place: bool = False,
    ) -> Iterator[Any]:
        """
        Lazily infer ``query``'s ``...`` attribute on each filtered instance.

        If no model exists for the attribute yet, fit mode runs first (using
        ``ground_truth`` when given, otherwise the expert).

        :param query: An underspecified ``Match`` (with a domain).
        :param ground_truth: Used only if fit mode is triggered (see :meth:`fit`).
        :param fill_in_place: When ``True``, set the attribute on each instance and yield the
            instance. Otherwise (default) yield a :class:`UnificationDict` mapping the case
            variable to the instance and the target attribute to the inferred value.
        :return: A lazy iterator of :class:`UnificationDict` (default) or filled instances.
        """
        statement = UnderspecifiedMatch(query)
        key = self._key(statement)
        if key not in self.models:
            self.fit(query, ground_truth)
        rdr = self.models[key]
        target = statement.single_target()
        for case in statement.filtered_cases():
            value = rdr.classify(case)
            if fill_in_place:
                setattr(case, target.attribute_name, value)
                yield case
            else:
                yield UnificationDict(
                    {statement.variable: case, target.attribute: value}
                )

    def _get_or_create(self, statement: UnderspecifiedMatch) -> EQLSingleClassRDR:
        key = self._key(statement)
        if key not in self.models:
            self.models[key] = EQLSingleClassRDR(
                statement.case_type, statement.target_attribute_name
            )
        return self.models[key]

    @staticmethod
    def _key(statement: UnderspecifiedMatch) -> ModelKey:
        return key_from_attribute(statement.single_target().attribute)

    @staticmethod
    def _target_for(case: Any, ground_truth: Optional[GroundTruth]) -> Optional[Any]:
        if ground_truth is None:
            return UNSET
        if callable(ground_truth):
            return ground_truth(case)
        return ground_truth
