"""
Asking the clutter circuit what makes a pick succeed.

Each question marks one cause and one effect in the query itself, grounds the relational
circuit against a clutter of the size the question asks about, and reads the effect off
every region of the cause twice: once by conditioning alone and once with backdoor
adjustment, so the two can be set side by side.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum

from krrood.entity_query_language.factories import a, cause, confounder, variable
from typing_extensions import Any, Dict, List, Tuple, Type

from experiments.causal_reasoning.do_query import (
    CausalQuestion,
    CauseStratification,
    DoQuery,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutteredObject,
    ClutterPickScene,
    ClutterPickSceneAggregations,
    PartField,
)

# %% what an attempt offers as a cause


class AttemptAttribute(StrEnum):
    """
    An attempt attribute a question names as its cause or adjusts for.
    """

    ENVIRONMENT = "environment"
    """
    The kind of environment the clutter stood in.
    """

    FRICTION_COEFFICIENT = "friction_coefficient"
    """
    Sliding friction of the grasp contact.
    """

    @property
    def noun(self) -> str:
        """
        What the attribute is, in words.
        """
        return {
            AttemptAttribute.ENVIRONMENT: "the environment",
            AttemptAttribute.FRICTION_COEFFICIENT: "the grasp friction",
        }[self]

    @property
    def circuit_variable_name(self) -> str:
        """
        The name a fitted circuit gives the attribute.
        """
        attempt = variable(ClutterPickScene)
        return {
            AttemptAttribute.ENVIRONMENT: attempt.environment,
            AttemptAttribute.FRICTION_COEFFICIENT: attempt.friction_coefficient,
        }[self]._name_


class NeighbourAttribute(StrEnum):
    """
    A neighbour attribute a question names as its cause.

    A part template names its columns by the bare attribute, so the member's value is
    also the column a fit groups by.
    """

    CLOSING_AXIS_SIDE = "closing_axis_side"
    """
    Whether it stands in the direction the fingers closed in.
    """

    @property
    def noun(self) -> str:
        """
        What the attribute is, in words.
        """
        return {
            NeighbourAttribute.CLOSING_AXIS_SIDE: (
                "which side of the closing axis it stands on"
            )
        }[self]


class AttemptCount(StrEnum):
    """
    An aggregation statistic counting over the clutter's neighbours.
    """

    CROWDING = "crowding_count"
    """
    How many neighbours stand adjacent to the target.
    """

    @property
    def noun(self) -> str:
        """
        What the statistic counts, in words.
        """
        return {AttemptCount.CROWDING: "adjacent neighbours"}[self]

    @property
    def circuit_variable_name(self) -> str:
        """
        The name a fitted circuit gives the statistic.
        """
        aggregations = variable(ClutterPickSceneAggregations)
        return {AttemptCount.CROWDING: aggregations.crowding_count()}[self]._name_


def stratification_of_attempt(attribute: AttemptAttribute) -> CauseStratification:
    """
    :param attribute: The attempt's own attribute acting as the cause.
    :return: The grouping for it.
    """
    return CauseStratification(column=attribute.circuit_variable_name)


def stratification_of_count(count: AttemptCount) -> CauseStratification:
    """
    :param count: The statistic over the neighbours acting as the cause.
    :return: The grouping for it.
    """
    return CauseStratification(column=count.circuit_variable_name)


def stratification_of_neighbour(
    attribute: NeighbourAttribute, part_field: PartField
) -> CauseStratification:
    """
    :param attribute: One neighbour's own attribute acting as the cause.
    :param part_field: The field those neighbours are held in.
    :return: The grouping for it, applied to that field's template.
    """
    return CauseStratification(column=attribute.value, part_field=part_field)


# %% the questions


@dataclass(frozen=True, kw_only=True)
class ClutterQuestion(CausalQuestion):
    """
    One question of the form: does this property of an attempt cause that, once the
    confounders are adjusted for?
    """

    neighbour_count: int
    """
    How many neighbours the queried clutter holds.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        What the question is called where answers are collected.
        """

    @property
    @abstractmethod
    def asked(self) -> str:
        """
        The question, in words.
        """

    @property
    @abstractmethod
    def stratification(self) -> CauseStratification:
        """
        How the fit is grouped so that the cause comes out support-deterministic.
        """

    @property
    @abstractmethod
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        """
        The names a fitted circuit gives the variables to adjust for.
        """

    @property
    @abstractmethod
    def effect_value(self) -> Any:
        """
        The value of the effect whose probability the question asks for.
        """

    @abstractmethod
    def build(self) -> Any:
        """
        The grounding query, with its cause and its effect marked.
        """

    def _neighbour(self, **specified: Any) -> Any:
        """
        :param specified: Neighbour attributes to mark or fix; the rest are left open.
        :return: A query for one neighbour.
        """
        attributes: Dict[str, Any] = {
            field.name: ... for field in fields(ClutteredObject)
        }
        attributes.update(specified)
        return a(ClutteredObject)(**attributes)

    def _attempt(self, neighbours: List[Any], **specified: Any) -> Any:
        """
        An attempt query leaving every attribute it does not mark open, so that
        grounding has to retain the statistics over the neighbours rather than integrate
        them out.

        :param neighbours: One query per neighbour of the clutter.
        :param specified: Attempt attributes to mark or fix; the rest are left open.
        :return: The attempt query.
        """
        parts: Dict[str, Any] = {PartField.NEIGHBOURS.value: neighbours}
        attributes: Dict[str, Any] = {
            field.name: ...
            for field in fields(ClutterPickScene)
            if field.name not in parts
        }
        attributes.update(specified)
        return a(ClutterPickScene)(**parts, **attributes)

    def _open_neighbours(self) -> List[Any]:
        """
        :return: One fully open query per neighbour of the clutter.
        """
        return [self._neighbour() for _ in range(self.neighbour_count)]


@dataclass(frozen=True, kw_only=True)
class FrictionCausesLift(ClutterQuestion):
    """
    Does the friction of the grasp cause the target to come up, once the environment is
    adjusted for?

    The environment decides both how slippery and how crowded an attempt is, which is
    what makes it a confounder rather than a nuisance.
    """

    @property
    def name(self) -> str:
        return f"friction_causes_lift_{self.neighbour_count}_neighbours"

    @property
    def asked(self) -> str:
        return (
            f"In a clutter of {self.neighbour_count} neighbours, does the grasp "
            "friction cause the target to be lifted, adjusting for the environment?"
        )

    @property
    def stratification(self) -> CauseStratification:
        return stratification_of_attempt(AttemptAttribute.FRICTION_COEFFICIENT)

    @property
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        return (AttemptAttribute.ENVIRONMENT.circuit_variable_name,)

    @property
    def effect_value(self) -> bool:
        return True

    def build(self) -> Any:
        query = self._attempt(
            self._open_neighbours(),
            **{
                AttemptAttribute.FRICTION_COEFFICIENT.value: cause,
                AttemptAttribute.ENVIRONMENT.value: confounder,
            },
        )
        query.causes_effect(query.lifted == True)
        return query


@dataclass(frozen=True, kw_only=True)
class CrowdingCausesLift(ClutterQuestion):
    """
    Does how many neighbours stand adjacent to the target cause it to come up?

    The cause counts over the exchangeable parts, so it is a question a model that only
    holds a fixed set of columns cannot be asked.
    """

    adjusted_for: Tuple[AttemptAttribute, ...] = (AttemptAttribute.ENVIRONMENT,)
    """
    The attempt attributes to adjust for.
    """

    @property
    def name(self) -> str:
        adjusting = (
            "_and_".join(attribute.value for attribute in self.adjusted_for)
            if self.adjusted_for
            else "nothing"
        )
        return (
            f"crowding_causes_lift_{self.neighbour_count}_neighbours"
            f"_adjusting_{adjusting}"
        )

    @property
    def asked(self) -> str:
        adjusting = (
            "adjusting for "
            + " and ".join(attribute.noun for attribute in self.adjusted_for)
            if self.adjusted_for
            else "adjusting for nothing"
        )
        return (
            f"In a clutter of {self.neighbour_count} neighbours, does how many of them "
            f"stand adjacent to the target cause it to be lifted, {adjusting}?"
        )

    @property
    def stratification(self) -> CauseStratification:
        return stratification_of_count(AttemptCount.CROWDING)

    @property
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        return tuple(attribute.circuit_variable_name for attribute in self.adjusted_for)

    @property
    def effect_value(self) -> bool:
        return True

    def build(self) -> Any:
        query = self._attempt(
            self._open_neighbours(),
            **{AttemptCount.CROWDING.value: cause},
            **{attribute.value: confounder for attribute in self.adjusted_for},
        )
        query.causes_effect(query.lifted == True)
        return query


@dataclass(frozen=True, kw_only=True)
class ClosingAxisSideCausesDisturbance(ClutterQuestion):
    """
    Does one neighbour standing where the fingers close cause that neighbour to be
    shoved aside?

    Cause and effect both live on one neighbour, so the question is about the clutter's
    structure rather than about the attempt as a whole.
    """

    neighbour_index: int = 0
    """
    Which of the listed neighbours the question is about.
    """

    @property
    def name(self) -> str:
        return (
            "closing_axis_side_causes_disturbance_of_neighbour_"
            f"{self.neighbour_index}_of_{self.neighbour_count}"
        )

    @property
    def asked(self) -> str:
        return (
            f"In a clutter of {self.neighbour_count} neighbours, does neighbour "
            f"{self.neighbour_index} standing along the fingers' closing axis cause it "
            "to be disturbed by the pick?"
        )

    @property
    def stratification(self) -> CauseStratification:
        return stratification_of_neighbour(
            NeighbourAttribute.CLOSING_AXIS_SIDE, PartField.NEIGHBOURS
        )

    @property
    def adjustment_variable_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def effect_value(self) -> bool:
        return True

    def build(self) -> Any:
        neighbours = self._open_neighbours()
        neighbours[self.neighbour_index] = self._neighbour(
            **{NeighbourAttribute.CLOSING_AXIS_SIDE.value: cause}
        )
        query = self._attempt(neighbours)
        query.causes_effect(query.neighbours[self.neighbour_index].disturbed == True)
        return query


def question_catalogue(neighbour_count: int) -> List[ClutterQuestion]:
    """
    :param neighbour_count: How many neighbours the queried clutter holds.
    :return: The questions asked of the recorded attempts: the grasp friction as a cause
        of the lift adjusting for the environment, the crowding around the target as a
        cause of the lift both adjusted and unadjusted so the difference adjusting makes
        is visible, and one neighbour's own position as a cause of its own disturbance.
    """
    return [
        FrictionCausesLift(neighbour_count=neighbour_count),
        CrowdingCausesLift(neighbour_count=neighbour_count),
        CrowdingCausesLift(neighbour_count=neighbour_count, adjusted_for=()),
        ClosingAxisSideCausesDisturbance(neighbour_count=neighbour_count),
    ]


# %% asking one question of a fitted circuit


@dataclass
class ClutterDoQuery(DoQuery):
    """
    Asks one clutter question of a circuit fitted on recorded attempts.
    """

    question: ClutterQuestion
    """
    The question to ask.
    """

    @property
    def fitted_type(self) -> Type:
        """
        :return: The attempt the circuit is fitted on.
        """
        return ClutterPickScene
