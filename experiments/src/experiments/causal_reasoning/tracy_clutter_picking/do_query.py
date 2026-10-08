"""
Asking the clutter circuit what makes a pick succeed.

Each question marks one cause and one effect in the query itself, grounds the relational
circuit against a clutter of the size the question asks about, and reads the effect off
every region of the cause twice: once by conditioning alone and once with backdoor
adjustment, so the two can be set side by side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum

import numpy as np
from krrood.entity_query_language.factories import a, cause, confounder, variable
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from typing_extensions import Any, Dict, List, Optional, Tuple

from experiments.causal_reasoning.do_query import CauseRegion, DoQueryAnswer
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutteredObject,
    ClutterPickScene,
    ClutterPickSceneAggregations,
    PartField,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
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


@dataclass(frozen=True)
class CauseStratification:
    """
    The column a fit groups by so that every value of the cause lands under one branch,
    and which circuit has to do the grouping.
    """

    column: str
    """
    The column to group the rows by.
    """

    part_field: Optional[PartField] = None
    """
    The exchangeable-part field whose template holds the cause, or ``None`` when the
    attempt itself carries it.
    """

    @classmethod
    def of_attempt(cls, attribute: AttemptAttribute) -> CauseStratification:
        """
        :param attribute: The attempt's own attribute acting as the cause.
        :return: The grouping for it.
        """
        return cls(column=attribute.circuit_variable_name)

    @classmethod
    def of_count(cls, count: AttemptCount) -> CauseStratification:
        """
        :param count: The statistic over the neighbours acting as the cause.
        :return: The grouping for it.
        """
        return cls(column=count.circuit_variable_name)

    @classmethod
    def of_neighbour(
        cls, attribute: NeighbourAttribute, part_field: PartField
    ) -> CauseStratification:
        """
        :param attribute: One neighbour's own attribute acting as the cause.
        :param part_field: The field those neighbours are held in.
        :return: The grouping for it, applied to that field's template.
        """
        return cls(column=attribute.value, part_field=part_field)


# %% the questions


@dataclass(frozen=True, kw_only=True)
class ClutterQuestion(ABC):
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
        return CauseStratification.of_attempt(AttemptAttribute.FRICTION_COEFFICIENT)

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
        return CauseStratification.of_count(AttemptCount.CROWDING)

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
        return CauseStratification.of_neighbour(
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
class ClutterDoQuery:
    """
    Fits a relational circuit on recorded attempts, grounds it against one question, and
    reads the question's effect off every region of its cause.
    """

    question: ClutterQuestion
    """
    The question to ask.
    """

    monte_carlo_sample_count: int = 2000
    """
    How many samples grounding draws when it retains a statistic over the neighbours.
    """

    random_seed: int = 0
    """
    Seed applied to the global NumPy random state before grounding, which draws those
    samples.
    """

    def run(self, training_attempts: List[ClutterPickScene]) -> DoQueryAnswer:
        """
        Fit a circuit on the attempts and answer the question on it.

        :param training_attempts: Attempts to fit on.
        :return: The answer, one entry per region of the cause.
        """
        causal_circuit = self._grounded_against(training_attempts)
        [cause_variable] = causal_circuit.causal_variables
        [effect_variable] = causal_circuit.effect_variables
        conditioned_circuit = causal_circuit.backdoor_adjustment(
            cause_variable, effect_variable
        )
        adjusted_circuit = causal_circuit.backdoor_adjustment(
            cause_variable,
            effect_variable,
            adjustment_variables=self._adjustment_variables(causal_circuit),
        )

        regions = [
            CauseRegion(
                description=str(region.event.simple_sets[0][cause_variable]),
                probability=region.probability,
                conditioned_probability=self._effect_probability(
                    conditioned_circuit, region.event, effect_variable
                ),
                adjusted_probability=self._effect_probability(
                    adjusted_circuit, region.event, effect_variable
                ),
            )
            for region in causal_circuit.disjoint_support_regions_of(
                cause_variable
            )
        ]
        return DoQueryAnswer(
            asked=self.question.asked,
            training_example_count=len(training_attempts),
            regions=tuple(sorted(regions, key=lambda region: region.description)),
        )

    def _grounded_against(
        self, training_attempts: List[ClutterPickScene]
    ) -> RelationalCausalCircuit:
        """
        Fit a circuit whose class-level branches each hold one value of the cause, and
        ground it against the question.

        Stratifying by the cause is what lets the registration verify support
        determinism: an unconstrained fit gives no guarantee that attempts sharing a
        value of the cause end up under one branch, and two branches claiming one value
        are rejected.

        :param training_attempts: Attempts to fit on.
        :return: The grounded circuit, with its cause and effect registered.
        """
        model = self._unfitted()
        model.fit(training_attempts)
        registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)
        np.random.seed(self.random_seed)
        return registry.get_model(UnderspecifiedParameters(self.question.build()))

    def _unfitted(self) -> RelationalProbabilisticCircuit:
        """
        :return: The circuit to fit, grouped by the question's cause: the class circuit
            when the attempt carries the cause, that part field's template when one of
            its neighbours does.
        """
        stratification = self.question.stratification
        grouped = StratifiedLearning(
            variables=[stratification.column], method=JointProbabilityTree()
        )
        if stratification.part_field is None:
            return RelationalProbabilisticCircuit(
                ClutterPickScene,
                monte_carlo_sample_count=self.monte_carlo_sample_count,
                learning_method=grouped,
            )
        return RelationalProbabilisticCircuit(
            ClutterPickScene,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
            part_learning_methods={stratification.part_field.value: grouped},
        )

    def _adjustment_variables(
        self, causal_circuit: RelationalCausalCircuit
    ) -> List[Variable]:
        """
        :param causal_circuit: The grounded circuit to resolve against.
        :return: One variable per attribute the question adjusts for.
        """
        return [
            RelationalCausalCircuit.resolve_variable(
                causal_circuit.probabilistic_circuit, name
            )
            for name in self.question.adjustment_variable_names
        ]

    def _effect_probability(
        self, interventional_circuit: Any, region: Event, effect_variable: Variable
    ) -> float:
        """
        Read how likely the question's effect is on one region of the cause.

        :param interventional_circuit: A joint circuit over the cause and the effect.
        :param region: The region of the cause to truncate to.
        :param effect_variable: The effect's variable.
        :return: The truncated circuit's probability of the effect's value.
        """
        truncated_circuit, _ = interventional_circuit.truncated(
            region.fill_missing_variables_pure(interventional_circuit.variables)
        )
        effect = (
            SimpleEvent.from_data({effect_variable: self.question.effect_value})
            .as_composite_set()
            .fill_missing_variables_pure(truncated_circuit.variables)
        )
        return float(truncated_circuit.probability(effect))
