"""
Asking a do() question of a relational circuit, whatever the question is about.

A question marks one cause and one effect, and the effect is read off every region of
the cause twice: once by conditioning alone and once with backdoor adjustment. The
answer types hold the two side by side, and :class:`DoQuery` does the asking: it fits a
circuit grouped by the cause, grounds it against the question, and reads the effect off
every region. An experiment says what it fits on and what its questions are; nothing
about the asking itself is particular to one dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from typing_extensions import Any, List, Optional, Tuple, Type

from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

# %% what one answer holds


@dataclass(frozen=True)
class CauseRegion:
    """
    One region of the cause, with how likely the effect is on it.
    """

    description: str
    """
    The values of the cause this region covers.
    """

    probability: float
    """
    How much of the training population the region accounts for.
    """

    conditioned_probability: float
    """
    How likely the effect is given the region, read off the circuit with no adjustment.
    """

    adjusted_probability: float
    """
    How likely the effect is under an intervention setting the cause to the region,
    adjusted for the question's confounders.
    """

    @property
    def shift_from_adjusting(self) -> float:
        """
        How far adjusting moves the answer away from plain conditioning.
        """
        return self.adjusted_probability - self.conditioned_probability


@dataclass(frozen=True)
class DoQueryAnswer:
    """
    What one question yielded on one fitted circuit.
    """

    asked: str
    """
    The question, in words.
    """

    training_example_count: int
    """
    How many examples the circuit was fitted on.
    """

    regions: Tuple[CauseRegion, ...]
    """
    One entry per region of the cause the grounded circuit's support covers.
    """

    @property
    def largest_shift_from_adjusting(self) -> float:
        """
        The largest distance between a conditioned and an adjusted answer over the
        regions: how much the confounders mattered at all.
        """
        return max(abs(region.shift_from_adjusting) for region in self.regions)


# %% how a fit is grouped by the cause


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

    part_field: Optional[str] = None
    """
    The exchangeable-part field whose template holds the cause, or ``None`` when the
    example itself carries it.
    """


# %% what a question has to say for itself


class CausalQuestion(ABC):
    """
    One question of the form: does this property cause that, once the confounders are
    adjusted for?
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
        How a fit has to be grouped for this question's cause to come out
        support-deterministic.
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


# %% asking one question of one fitted circuit


@dataclass
class DoQuery(ABC):
    """
    Fits a relational circuit on an experiment's examples, grounds it against one
    question, and reads the question's effect off every region of its cause.
    """

    question: CausalQuestion
    """
    The question to ask.
    """

    monte_carlo_sample_count: int = 2000
    """
    How many samples grounding draws when it retains a statistic over the parts.
    """

    random_seed: int = 0
    """
    Seed applied to the global NumPy random state before grounding, which draws those
    samples.
    """

    @property
    @abstractmethod
    def fitted_type(self) -> Type:
        """
        The class the circuit is fitted on, whose exchangeable parts the grounding
        retains statistics over.
        """

    def run(self, training_examples: List[Any]) -> DoQueryAnswer:
        """
        Fit a circuit on the examples and answer the question on it.

        :param training_examples: Examples to fit on.
        :return: The answer, one entry per region of the cause.
        """
        causal_circuit = self._grounded_against(training_examples)
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
            for region in causal_circuit.disjoint_support_regions_of(cause_variable)
        ]
        return DoQueryAnswer(
            asked=self.question.asked,
            training_example_count=len(training_examples),
            regions=tuple(sorted(regions, key=lambda region: region.description)),
        )

    def _grounded_against(
        self, training_examples: List[Any]
    ) -> RelationalCausalCircuit:
        """
        Fit a circuit whose branches each hold one value of the cause, and ground it
        against the question.

        Stratifying by the cause is what lets the registration verify support
        determinism: an unconstrained fit gives no guarantee that examples sharing a
        value of the cause end up under one branch, and two branches claiming one value
        are rejected.

        :param training_examples: Examples to fit on.
        :return: The grounded circuit, with its cause and effect registered.
        """
        model = self._unfitted()
        model.fit(training_examples)
        registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)
        np.random.seed(self.random_seed)
        return registry.get_model(UnderspecifiedParameters(self.question.build()))

    def _unfitted(self) -> RelationalProbabilisticCircuit:
        """
        :return: The circuit to fit, grouped by the question's cause: the class circuit
            when the example itself carries the cause, that part field's template when
            one of its parts does.
        """
        stratification = self.question.stratification
        grouped = StratifiedLearning(
            variables=[stratification.column], method=JointProbabilityTree()
        )
        if stratification.part_field is None:
            return RelationalProbabilisticCircuit(
                self.fitted_type,
                monte_carlo_sample_count=self.monte_carlo_sample_count,
                learning_method=grouped,
            )
        return RelationalProbabilisticCircuit(
            self.fitted_type,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
            part_learning_methods={stratification.part_field: grouped},
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
