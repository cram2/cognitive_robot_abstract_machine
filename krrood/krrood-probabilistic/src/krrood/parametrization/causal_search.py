"""
The results of searching for the intervention on a cause that best explains an effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import random_events.variable
from random_events.product_algebra import Event
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from probabilistic_model.causal_structure.identification import Identification
    from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
        ProbabilisticCircuitRepresentation,
    )

# %% cause and effect variables


@dataclass(kw_only=True)
class CauseEffectVariables:
    """
    The cause candidates and effect variable a ``cause`` search resolves to.
    """

    cause_variables: list[random_events.variable.Variable]
    """
    The variable(s) a ``cause`` intervention is searched over.

    When there is more than one, each is tried independently -- there is no joint,
    multi-variable intervention -- and the one with the highest
    :attr:`~krrood.parametrization.causal_search.ScoredIntervention.effect_probability_given_region`
    becomes the primary cause: the candidate whose own best region gives the highest
    ``P(effect | do(cause in best_region))``.
    """

    effect_variable: random_events.variable.Variable
    """
    The variable a ``causes_effect(...)`` condition declares as the effect.
    """

    adjustment_variables_by_cause: dict[
        random_events.variable.Variable, list[random_events.variable.Variable]
    ]
    """
    Per cause candidate, the variables its interventional search adjusts for (Pearl's
    backdoor-criterion adjustment set): those marked ``confounder`` and those a learned
    causal structure chose for the candidate, empty for a candidate with neither.
    """


@dataclass(kw_only=True)
class ScoredIntervention:
    """
    One cause candidate's best-region search result, scored for comparison against the
    other candidates when a query has more than one ``cause`` field.
    """

    cause_variable: random_events.variable.Variable
    """
    The candidate cause variable this result is for.
    """

    effect_probability_given_region: float
    """
    The interventional probability that the effect holds, given this variable is
    restricted to its best region.

    Comparable across candidates whatever share of their
    domain the region covers: the higher one is the better explanation.
    """

    narrowed_circuit: ProbabilisticCircuitRepresentation
    """
    The interventional joint, truncated to the effect condition and this candidate's
    best region -- what the query samples from if this candidate turns out to be the
    primary cause.

    Of the circuit type the causal circuit was read from.
    """


@dataclass(kw_only=True)
class CandidateAdjustment:
    """
    One set of a query's variables that the effect of a cause may be identified by
    adjusting for.
    """

    variables: list[random_events.variable.Variable]
    """
    The variables to adjust for.
    """


@dataclass(kw_only=True)
class AdjustmentCandidates:
    """
    What a learned causal structure says one cause candidate adjusts for.
    """

    cause_variable: random_events.variable.Variable
    """
    The cause candidate.
    """

    identification: Identification
    """
    What the structure says about the effect of the cause.
    """

    adjustments: list[CandidateAdjustment]
    """
    Each set the effect may be identified by adjusting for: the one chosen set if the
    effect is identified, one per candidate if it is partially identified, none if it is
    not identified.
    """

    @property
    def variables(self) -> list[random_events.variable.Variable]:
        """
        :return: The variables of every candidate, each once, in order of first use.
        """
        return list(
            dict.fromkeys(
                variable
                for adjustment in self.adjustments
                for variable in adjustment.variables
            )
        )


@dataclass(kw_only=True)
class RegionRange:
    """
    The effect's probability under an intervention setting the cause to one region,
    across every candidate adjustment set.
    """

    cause_region: Event
    """
    The region of the cause.
    """

    lowest_effect_probability: float
    """
    The lowest probability of the effect across the candidates.
    """

    highest_effect_probability: float
    """
    The highest probability of the effect across the candidates.
    """


@dataclass(kw_only=True)
class InterventionalRange:
    """
    The range an effect's interventional probability lies in, per region of one cause,
    when the learned causal structure identifies it only up to a few candidate
    adjustment sets.
    """

    cause_variable: random_events.variable.Variable
    """
    The cause.
    """

    candidate_adjustments: list[CandidateAdjustment]
    """
    The candidate adjustment sets the range is taken across.
    """

    includes_no_effect: bool
    """
    Whether the range also takes in the effect's probability without any intervention,
    since in some graph the structure stands for the cause does not cause it at all.
    """

    regions: list[RegionRange]
    """
    The range per region of the cause.
    """
