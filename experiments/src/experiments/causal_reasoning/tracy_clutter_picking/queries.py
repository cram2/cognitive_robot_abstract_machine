"""
The causal questions both pipelines are asked, each as one ``cause``/``causes_effect``
EQL query that reads the same for either pipeline.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from krrood.entity_query_language.factories import cause, confounder
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, List, Tuple

from experiments.causal_reasoning.comparison.domain import RelationalDomain
from experiments.causal_reasoning.comparison.queries import (
    AdjustedCountCase,
    CausalQueryCase,
    Confounder,
    part_query,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutteredObject,
    PartField,
    attempt_domain,
)

# %% building blocks


@dataclass(frozen=True, kw_only=True)
class AttemptQueryCase(CausalQueryCase, ABC):
    """
    One question about an attempt, asked of a clutter with a chosen number of
    neighbours.
    """

    @property
    def domain(self) -> RelationalDomain:
        return attempt_domain()

    @property
    def neighbour_count(self) -> int:
        """
        How many neighbours the queried attempt has.
        """
        return self.open_part_count

    def _open_neighbours(self) -> List[Match]:
        """
        :return: One fully open query per neighbour.
        """
        return self.open_parts()[PartField.NEIGHBOURS]

    def _attempt_query(self, neighbours: List[Match], **specified: Any) -> Match:
        """
        :param neighbours: One query per neighbour.
        :param specified: Attempt attribute markers or values to set instead of
            leaving open.
        :return: The query for the attempt.
        """
        return self.example_query({PartField.NEIGHBOURS: neighbours}, **specified)


ENVIRONMENT = Confounder(name="environment", noun="the environment")
"""
Adjusting for the environment, which decides both how slippery and how crowded an
attempt is.
"""


@dataclass(frozen=True, kw_only=True)
class FrictionCausesLift(AttemptQueryCase):
    """
    Which grasp friction makes the target come up, once the environment -- which decides
    both how slippery and how crowded an attempt is -- is adjusted for?
    """

    @property
    def name(self) -> str:
        return f"friction_causes_lift_{self.neighbour_count}_neighbours"

    @property
    def question(self) -> str:
        return (
            f"In a clutter of {self.neighbour_count} neighbours, which grasp friction "
            "coefficient causes the target to be lifted, adjusting for the environment?"
        )

    def build(self) -> Match:
        query = self._attempt_query(
            self._open_neighbours(), friction_coefficient=cause, environment=confounder
        )
        query.causes_effect(query.variable.lifted == True)
        return query

    def describe_cause(self, region: str) -> str:
        return f"a grasp friction coefficient of {region}"

    @property
    def effect(self) -> str:
        return "the target is lifted"


@dataclass(frozen=True, kw_only=True)
class CrowdingCausesLift(AttemptQueryCase, AdjustedCountCase):
    """
    How many adjacent neighbours can the target have and still come up, once the given
    confounders are adjusted for?

    The cause is a count over the exchangeable parts.
    """

    statistic_name: str = "crowding_count"
    """
    The count of neighbours standing adjacent to the target.
    """

    confounders: Tuple[Confounder, ...] = (ENVIRONMENT,)
    """
    What to adjust for: the environment unless asked otherwise.
    """

    @property
    def name(self) -> str:
        adjusting = (
            "adjusting_"
            + "_and_".join(confounder.name for confounder in self.confounders)
            if self.confounders
            else "unadjusted"
        )
        return f"crowding_causes_lift_{self.neighbour_count}_neighbours_{adjusting}"

    @property
    def question(self) -> str:
        adjusting = (
            "adjusting for "
            + " and ".join(confounder.noun for confounder in self.confounders)
            if self.confounders
            else "with nothing adjusted for"
        )
        return (
            f"In a clutter of {self.neighbour_count} neighbours, how many of them "
            f"standing adjacent to the target causes it to be lifted, {adjusting}?"
        )

    def build(self) -> Match:
        query = self._attempt_query(
            self._open_neighbours(),
            **{self.statistic_name: cause},
            **{adjusted.name: confounder for adjusted in self.confounders},
        )
        query.causes_effect(query.variable.lifted == True)
        return query

    def describe_cause(self, region: str) -> str:
        return f"{region} adjacent neighbours"

    @property
    def effect(self) -> str:
        return "the target is lifted"


@dataclass(frozen=True, kw_only=True)
class ClosingAxisSideCausesDisturbance(AttemptQueryCase):
    """
    Does one neighbour standing where the fingers close cause that neighbour to be
    shoved aside?

    Cause and effect both live on one exchangeable part.
    """

    neighbour_index: int
    """
    Which neighbour the question is about.
    """

    @property
    def name(self) -> str:
        return (
            f"closing_axis_side_causes_disturbance_of_neighbour_{self.neighbour_index}"
            f"_of_{self.neighbour_count}"
        )

    @property
    def question(self) -> str:
        return (
            f"In a clutter of {self.neighbour_count} neighbours, does neighbour "
            f"{self.neighbour_index} standing along the fingers' closing axis cause it "
            "to be disturbed by the pick?"
        )

    def build(self) -> Match:
        neighbours = self._open_neighbours()
        neighbours[self.neighbour_index] = part_query(
            ClutteredObject, closing_axis_side=cause
        )
        query = self._attempt_query(neighbours)
        query.causes_effect(
            query.variable.neighbours[self.neighbour_index].disturbed == True
        )
        return query

    def describe_cause(self, region: str) -> str:
        return f"neighbour {self.neighbour_index} standing {region} the closing axis"

    @property
    def effect(self) -> str:
        return f"neighbour {self.neighbour_index} is disturbed"


def attempt_level_cases(neighbour_count: int) -> List[CausalQueryCase]:
    """
    :param neighbour_count: How many neighbours the queried attempt has.
    :return: The questions whose cause and effect are both attempt attributes or
        counts, about a clutter of that size.
    """
    return [
        FrictionCausesLift(open_part_count=neighbour_count),
        CrowdingCausesLift(open_part_count=neighbour_count),
        CrowdingCausesLift(open_part_count=neighbour_count, confounders=()),
    ]


def neighbour_level_cases(neighbour_count: int) -> List[CausalQueryCase]:
    """
    :param neighbour_count: How many neighbours the queried attempt has.
    :return: The questions whose cause or effect lives on one neighbour, about a
        clutter of that size.
    """
    return [
        ClosingAxisSideCausesDisturbance(
            open_part_count=neighbour_count, neighbour_index=0
        )
    ]


def query_catalogue(
    recorded_neighbour_count: int,
    smaller_neighbour_count: int = 4,
    larger_neighbour_count: int = 12,
) -> List[CausalQueryCase]:
    """
    Every question of the experiment: each kind of cause asked about a clutter of the
    recorded size, then again about clutters of other sizes.

    :param recorded_neighbour_count: How many neighbours the recorded attempts have.
    :param smaller_neighbour_count: Neighbour count of the question asked about a
        smaller clutter than recorded.
    :param larger_neighbour_count: Neighbour count of the questions asked about a larger
        clutter than recorded.
    :return: The questions, in the order they are asked.
    """
    return (
        attempt_level_cases(recorded_neighbour_count)
        + neighbour_level_cases(recorded_neighbour_count)
        + [
            FrictionCausesLift(open_part_count=smaller_neighbour_count),
            CrowdingCausesLift(open_part_count=larger_neighbour_count),
            ClosingAxisSideCausesDisturbance(
                open_part_count=larger_neighbour_count,
                neighbour_index=larger_neighbour_count - 1,
            ),
        ]
    )


def monte_carlo_cases(recorded_neighbour_count: int) -> List[CausalQueryCase]:
    """
    The questions whose answers are followed as grounding draws more samples: the
    crowding question and the one whose cause and effect live on a neighbour, both of
    which leave the crowding count open.

    :param recorded_neighbour_count: How many neighbours the recorded attempts have.
    :return: The two questions.
    """
    return [
        CrowdingCausesLift(open_part_count=recorded_neighbour_count),
        ClosingAxisSideCausesDisturbance(
            open_part_count=recorded_neighbour_count, neighbour_index=0
        ),
    ]
