"""
The diagnoses of a plan failure: each says how the run where the failure started failed,
and which other runs caused it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Self

from coraplex.plans.failure_diagnosis.causal_graph import RootCause
from coraplex.plans.failure_diagnosis.counterfactual import StallCounterfactual
from coraplex.plans.failures import PlanFailure
from coraplex.robot_plans.actions.base import Action
from cramph.composites import StatementNode
from cramph.state_history import NodeRun
from cramph.world_access import WorldAccess
from krrood.patterns.classproperty import classproperty
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

if TYPE_CHECKING:
    from coraplex.plans.failure_diagnosis.diagnosis import FailureDiagnoser

# %% what a diagnosis could say about the cause


class Verdict(StrEnum):
    """
    What a diagnosis could say about the cause.
    """

    ONE_CAUSING_NODE = "one other node caused it"
    SEVERAL_CANDIDATES = "one of several other nodes caused it"
    NO_OTHER_NODE = "no other node explains it"


# %% a diagnosis


@dataclass(kw_only=True)
class FailureDiagnosis(PlanFailure, ABC):
    """
    Why a run of a plan failed, in terms of other runs.

    Each subclass is one way the run where the failure started can fail.
    """

    symptom: NodeRun
    """
    The run the failure was reported for.
    """

    origin: NodeRun
    """
    The run where the failure started.
    """

    root_causes: list[RootCause]
    """
    The runs that caused the failure, the most likely first.
    """

    @classmethod
    @abstractmethod
    def found_at(
        cls, *, symptom: NodeRun, origin: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        """
        :param symptom: A run a failure is reported for.
        :param origin: The run where that failure started.
        :param diagnoser: The diagnoser of the statechart the runs belong to.
        :return: The diagnosis of the failure if `origin` failed this way, else None.
        """

    @classproperty
    @abstractmethod
    def how_the_origin_failed(cls) -> str:
        """
        :return: How the origin failed, as the verb phrase following it.
        """

    @property
    def verdict(self) -> Verdict:
        """
        :return: What this diagnosis could say about the cause.
        """
        causing_nodes = {root.run.node for root in self.root_causes}
        if not causing_nodes:
            return Verdict.NO_OTHER_NODE
        if len(causing_nodes) == 1:
            return Verdict.ONE_CAUSING_NODE
        return Verdict.SEVERAL_CANDIDATES

    def inputs_from_before(self, *, access: WorldAccess) -> frozenset[DegreeOfFreedom]:
        """
        :param access: What the node of :attr:`origin` does to the world.
        :return: What :attr:`origin` read whose state its statechart was given from
            before it.
        """
        return access.read

    @staticmethod
    def described(*, run: NodeRun) -> str:
        """
        :param run: A run.
        :return: `run`, with the action holding its node if that is another node.
        """
        action = Action.enclosing(node=run.node)
        if action is None or action is run.node:
            return str(run)
        return f"{run} in {action.name}"

    def report_lines(self) -> list[str]:
        """
        :return: The symptom, the origin and how it failed, one per line.
        """
        return [
            f"symptom  {self.described(run=self.symptom)}",
            f"origin   {self.described(run=self.origin)} {self.how_the_origin_failed}",
        ]

    def error_message(self) -> str:
        """
        :return: The symptom, the origin and how it failed, the verdict and each cause
            with the edges from it to the origin, one per line.
        """
        lines = [*self.report_lines(), f"verdict  {self.verdict.value}"]
        for rank, root in enumerate(self.root_causes, start=1):
            lines.append(f"cause {rank}  {self.described(run=root.run)}")
            lines.extend(
                f"           {edge.cause.node.name} {edge.describe()} "
                f"{edge.effect.node.name}"
                for edge in root.chain
            )
        return "\n".join(lines)

    def suggest_correction(self) -> str:
        if not self.root_causes:
            return ""
        return "Repair the node of the first cause before trying the failed one again."


# %% the ways the origin of a failure can fail


@dataclass(kw_only=True)
class TaskStalled(FailureDiagnosis):
    """
    The origin is a motion task that stopped approaching its goal.
    """

    counterfactual: StallCounterfactual | None = None
    """
    What trying the task again, free to move what an earlier node left, showed; None if
    no earlier node was suspected.
    """

    @classmethod
    def found_at(
        cls, *, symptom: NodeRun, origin: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if origin.node not in diagnoser.stalled_nodes:
            return None
        root_causes, counterfactual = diagnoser.causes_of_stall(stalled=origin)
        return cls(
            symptom=symptom,
            origin=origin,
            root_causes=root_causes,
            counterfactual=counterfactual,
        )

    @classproperty
    def how_the_origin_failed(cls) -> str:
        return "stopped approaching its goal"

    def inputs_from_before(self, *, access: WorldAccess) -> frozenset[DegreeOfFreedom]:
        """
        Differs from :meth:`FailureDiagnosis.inputs_from_before` in leaving out what the
        task moves or holds itself.
        """
        return access.read - access.driven

    def report_lines(self) -> list[str]:
        lines = super().report_lines()
        if self.counterfactual is not None:
            lines.append(f"trial    {self.counterfactual.value}")
        return lines


@dataclass(kw_only=True)
class TaskCollided(FailureDiagnosis):
    """
    The origin moved bodies closer to each other than collision avoidance allows.
    """

    @classmethod
    def found_at(
        cls, *, symptom: NodeRun, origin: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if origin.node not in diagnoser.colliding_nodes:
            return None
        return cls(
            symptom=symptom,
            origin=origin,
            root_causes=diagnoser.causes_of_collision(colliding=origin),
        )

    @classproperty
    def how_the_origin_failed(cls) -> str:
        return "brought bodies closer than collision avoidance allows"


@dataclass(kw_only=True)
class StatementRanOutOfCandidates(FailureDiagnosis):
    """
    The origin is an underspecified statement none of whose candidates succeeded.
    """

    @classmethod
    def found_at(
        cls, *, symptom: NodeRun, origin: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if not isinstance(origin.node, StatementNode):
            return None
        return cls(
            symptom=symptom,
            origin=origin,
            root_causes=diagnoser.causes_shared_by_candidates(statement=origin),
        )

    @classproperty
    def how_the_origin_failed(cls) -> str:
        return "ran out of candidates to try"


@dataclass(kw_only=True)
class NodeFailedByACondition(FailureDiagnosis):
    """
    The origin was failed or interrupted by a condition reading another node.
    """

    @classmethod
    def found_at(
        cls, *, symptom: NodeRun, origin: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if not diagnoser.failing_conditions_into(run=origin):
            return None
        return cls(
            symptom=symptom,
            origin=origin,
            root_causes=diagnoser.causes_of_failing_condition(run=origin),
        )

    @classproperty
    def how_the_origin_failed(cls) -> str:
        return "was failed by a condition reading another node"


@dataclass(kw_only=True)
class NodeFailedOnItsOwn(FailureDiagnosis):
    """
    The origin failed on what it observed itself; this way of failing applies to every
    origin.
    """

    @classmethod
    def found_at(
        cls, *, symptom: NodeRun, origin: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self:
        return cls(
            symptom=symptom,
            origin=origin,
            root_causes=diagnoser.causes_of_observation(run=origin),
        )

    @classproperty
    def how_the_origin_failed(cls) -> str:
        return "failed on what it observed itself"
