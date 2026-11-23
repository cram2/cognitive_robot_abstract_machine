"""
Explains the failure of a plan by the nodes that caused it, from the causal graph of its
statechart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatchmethod

from coraplex.plans.failure_diagnosis.causal_graph import NodeCausalGraph, RootCause
from coraplex.plans.failure_diagnosis.counterfactual import (
    StallCounterfactual,
    try_stalled_task_freed,
)
from coraplex.plans.failure_diagnosis.exceptions import (
    FailureDiagnosis,
    NodeFailedByACondition,
    NodeFailedOnItsOwn,
    StatementRanOutOfCandidates,
    TaskCollided,
    TaskStalled,
)
from coraplex.plans.failures import (
    EmptyUnderspecified,
    MotionMadeNoProgress,
    MotionViolatedCollisionAvoidance,
)
from coraplex.plans.underspecified import (
    ActionTrial,
    FailedTrial,
    SubtreeRebuild,
    UnderspecifiedChildChooser,
)
from cramph.causal_edges import ControlEdge, RunEdge, WorldEdge
from cramph.composites import ChildChooserAccess, StatementNode
from cramph.data_types import TransitionKind
from cramph.node import StatechartNode
from cramph.node_runs import StatechartRuns
from cramph.state_history import NodeRun
from cramph.statechart import Statechart
from giskardpy.motion_statechart.conflict_edges import ConflictEdge
from krrood.patterns.called_positionally import called_positionally
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

# %% diagnosing


@dataclass(kw_only=True)
class FailureDiagnoser:
    """
    Diagnoses failures of a statechart from its causal graph.
    """

    causal_graph: NodeCausalGraph
    """
    The causal graph of the statechart.
    """

    trial: ActionTrial | None = None
    """
    The trial the plan tries its candidates in, which also tries the counterfactuals of
    a diagnosis; None if the plan has none.
    """

    stalled_nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The nodes a motion failure reported as no longer approaching their goal.
    """

    colliding_nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The nodes moving bodies a motion failure reported as too close to each other.
    """

    colliding_bodies: set[Body] = field(default_factory=set, init=False)
    """
    The bodies a motion failure reported as too close to each other.
    """

    diagnosis_kinds: list[type[FailureDiagnosis]] = field(
        default_factory=lambda: [
            TaskStalled,
            TaskCollided,
            StatementRanOutOfCandidates,
            NodeFailedByACondition,
        ]
    )
    """
    The ways an origin can fail, tried in order; an origin that fails none of these ways
    failed on its own, see :class:`NodeFailedOnItsOwn`.
    """

    @classmethod
    def from_statechart(cls, *, statechart: Statechart) -> FailureDiagnoser:
        """
        :param statechart: A statechart that ran, at least in part.
        :return: A diagnoser over its causal graph, trying counterfactuals in the trial
            its plan tries candidates in.
        """
        access = statechart.context.get_extension(extension_type=ChildChooserAccess)
        choosers = [] if access is None else access.choosers
        trial = next(
            (
                chooser.trial
                for chooser in choosers
                if isinstance(chooser, UnderspecifiedChildChooser)
            ),
            None,
        )
        return cls(
            causal_graph=NodeCausalGraph.from_statechart(statechart=statechart),
            trial=trial,
        )

    @property
    def runs(self) -> StatechartRuns:
        """
        :return: The runs of the statechart.
        """
        return self.causal_graph.runs

    # %% where a failure is reported

    @singledispatchmethod
    @called_positionally
    def diagnose_failure(self, failure: Exception | None) -> list[FailureDiagnosis]:
        """
        :param failure: What the plan failed with, None if it ended without an
            exception.
        :return: One diagnosis per run the failure is reported for.
        """
        return [self.diagnose_run(symptom=run) for run in self.failed_leaves()]

    @diagnose_failure.register
    @called_positionally
    def _(self, failure: MotionMadeNoProgress) -> list[FailureDiagnosis]:
        """
        :return: One diagnosis per task the failure reports as no longer approaching
            its goal.
        """
        self.stalled_nodes = list(failure.no_progress.stalled_tasks) or [
            failure.no_progress.progress_monitor.monitored_node
        ]
        return [
            self.diagnose_run(symptom=self.runs.latest_run(node=node))
            for node in self.stalled_nodes
        ]

    @diagnose_failure.register
    @called_positionally
    def _(self, failure: MotionViolatedCollisionAvoidance) -> list[FailureDiagnosis]:
        """
        :return: One diagnosis per run moving a body the failure reports as too close
            to another.
        """
        self.colliding_bodies = {
            body
            for violation in failure.violation.violated_collisions
            for body in (
                violation.closest_points.body_a,
                violation.closest_points.body_b,
            )
        }
        self.colliding_nodes = [
            run.node for run in self.runs_moving(bodies=self.colliding_bodies)
        ]
        return [
            self.diagnose_run(symptom=self.runs.latest_run(node=node))
            for node in self.colliding_nodes
        ]

    @diagnose_failure.register
    @called_positionally
    def _(self, failure: EmptyUnderspecified) -> list[FailureDiagnosis]:
        """
        :return: The diagnosis of the statement that ran out of candidates.
        """
        return [self.diagnose_run(symptom=self.runs.latest_run(node=failure.node))]

    def diagnose_node(self, *, node: StatechartNode) -> FailureDiagnosis:
        """
        :param node: A node whose latest run failed.
        :return: Why it failed.
        """
        return self.diagnose_run(symptom=self.runs.latest_run(node=node))

    def failed_leaves(self) -> list[NodeRun]:
        """
        :return: The failed runs no failed run of a node they hold explains and no later
            run of their node replaced, the earliest first.
        """
        failed = [
            run
            for run in self.runs.all_runs
            if run.failed and run == self.runs.latest_run(node=run.node)
        ]
        return sorted(
            (run for run in failed if not self.failed_runs_inside(run=run)),
            key=lambda run: run.last_tick,
        )

    def failed_runs_inside(self, *, run: NodeRun) -> list[NodeRun]:
        """
        :param run: A run.
        :return: The failed runs of nodes below `run`'s node within it, the earliest
            first.
        """
        descendants = set(run.node.descendants)
        return sorted(
            (
                other
                for other in self.runs.all_runs
                if other.failed
                and other.node in descendants
                and run.contains(other=other)
            ),
            key=lambda other: other.last_tick,
        )

    def runs_moving(self, *, bodies: set[Body]) -> list[NodeRun]:
        """
        :param bodies: Bodies that came too close to each other.
        :return: The runs active at the newest tick that move a degree of freedom
            moving one of `bodies`, or every active run that moves anything if none
            does.
        """
        chain = self.degrees_of_freedom_moving(bodies=bodies)
        active = [
            run
            for run in self.runs.all_runs
            if run.last_tick >= self.runs.last_tick
            and self.causal_graph.access[run.node].moved
        ]
        moving = [
            run for run in active if self.causal_graph.access[run.node].moved & chain
        ]
        return moving or active

    def degrees_of_freedom_moving(
        self, *, bodies: set[Body]
    ) -> frozenset[DegreeOfFreedom]:
        """
        :return: The degrees of freedom that move one of `bodies` relative to the root
            of the world.
        """
        world = self.runs.statechart.context.world
        return frozenset(
            degree_of_freedom
            for body in bodies
            for connection in world.compute_chain_of_connections(world.root, body)
            for degree_of_freedom in connection.dofs
        )

    # %% from the report to the cause

    def origin_of(self, *, run: NodeRun) -> NodeRun:
        """
        :param run: A run the failure is reported for.
        :return: The run where the failure started: the earliest failed run below it
            that no failed run below that one explains, `run` itself if there is none.
        """
        if isinstance(run.node, StatementNode):
            return run
        inside = self.failed_runs_inside(run=run)
        if not inside:
            return run
        return self.origin_of(run=inside[0])

    def diagnose_run(self, *, symptom: NodeRun) -> FailureDiagnosis:
        """
        :param symptom: A run the failure is reported for.
        :return: Why it failed, by the first of :attr:`diagnosis_kinds` its origin
            failed in.
        """
        origin = self.origin_of(run=symptom)
        for kind in self.diagnosis_kinds:
            diagnosis = kind.found_at(symptom=symptom, origin=origin, diagnoser=self)
            if diagnosis is not None:
                return diagnosis
        return NodeFailedOnItsOwn.found_at(
            symptom=symptom, origin=origin, diagnoser=self
        )

    def failing_conditions_into(self, *, run: NodeRun) -> list[ControlEdge]:
        """
        :param run: A run.
        :return: The control edges into `run` from its fail and interrupt conditions.
        """
        return [
            edge
            for edge in self.causal_graph.edges_into(run=run)
            if isinstance(edge, ControlEdge)
            and edge.transition_kind in (TransitionKind.FAIL, TransitionKind.INTERRUPT)
        ]

    # %% the causes of each kind of failure

    def causes_of_stall(
        self, *, stalled: NodeRun
    ) -> tuple[list[RootCause], StallCounterfactual | None]:
        """
        A task stops approaching its goal when another task drives its degrees of
        freedom at the same time, or when its goal depends on degrees of freedom it
        cannot move, which an earlier run left where they are. The latter holds only if
        the task, tried again free to move them, reaches its goal.

        :param stalled: The run that stopped approaching its goal.
        :return: The runs that made it stall, the most likely first, and what trying it
            free of what an earlier run left showed, None if that was not asked.
        """
        edges = self.causal_graph.edges_into(run=stalled)
        access = self.causal_graph.access[stalled.node]
        conflicts = [
            edge
            for edge in edges
            if isinstance(edge, ConflictEdge) and edge.degrees_of_freedom & access.moved
        ]
        if conflicts:
            return [
                RootCause(run=edge.cause, chain=(edge,)) for edge in conflicts
            ], None
        unmovable = access.read - access.driven
        left_by_earlier = [
            RootCause(run=edge.cause, chain=(edge,))
            for edge in self.latest_first(edges=edges)
            if isinstance(edge, WorldEdge) and edge.degrees_of_freedom & unmovable
        ]
        if not left_by_earlier:
            return [], None
        counterfactual = try_stalled_task_freed(task=stalled.node, trial=self.trial)
        if counterfactual == StallCounterfactual.STILL_STALLED:
            return [], counterfactual
        return left_by_earlier, counterfactual

    def causes_of_collision(self, *, colliding: NodeRun) -> list[RootCause]:
        """
        :param colliding: A run moving a body that came too close to another.
        :return: The last runs before it that moved a degree of freedom moving one of
            the bodies that came too close, among those it does not drive itself.
        """
        moving = self.degrees_of_freedom_moving(bodies=self.colliding_bodies)
        return self.plan_runs_leaving(
            degrees_of_freedom=moving - self.causal_graph.access[colliding.node].driven,
            statement=colliding,
        )

    def causes_of_observation(self, *, run: NodeRun) -> list[RootCause]:
        """
        :param run: A run that failed on what it observed or aimed for.
        :return: The runs that left what it read where it was, the latest first.
        """
        return [
            RootCause(run=edge.cause, chain=(edge,))
            for edge in self.latest_first(edges=self.causal_graph.edges_into(run=run))
            if isinstance(edge, WorldEdge)
        ]

    def causes_of_failing_condition(self, *, run: NodeRun) -> list[RootCause]:
        """
        :param run: A run a fail or interrupt condition ended.
        :return: The runs that caused what the condition read, through the nodes it
            read; a read node that nothing else explains is the cause itself.
        """
        root_causes = []
        for edge in self.failing_conditions_into(run=run):
            upstream = self.causes_of_observation(run=edge.cause)
            if not upstream:
                root_causes.append(RootCause(run=edge.cause, chain=(edge,)))
                continue
            root_causes.extend(
                RootCause(run=root.run, chain=(*root.chain, edge)) for root in upstream
            )
        return root_causes

    def causes_shared_by_candidates(self, *, statement: NodeRun) -> list[RootCause]:
        """
        :param statement: A run of a statement that ran out of candidates.
        :return: The causes every failed candidate shares, whether it failed in the
            plan or in its trial, or the causes of any of them if they share none.
        """
        per_candidate = [
            *(
                self.diagnose_run(symptom=child).root_causes
                for child in self.runs.all_runs
                if child.failed
                and child.node.parent_node is statement.node
                and statement.contains(other=child)
            ),
            *(
                self.causes_of_trial(failed_trial=failed_trial, statement=statement)
                for failed_trial in self.trials_of(statement=statement.node)
            ),
        ]
        nodes_per_candidate = [
            {root.run.node for root in roots} for roots in per_candidate
        ]
        shared = (
            set.intersection(*nodes_per_candidate) if nodes_per_candidate else set()
        )
        roots = [root for roots in per_candidate for root in roots]
        kept = [root for root in roots if root.run.node in shared] if shared else roots
        return list({root.run: root for root in kept}.values())

    def trials_of(self, *, statement: StatementNode) -> list[FailedTrial]:
        """
        :return: The failed trials of candidates of `statement`.
        """
        if self.trial is None:
            return []
        return [
            failed_trial
            for failed_trial in self.trial.failed_trials
            if isinstance(failed_trial.build, SubtreeRebuild)
            and statement in failed_trial.build.candidates
        ]

    def causes_of_trial(
        self, *, failed_trial: FailedTrial, statement: NodeRun
    ) -> list[RootCause]:
        """
        A trial starts from the world the plan left, so what it read and no node of its
        own changed was left by the plan's runs before the statement. Only the recorded
        history of the trial is read; no trial runs again.

        :param failed_trial: A failed trial of a candidate of `statement`.
        :param statement: The run of the statement.
        :return: The runs of the plan that left what the trial read, for every failure
            of the trial no node of its own explains.
        """
        trial_diagnoser = FailureDiagnoser(
            causal_graph=NodeCausalGraph.from_statechart(
                statechart=failed_trial.statechart
            )
        )
        root_causes = []
        for diagnosis in trial_diagnoser.diagnose_failure(failed_trial.failure):
            if diagnosis.root_causes:
                continue
            access = trial_diagnoser.causal_graph.access[diagnosis.origin.node]
            root_causes.extend(
                self.plan_runs_leaving(
                    degrees_of_freedom=diagnosis.inputs_from_before(access=access),
                    statement=statement,
                )
            )
        return root_causes

    def plan_runs_leaving(
        self, *, degrees_of_freedom: frozenset[DegreeOfFreedom], statement: NodeRun
    ) -> list[RootCause]:
        """
        :param degrees_of_freedom: Degrees of freedom of a copy of the plan's world.
        :param statement: A run of the plan.
        :return: The last runs of the plan before `statement` that moved them, matched
            by their ids.
        """
        changes = self.causal_graph.changes_of_the_world
        left_by: dict[NodeRun, set[DegreeOfFreedom]] = {}
        for degree_of_freedom in degrees_of_freedom:
            mover = changes.last_mover(
                degree_of_freedom_id=degree_of_freedom.id, reader=statement
            )
            if mover is not None:
                left_by.setdefault(mover, set()).add(degree_of_freedom)
        return [
            RootCause(
                run=mover,
                chain=(
                    WorldEdge(
                        cause=mover,
                        effect=statement,
                        degrees_of_freedom=frozenset(moved),
                    ),
                ),
            )
            for mover, moved in left_by.items()
        ]

    @staticmethod
    def latest_first(*, edges: list[RunEdge]) -> list[RunEdge]:
        """
        :return: `edges`, those from the latest cause first.
        """
        return sorted(edges, key=lambda edge: edge.cause.last_tick, reverse=True)
