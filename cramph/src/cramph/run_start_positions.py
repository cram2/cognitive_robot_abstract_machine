"""
Where the degrees of freedom a node moves were when each run of that node started, so
that a later diagnosis can ask where a run found them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from cramph.context import ContextExtension, StatechartContext
from cramph.data_types import LifeCycleValue
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.node import StatechartNode
from cramph.state_history import NodeRun
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

# %% where each run found what it moves


@dataclass(frozen=True, kw_only=True)
class RunStart:
    """
    The start of one run of a node: the node, and the tick its run started on.
    """

    node: StatechartNode
    """
    The node that ran.
    """

    tick: int
    """
    The tick the run started on.
    """


@dataclass(kw_only=True)
class RunStartPositions(ContextExtension):
    """
    The positions the degrees of freedom a node moves had when each run of that node
    started.
    """

    positions_at_run_start: dict[RunStart, dict[DegreeOfFreedom, float]] = field(
        default_factory=dict
    )
    """
    For the start of each run of a node that moves something, the position of each
    degree of freedom the node moves, from before the tick the run started on.
    """

    def position_at_start_of(
        self, *, run: NodeRun, degree_of_freedom: DegreeOfFreedom
    ) -> float | None:
        """
        :param run: A run of a statechart in the context holding this extension.
        :param degree_of_freedom: A degree of freedom the node of `run` moves.
        :return: Its position when `run` started, None if none was recorded.
        """
        return self.positions_at_run_start.get(
            RunStart(node=run.node, tick=run.start_tick), {}
        ).get(degree_of_freedom)


# %% recording them while the statechart runs


@dataclass(kw_only=True)
class RunStartPositionRecording(TickingExecutorExtension):
    """
    Records into the :class:`RunStartPositions` of the executor's context where the
    degrees of freedom a node moves were before the tick its run starts on.
    """

    _positions_before_tick: dict[UUID, float] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The position of every degree of freedom of the world before the current tick, by its
    id.
    """

    _running: set[StatechartNode] = field(default_factory=set, init=False, repr=False)
    """
    The nodes whose run had started by the end of the previous tick.
    """

    def extend_context(self, *, context: StatechartContext) -> None:
        """
        Gives `context` a :class:`RunStartPositions` to record into.
        """
        context.ensure_extension(extension=RunStartPositions())

    def after_compile(self, *, executor: StatechartExecutor) -> None:
        """
        Forgets which nodes ran, since a new statechart was compiled, and remembers
        where every degree of freedom is before the tick that compiling runs.
        """
        self._running = set()
        self._remember_positions(executor=executor)

    def before_tick(self, *, executor: StatechartExecutor) -> None:
        """
        Records the runs the previous tick started, then remembers where every degree of
        freedom is before anything moves it this tick.
        """
        self._record_started_runs(executor=executor)
        self._remember_positions(executor=executor)

    def after_run(self, *, executor: StatechartExecutor) -> None:
        """
        Records the runs the last tick started.
        """
        self._record_started_runs(executor=executor)

    def _remember_positions(self, *, executor: StatechartExecutor) -> None:
        """
        Keep where every degree of freedom of the world is now.
        """
        state = executor.context.world.state
        self._positions_before_tick = dict(
            zip(state.ids, state.positions.tolist(), strict=True)
        )

    def _record_started_runs(self, *, executor: StatechartExecutor) -> None:
        """
        Record, for each node whose run the latest tick started, where the degrees of
        freedom it moves were before that tick.
        """
        record = executor.context.require_extension(extension_type=RunStartPositions)
        for node in executor.statechart.nodes:
            if node.life_cycle_state == LifeCycleValue.NOT_STARTED:
                self._running.discard(node)
                continue
            if node in self._running:
                continue
            self._running.add(node)
            moved = node.compute_world_access().moved_degrees_of_freedom
            if not moved:
                continue
            record.positions_at_run_start[
                RunStart(node=node, tick=executor.tick_count)
            ] = {
                degree_of_freedom: self._positions_before_tick[degree_of_freedom.id]
                for degree_of_freedom in moved
                if degree_of_freedom.id in self._positions_before_tick
            }
