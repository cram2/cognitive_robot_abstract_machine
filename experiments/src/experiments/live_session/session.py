"""
A live session: the cells performed so far, what each reported, and the moments of the
simulation a developer can go back to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any

from coraplex.plans.executors import PlanOutcome
from experiments.live_session.cells import Cell
from experiments.live_session.exceptions import (
    NoPerformedCellError,
    UnknownSnapshotError,
)
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoSimulationSnapshot
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% reports


class StepStatus(StrEnum):
    """
    How a reported step ended, unless a plan it ran failed; a failed plan's status is the
    name of its failure's class.
    """

    NO_PLAN = ""
    """
    The step ran no plan.
    """

    SUCCEEDED = "ok"
    """
    The plan the step ran succeeded.
    """


@dataclass(kw_only=True)
class StepReport:
    """
    What a developer is told after a step: how its plan ended and where the robot is.
    """

    label: str
    """
    What the step did.
    """

    outcome: PlanOutcome | None
    """
    How the plan the step ran ended; ``None`` if it ran none.
    """

    tool_position: Point3
    """
    Where the tool frame was afterwards, in the world's root frame.
    """

    base_position: Point3
    """
    Where the robot's base was afterwards, in the world's root frame.
    """

    @property
    def status(self) -> str:
        """
        A :class:`StepStatus`, or the name of the failure of the plan the step ran.
        """
        if self.outcome is None:
            return StepStatus.NO_PLAN
        if self.outcome.succeeded:
            return StepStatus.SUCCEEDED
        return type(self.outcome.failure).__name__

    def to_line(self) -> str:
        """
        :return: The report as one line of text, its columns aligned with every other
            report's.
        """
        return "%-34s %-34s tool %s base %s" % (
            self.label,
            self.status,
            self.tool_position.to_np()[:3].round(3),
            self.base_position.to_np()[:3].round(3),
        )


# %% performed cells


@dataclass(kw_only=True)
class PerformedCell:
    """
    A cell a session performed, and the moment before it, to go back to.
    """

    cell: Cell
    """
    The cell.
    """

    snapshot_before: MujocoSimulationSnapshot
    """
    The simulation just before the cell was performed.
    """

    reports: list[StepReport] = field(default_factory=list)
    """
    What the cell reported, in order.
    """

    finished: bool = False
    """
    Whether the cell ran to its end; a cell that raised stays unfinished.
    """


# %% the session


@dataclass(kw_only=True)
class LiveSession:
    """
    A developer driving a simulated robot one cell at a time, with every moment before a
    cell kept, so any step can be undone, retried or replayed.
    """

    simulation: MujocoSim
    """
    The stepped simulation the robot acts in.
    """

    tool_frame: KinematicStructureEntity
    """
    The frame of the tool the steps move, reported after every step.
    """

    robot_base: KinematicStructureEntity
    """
    The robot's base, reported after every step.
    """

    snapshots: dict[str, MujocoSimulationSnapshot] = field(default_factory=dict)
    """
    The moments the developer named, to go back to.
    """

    history: list[PerformedCell] = field(default_factory=list)
    """
    The cells performed, first to last.
    """

    def report(self, *, label: str, outcome: PlanOutcome | None = None) -> StepReport:
        """
        Tell the developer how a step ended and where the robot is now, and keep that
        with the cell being performed.

        :param label: What the step did.
        :param outcome: How the plan the step ran ended, if it ran one.
        :return: The report.
        """
        report = StepReport(
            label=label,
            outcome=outcome,
            tool_position=self.tool_frame.global_pose.position,
            base_position=self.robot_base.global_pose.position,
        )
        if self.history and not self.history[-1].finished:
            self.history[-1].reports.append(report)
        print(report.to_line())
        return report

    def take_snapshot(self, *, name: str) -> MujocoSimulationSnapshot:
        """
        Keep the current moment of the simulation under a name.

        :param name: The name to restore it by; an earlier snapshot of that name is
            replaced.
        :return: The snapshot.
        """
        snapshot = self.simulation.take_snapshot()
        self.snapshots[name] = snapshot
        return snapshot

    def restore_snapshot(self, *, name: str) -> None:
        """
        Put the simulation back to a moment kept by :meth:`take_snapshot`.

        :param name: The name the moment was kept under.
        :raises UnknownSnapshotError: If no moment was kept under that name.
        """
        if name not in self.snapshots:
            raise UnknownSnapshotError(
                snapshot_name=name, known_names=list(self.snapshots)
            )
        self.simulation.restore_snapshot(snapshot=self.snapshots[name])

    def perform(self, *, cell: Cell, namespace: dict[str, Any]) -> None:
        """
        Run a cell and keep it in :attr:`history`, together with the moment before it.

        :param cell: The cell.
        :param namespace: The names the cell's code runs with; names it defines are
            added to it.
        """
        performed = PerformedCell(
            cell=cell, snapshot_before=self.simulation.take_snapshot()
        )
        self.history.append(performed)
        exec(compile(cell.source, f"<cell {cell.title}>", "exec"), namespace)
        performed.finished = True

    def revert_last_cell(self) -> Cell:
        """
        Put the simulation back to the moment before the last cell and drop that cell.

        :return: The cell that was dropped.
        :raises NoPerformedCellError: If no cell has been performed.
        """
        last = self._performed_cell(cell_index=len(self.history) - 1)
        self.simulation.restore_snapshot(snapshot=last.snapshot_before)
        self.history.pop()
        return last.cell

    def replay(self, *, first_cell: int, namespace: dict[str, Any]) -> None:
        """
        Put the simulation back to the moment before a cell and perform that cell and
        every cell after it again.

        :param first_cell: The position in :attr:`history` of the first cell to replay.
        :param namespace: The names the cells' code runs with.
        :raises NoPerformedCellError: If no cell was performed at that position.
        """
        first = self._performed_cell(cell_index=first_cell)
        cells = [performed.cell for performed in self.history[first_cell:]]
        self.simulation.restore_snapshot(snapshot=first.snapshot_before)
        del self.history[first_cell:]
        for cell in cells:
            self.perform(cell=cell, namespace=namespace)

    def _performed_cell(self, *, cell_index: int) -> PerformedCell:
        """
        :param cell_index: A position in :attr:`history`.
        :return: The cell performed at that position.
        :raises NoPerformedCellError: If no cell was performed at that position.
        """
        if not 0 <= cell_index < len(self.history):
            raise NoPerformedCellError(
                cell_index=cell_index, performed_cells=len(self.history)
            )
        return self.history[cell_index]
