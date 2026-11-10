"""
Tests for a live session: reports, named snapshots, performed cells and replaying them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coraplex.plans.executors import PlanOutcome
from coraplex.plans.failures import PlanFailure
from experiments.live_session.cells import CellScript
from experiments.live_session.exceptions import (
    NoPerformedCellError,
    UnknownSnapshotError,
)
from experiments.live_session.session import StepStatus

from .conftest import PendulumSession

# %% reports


def test_a_report_shows_where_the_tool_and_the_base_are(
    pendulum_session: PendulumSession, capsys: pytest.CaptureFixture[str]
):
    """
    The positions are read from the world when the report is shown, not kept with it.
    """
    session = pendulum_session.session
    tool_position = session.tool_frame.global_pose.position.to_np()[:3]
    base_position = session.robot_base.global_pose.position.to_np()[:3]

    report = session.report(label="looked")

    assert capsys.readouterr().out == (
        f"{report.to_line()} tool {tool_position.round(3)} "
        f"base {base_position.round(3)}\n"
    )


@pytest.mark.parametrize(
    "outcome, status",
    [
        (None, StepStatus.NO_PLAN),
        (PlanOutcome(), StepStatus.SUCCEEDED),
        (PlanOutcome(failure=PlanFailure()), PlanFailure.__name__),
    ],
)
def test_a_report_tells_how_its_plan_ended(
    pendulum_session: PendulumSession, outcome: PlanOutcome | None, status: str
):
    report = pendulum_session.session.report(label="ran", outcome=outcome)

    assert report.status == status


# %% named snapshots


def test_restoring_a_named_snapshot_puts_the_hinge_back(
    pendulum_session: PendulumSession, swing_the_hinge_script: Path
):
    session = pendulum_session.session
    [swing, swing_back] = CellScript.from_file(path=swing_the_hinge_script).cells
    session.perform(cell=swing, namespace=pendulum_session.namespace)
    session.take_snapshot(name="swung")
    position_when_swung = pendulum_session.hinge_position
    session.perform(cell=swing_back, namespace=pendulum_session.namespace)

    session.restore_snapshot(name="swung")

    assert pendulum_session.hinge_position == position_when_swung


def test_restoring_an_unknown_snapshot_is_refused(pendulum_session: PendulumSession):
    with pytest.raises(UnknownSnapshotError):
        pendulum_session.session.restore_snapshot(name="never taken")


# %% performing cells


def test_a_performed_cell_is_kept_with_its_reports(
    pendulum_session: PendulumSession, swing_the_hinge_script: Path
):
    session = pendulum_session.session
    [swing, _] = CellScript.from_file(path=swing_the_hinge_script).cells

    session.perform(cell=swing, namespace=pendulum_session.namespace)

    [performed] = session.history
    assert performed.cell == swing
    assert performed.finished
    assert [report.label for report in performed.reports] == ["swung to half a radian"]


def test_a_cell_is_performed_in_the_namespace_it_is_given(
    pendulum_session: PendulumSession, swing_the_hinge_script: Path
):
    [swing, _] = CellScript.from_file(path=swing_the_hinge_script).cells

    pendulum_session.session.perform(cell=swing, namespace=pendulum_session.namespace)

    assert pendulum_session.namespace["swung_to"] == pendulum_session.hinge_position


def test_a_cell_that_raises_is_kept_unfinished(
    pendulum_session: PendulumSession, divide_by_zero_script: Path
):
    session = pendulum_session.session
    [divide] = CellScript.from_file(path=divide_by_zero_script).cells

    with pytest.raises(ZeroDivisionError):
        session.perform(cell=divide, namespace=pendulum_session.namespace)

    [performed] = session.history
    assert not performed.finished
    assert [report.label for report in performed.reports] == ["about to divide"]


# %% going back and replaying


def test_reverting_the_last_cell_returns_to_the_moment_before_it(
    pendulum_session: PendulumSession, swing_the_hinge_script: Path
):
    session = pendulum_session.session
    [swing, swing_back] = CellScript.from_file(path=swing_the_hinge_script).cells
    session.perform(cell=swing, namespace=pendulum_session.namespace)
    position_after_the_first_cell = pendulum_session.hinge_position
    session.perform(cell=swing_back, namespace=pendulum_session.namespace)

    reverted = session.revert_last_cell()

    assert reverted == swing_back
    assert pendulum_session.hinge_position == position_after_the_first_cell
    assert [performed.cell for performed in session.history] == [swing]


def test_replaying_the_cells_ends_where_they_ended_before(
    pendulum_session: PendulumSession, swing_the_hinge_script: Path
):
    session = pendulum_session.session
    cells = CellScript.from_file(path=swing_the_hinge_script).cells
    for cell in cells:
        session.perform(cell=cell, namespace=pendulum_session.namespace)
    position_after_the_cells = pendulum_session.hinge_position

    session.replay(first_cell=0, namespace=pendulum_session.namespace)

    assert pendulum_session.hinge_position == pytest.approx(
        position_after_the_cells, abs=1e-9
    )
    assert [performed.cell for performed in session.history] == cells


def test_replaying_from_a_cell_that_was_never_performed_is_refused(
    pendulum_session: PendulumSession,
):
    with pytest.raises(NoPerformedCellError):
        pendulum_session.session.replay(
            first_cell=0, namespace=pendulum_session.namespace
        )
