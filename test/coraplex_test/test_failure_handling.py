from __future__ import annotations

from datetime import timedelta

from giskardpy.motion_statechart.exceptions import NoProgressError
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing

from coraplex.plans.failures import MotionMadeNoProgress, PlanFailure

# %% fixtures


def stalled_motion() -> NoProgressError:
    """
    :return: The error a motion statechart cancels itself with when it stops converging.
    """
    return NoProgressError(
        progress_monitor=StillProgressing(
            monitored_node=EndMotion(), timeout=timedelta(seconds=3)
        )
    )


# %% a stalled motion crossing into a plan


def test_a_stalled_motion_is_a_failure_a_plan_can_recover_from():
    """
    A plan chooses an alternative by catching :class:`PlanFailure`, so a stall has to be
    one rather than a motion error the plan would have to name separately.
    """
    assert isinstance(MotionMadeNoProgress(stalled_motion()), PlanFailure)


def test_a_stalled_motion_keeps_the_stall_it_was_made_from():
    """
    The stall names the tasks that stopped converging, which is the only account of why
    the attempt failed.
    """
    stall = stalled_motion()

    assert MotionMadeNoProgress(stall).no_progress is stall


def test_a_stalled_motion_reports_what_the_stall_reported():
    """
    Wrapping the stall must not cost the reader the tasks and timeout it names.
    """
    stall = stalled_motion()

    gave_up = MotionMadeNoProgress(stall)

    assert gave_up.error_message() == stall.error_message()
    assert gave_up.suggest_correction() == stall.suggest_correction()
