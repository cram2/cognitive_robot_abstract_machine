"""
Tests for :mod:`coraplex.plans.plan_video_recorder`.

Recording a plan spins up a headless MuJoCo mirror of its world (see
:mod:`semantic_digital_twin.adapters.mujoco_video_recording`), so the tests that actually
record a video follow the same CI-only gating as
``test/semantic_digital_twin_test/test_adapters/test_multi_sim.py``. The guard clause
that rejects a plan that was never performed needs no simulator and always runs.
"""

import os

import pytest

from coraplex.exceptions import PlanNotYetPerformedError
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan_video_recorder import PlanVideoRecorder
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState

only_run_test_in_CI = os.environ.get("CI", "false").lower() == "false"
requires_mujoco_ci = pytest.mark.skipif(
    only_run_test_in_CI,
    reason="Only run MuJoCo-backed video recording tests in CI.",
)


def test_record_raises_if_plan_was_never_performed(immutable_model_world):
    _, _, context = immutable_model_world

    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan

    with pytest.raises(PlanNotYetPerformedError):
        PlanVideoRecorder(plan).record("/tmp/never_performed.mp4")


@requires_mujoco_ci
def test_record_produces_a_video_with_one_caption_per_leaf_node(
    immutable_model_world, tmp_path
):
    _, _, context = immutable_model_world

    plan = sequential(
        [MoveTorsoAction(TorsoState.LOW), MoveTorsoAction(TorsoState.HIGH)],
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()

    leaf_node_count = sum(1 for node in plan.nodes if node.is_leaf)

    output_path = tmp_path / "plan_video.mp4"
    rendered_video = PlanVideoRecorder(plan, frames_per_second=10).record(output_path)

    assert rendered_video.video_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    assert len(rendered_video.captions) == leaf_node_count
    for caption in rendered_video.captions:
        assert caption.start_time <= caption.end_time
    for earlier, later in zip(rendered_video.captions, rendered_video.captions[1:]):
        assert earlier.end_time <= later.start_time
