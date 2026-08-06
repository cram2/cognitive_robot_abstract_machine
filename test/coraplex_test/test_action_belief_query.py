import pytest

from coraplex.action_belief.action_belief_query import (
    ActionBeliefQuery,
    _candidate_key,
)
from coraplex.action_belief.results import ActionBeliefResult
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def _right_front_grasp(view):
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.right_arm.end_effector,
    )


def test_enumerate_candidates_is_full_cartesian_product(immutable_model_world):
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")

    query = ActionBeliefQuery(
        action_type=PickUpAction,
        fixed_kwargs={
            "object_designator": milk,
            "grasp_description": _right_front_grasp(view),
        },
        context=context,
    )

    candidates = list(query.enumerate_candidates())

    assert len(candidates) == 2 * len(ApproachDirection) * len(VerticalAlignment)
    assert {candidate["arm"] for candidate in candidates} == {Arms.LEFT, Arms.RIGHT}
    assert {
        candidate["grasp_description.approach_direction"] for candidate in candidates
    } == set(ApproachDirection)
    assert {
        candidate["grasp_description.vertical_alignment"] for candidate in candidates
    } == set(VerticalAlignment)


def test_materialize_kwargs_resyncs_end_effector_to_the_candidates_arm(
    immutable_model_world,
):
    """
    ``arm`` and ``grasp_description.approach_direction``/``vertical_alignment`` are
    independent choice points, but the grasp orientation is computed relative to
    ``grasp_description.end_effector``, so it must track whichever ``arm`` a candidate
    picks rather than staying fixed to the ``fixed_kwargs`` template.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")

    query = ActionBeliefQuery(
        action_type=PickUpAction,
        fixed_kwargs={
            "object_designator": milk,
            "grasp_description": _right_front_grasp(view),
        },
        context=context,
    )

    left_candidate = {
        "arm": Arms.LEFT,
        "grasp_description.approach_direction": ApproachDirection.BACK,
        "grasp_description.vertical_alignment": VerticalAlignment.TOP,
    }
    kwargs = query._materialize_kwargs(left_candidate, robot=view)

    assert kwargs["arm"] == Arms.LEFT
    assert kwargs["grasp_description"].end_effector is view.left_arm.end_effector
    assert kwargs["grasp_description"].approach_direction == ApproachDirection.BACK
    assert kwargs["grasp_description"].vertical_alignment == VerticalAlignment.TOP
    assert kwargs["object_designator"] is milk


def test_run_ranks_feasible_candidates_and_rejects_the_rest(immutable_model_world):
    """
    End to end, on the same reachable milk pose :mod:`test_pose_validator` uses for
    ``test_is_object_reachable_by_reachable``: 6 of the 24 candidates are feasible
    (all right-arm), and the rest - both left-arm and the other right-arm
    approach/alignment combinations - are rejected.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0
    )

    query = ActionBeliefQuery(
        action_type=PickUpAction,
        fixed_kwargs={
            "object_designator": milk,
            "grasp_description": _right_front_grasp(view),
        },
        context=context,
    )

    result = query.run()

    assert isinstance(result, ActionBeliefResult)
    assert result.candidates_enumerated == 24
    assert result.candidates_feasible == 6
    assert result.chosen == {
        "arm": Arms.RIGHT,
        "grasp_description.approach_direction": ApproachDirection.FRONT,
        "grasp_description.vertical_alignment": VerticalAlignment.NoAlignment,
    }
    assert result.posterior == pytest.approx(1 / 6)
    assert len(result.ranked_candidates) == 6
    assert all(
        candidate["arm"] == Arms.RIGHT for candidate, _ in result.ranked_candidates
    )
    assert len(result.rejected_examples) == 18
    # Every left-arm combination is rejected; the rest are the 6 right-arm
    # combinations other than the one that is feasible.
    assert sum(
        candidate["arm"] == Arms.LEFT for candidate, _ in result.rejected_examples
    ) == len(ApproachDirection) * len(VerticalAlignment)
    assert all(reason for _, reason in result.rejected_examples)


def test_run_reports_zero_posterior_when_nothing_is_feasible(immutable_model_world):
    """
    End to end, on the same unreachable milk pose
    ``test_is_object_reachable_by_not_reachable`` uses: no arm can reach it.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        5, 5, 0.7, 0, 0, 0
    )

    query = ActionBeliefQuery(
        action_type=PickUpAction,
        fixed_kwargs={
            "object_designator": milk,
            "grasp_description": _right_front_grasp(view),
        },
        context=context,
    )

    result = query.run()

    assert result.candidates_enumerated == 24
    assert result.candidates_feasible == 0
    assert result.chosen == {}
    assert result.posterior == 0.0
    assert result.ranked_candidates == []
    assert len(result.rejected_examples) == 24


def test_prior_reweights_the_chosen_candidate(immutable_model_world):
    """
    Among the 6 feasible right-arm candidates on the reachable milk pose, a strongly
    favored one must outrank the otherwise-first-found default.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0
    )

    favored_candidate = {
        "arm": Arms.RIGHT,
        "grasp_description.approach_direction": ApproachDirection.BACK,
        "grasp_description.vertical_alignment": VerticalAlignment.TOP,
    }

    result = ActionBeliefQuery(
        action_type=PickUpAction,
        fixed_kwargs={
            "object_designator": milk,
            "grasp_description": _right_front_grasp(view),
        },
        context=context,
        prior={_candidate_key(favored_candidate): 1000.0},
    ).run()

    assert result.candidates_feasible == 6
    assert result.chosen == favored_candidate
    assert result.posterior > 1 / 6


def test_evaluate_candidate_does_not_mutate_the_live_world(immutable_model_world):
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")
    live_pose_before = handle.global_pose.to_position().to_np()

    ActionBeliefQuery(
        action_type=OpenAction,
        fixed_kwargs={"object_designator": handle},
        context=context,
    ).run()

    assert (handle.global_pose.to_position().to_np() == live_pose_before).all()


def test_run_on_open_action_has_a_single_choice_point(immutable_model_world):
    """
    ``OpenAction`` registers only ``arm`` as a choice point, so this checks real
    reachability twice (once per arm) on the fixed apartment scene.
    """
    world, view, context = immutable_model_world

    result = ActionBeliefQuery(
        action_type=OpenAction,
        fixed_kwargs={"object_designator": world.get_body_by_name("handle_cab10_m")},
        context=context,
    ).run()

    assert result.candidates_enumerated == 2
    assert result.candidates_feasible == 1
    assert result.chosen["arm"] == Arms.LEFT
