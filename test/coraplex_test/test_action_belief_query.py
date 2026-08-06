import pytest

from coraplex.action_belief.action_belief_query import (
    ActionBeliefQuery,
    _candidate_key,
)
from coraplex.action_belief.results import ActionBeliefResult
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose


def _right_front_grasp(view):
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.right_arm.end_effector,
    )


# %% enumerate_candidates and _materialize_kwargs


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


# %% extract_candidate, perturbed_action, predict_feasible


def test_extract_candidate_reads_choice_point_values_off_an_action(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    action = PickUpAction(milk, Arms.LEFT, _right_front_grasp(view))
    query = ActionBeliefQuery(
        action_type=PickUpAction, fixed_kwargs={}, context=context
    )

    assert query.extract_candidate(action) == {
        "arm": Arms.LEFT,
        "grasp_description.approach_direction": ApproachDirection.FRONT,
        "grasp_description.vertical_alignment": VerticalAlignment.NoAlignment,
    }


def test_perturbed_action_overrides_one_field_and_keeps_the_rest(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    action = OpenAction(world.get_body_by_name("handle_cab10_m"), Arms.RIGHT)
    execute_single(action_like=action, context=context)
    query = ActionBeliefQuery(
        action_type=OpenAction,
        fixed_kwargs=action.designator_parameter,
        context=context,
    )

    perturbed = query.perturbed_action(action, "arm", Arms.LEFT)

    assert perturbed is not action
    assert perturbed.arm == Arms.LEFT
    assert perturbed.object_designator is action.object_designator


def test_predict_feasible_matches_run_on_the_same_scenario(immutable_model_world):
    """
    Same reachable milk pose ``test_run_ranks_feasible_candidates_and_rejects_the_rest``
    uses: the chosen candidate from ``run()`` must independently predict feasible via
    ``predict_feasible`` on the actual grounded action.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0
    )
    action = PickUpAction(milk, Arms.RIGHT, _right_front_grasp(view))
    execute_single(action_like=action, context=context)
    query = ActionBeliefQuery(
        action_type=PickUpAction,
        fixed_kwargs=action.designator_parameter,
        context=context,
    )

    assert query.predict_feasible(action) is True

    infeasible_action = query.perturbed_action(action, "arm", Arms.LEFT)

    assert query.predict_feasible(infeasible_action) is False


# %% run() / ActionBeliefResult


def test_run_ranks_feasible_candidates_and_rejects_the_rest(immutable_model_world):
    """
    End to end, on the same reachable milk pose :mod:`test_pose_validator` uses for.

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
    assert str(result) == (
        f"PickUpAction grounded: arm={Arms.RIGHT}, "
        f"approach_direction={ApproachDirection.FRONT}, "
        f"vertical_alignment={VerticalAlignment.NoAlignment} "
        f"(6/24 feasible, p=0.17)"
    )


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
    assert str(result) == "PickUpAction: no feasible candidate (0/24)"


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


# %% rank_grounded_actions


def test_rank_grounded_actions_bounds_a_long_candidate_stream(immutable_model_world):
    """
    ``rank_grounded_actions`` must not eagerly exhaust its whole input: each
    candidate re-checks ``pre_condition`` against a fresh deep copy of the world, so
    for a registered type with a bucket-B choice point (e.g.
    ``NavigateAction.target_location``) -- where the query backend can supply far
    more candidates than are ever useful -- exhausting everything up front would
    make ranking impractically expensive.
    """
    world, view, context = immutable_model_world

    consumed = []

    def many_candidates():
        for i in range(1000):
            consumed.append(i)
            yield NavigateAction(
                Pose.from_xyz_rpy(float(i), 0.0, 0.0, reference_frame=world.root)
            )

    ranked = ActionBeliefQuery(
        action_type=NavigateAction,
        fixed_kwargs={},
        context=context,
        max_location_candidates=5,
    ).rank_grounded_actions(many_candidates())

    first_five = [next(ranked) for _ in range(5)]

    assert len(consumed) == 5
    assert sorted(
        float(action.target_location.to_position().x) for action in first_five
    ) == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_rank_grounded_actions_still_yields_candidates_beyond_the_bound(
    immutable_model_world,
):
    """
    Candidates past :attr:`ActionBeliefQuery.max_location_candidates` are not
    dropped -- they are appended afterward, unranked, in their original order, so
    every candidate the query backend produced remains triable.
    """
    world, view, context = immutable_model_world

    def candidates():
        for i in range(7):
            yield NavigateAction(
                Pose.from_xyz_rpy(float(i), 0.0, 0.0, reference_frame=world.root)
            )

    ranked = list(
        ActionBeliefQuery(
            action_type=NavigateAction,
            fixed_kwargs={},
            context=context,
            max_location_candidates=3,
        ).rank_grounded_actions(candidates())
    )

    assert len(ranked) == 7
    unranked_tail = [action.target_location.to_position().x for action in ranked[3:]]
    assert unranked_tail == [3.0, 4.0, 5.0, 6.0]
