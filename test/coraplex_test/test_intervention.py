from coraplex.action_belief import intervention
from coraplex.action_belief.intervention import InterventionResult
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from .test_action_belief_query import _right_front_grasp

# %% diagnose()


def test_diagnose_finds_the_arm_that_fixes_an_infeasible_open_action(
    immutable_model_world,
):
    """
    Same handle ``test_action_belief_query.py`` uses: only the left arm is reachable, so
    diagnosing the right-arm attempt must report switching to left.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")
    action = OpenAction(handle, Arms.RIGHT)
    execute_single(action_like=action, context=context)

    result = intervention.diagnose(action)

    assert result.fix == ("arm", Arms.RIGHT, Arms.LEFT)
    assert result.other_fixes_count == 0
    # OpenAction registers only "arm", with domain [LEFT, RIGHT]: starting at
    # RIGHT, the only other value to try is LEFT.
    assert result.trials == [("arm", Arms.RIGHT, Arms.LEFT, True)]


def test_diagnose_reports_no_fix_when_nothing_reproduces_success(
    immutable_model_world,
):
    """
    Same unreachable milk pose ``test_action_belief_query.py`` uses for
    ``test_run_reports_zero_posterior_when_nothing_is_feasible``: no bucket-A
    perturbation of arm/approach/alignment can make the pick-up reachable.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        5, 5, 0.7, 0, 0, 0
    )
    action = PickUpAction(milk, Arms.RIGHT, _right_front_grasp(view))
    execute_single(action_like=action, context=context)

    result = intervention.diagnose(action)

    assert result.fix is None
    assert result.other_fixes_count == 0
    assert len(result.trials) == 6
    assert all(passed is False for *_, passed in result.trials)


def test_diagnose_reports_unregistered_action_types(immutable_model_world):
    world, view, context = immutable_model_world
    action = ParkArmsAction(Arms.BOTH)
    execute_single(action_like=action, context=context)

    result = intervention.diagnose(action)

    assert result.fix is None
    assert result.trials == []


# %% InterventionResult.__str__


def test_intervention_result_str_reports_the_fix():
    result = InterventionResult(action_type=OpenAction, fix=("arm", "RIGHT", "LEFT"))

    assert str(result) == (
        "OpenAction: arm RIGHT -> LEFT fixes it (pre_condition: fail -> pass)"
    )


def test_intervention_result_str_reports_extra_fixes_count():
    result = InterventionResult(
        action_type=OpenAction, fix=("arm", "RIGHT", "LEFT"), other_fixes_count=2
    )

    assert str(result) == (
        "OpenAction: arm RIGHT -> LEFT fixes it (pre_condition: fail -> pass) "
        "(+2 other fixes found)"
    )


def test_intervention_result_str_reports_no_fix():
    result = InterventionResult(action_type=OpenAction, fix=None)

    assert str(result) == (
        "OpenAction: no bucket A change reproduces success -- likely outside this "
        "pipeline's reach"
    )


def test_intervention_result_str_reports_unregistered_action_type():
    result = InterventionResult(action_type=ParkArmsAction, fix=None)

    assert str(result) == (
        "ParkArmsAction: not a registered action type -- outside this pipeline's "
        "reach"
    )


# %% format_suggestions()


def test_format_suggestions_joins_results_with_newlines():
    results = [
        InterventionResult(action_type=OpenAction, fix=("arm", "RIGHT", "LEFT")),
        InterventionResult(action_type=ParkArmsAction, fix=None),
    ]

    assert intervention.format_suggestions(results) == f"{results[0]}\n{results[1]}"
