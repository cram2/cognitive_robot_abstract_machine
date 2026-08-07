from coraplex.action_belief import intervention
from coraplex.action_belief.action_belief_query import ActionBeliefQuery
from coraplex.action_belief.intervention import InterventionResult, ParameterChange
from coraplex.datastructures.enums import Arms, ApproachDirection
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

    assert result.fix == (ParameterChange("arm", Arms.RIGHT, Arms.LEFT),)
    assert result.other_fixes_count == 0
    # OpenAction registers only "arm", with domain [LEFT, RIGHT]: starting at
    # RIGHT, the only other value to try is LEFT.
    assert len(result.trials) == 1
    assert result.trials[0].changes == (ParameterChange("arm", Arms.RIGHT, Arms.LEFT),)
    assert result.trials[0].passed is True


def test_diagnose_reports_no_fix_when_nothing_reproduces_success(
    immutable_model_world,
):
    """
    Same unreachable milk pose ``test_action_belief_query.py`` uses for
    ``test_run_reports_zero_posterior_when_nothing_is_feasible``: no single field, nor
    any pair of fields, changed together can make the pick-up reachable, so diagnose()
    must exhaust both the single-field and pairwise search before giving up.
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
    # PickUpAction registers arm(2)/approach_direction(4)/vertical_alignment(3): 6
    # single-field trials, then 3 pairs (1x3 + 1x2 + 3x2 = 11) once none of those pass.
    assert len(result.trials) == 17
    assert all(trial.passed is False for trial in result.trials)


def test_diagnose_reports_unregistered_action_types(immutable_model_world):
    world, view, context = immutable_model_world
    action = ParkArmsAction(Arms.BOTH)
    execute_single(action_like=action, context=context)

    result = intervention.diagnose(action)

    assert result.fix is None
    assert result.trials == []


def test_diagnose_finds_a_fix_that_needs_two_fields_changed_together(
    immutable_model_world, monkeypatch
):
    """
    Some failures have no single-field fix, but flipping two registered fields together
    does.

    ``predict_feasible`` is replaced with a deterministic rule (arm must be LEFT *and*
    approach_direction must be BACK) so the scenario doesn't depend on a specific
    world's reachability geometry: diagnose() must escalate past its single-field search
    and find that pair rather than reporting "no fix".
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    action = PickUpAction(milk, Arms.RIGHT, _right_front_grasp(view))
    execute_single(action_like=action, context=context)

    def only_left_arm_with_back_approach_passes(self, candidate):
        return (
            candidate.arm == Arms.LEFT
            and candidate.grasp_description.approach_direction == ApproachDirection.BACK
        )

    monkeypatch.setattr(
        ActionBeliefQuery, "predict_feasible", only_left_arm_with_back_approach_passes
    )

    result = intervention.diagnose(action)

    assert result.fix == (
        ParameterChange("arm", Arms.RIGHT, Arms.LEFT),
        ParameterChange(
            "grasp_description.approach_direction",
            ApproachDirection.FRONT,
            ApproachDirection.BACK,
        ),
    )
    assert result.other_fixes_count == 0
    # 6 single-field trials (all fail, since no single field satisfies both
    # conditions at once), then all 3 pairs (1x3 + 1x2 + 3x2 = 11) are evaluated --
    # diagnose() finishes a whole combination size before deciding whether to
    # report a fix, so it can also report other_fixes_count accurately.
    assert len(result.trials) == 17
    assert sum(trial.passed for trial in result.trials) == 1


# %% InterventionResult.__str__


def test_intervention_result_str_reports_the_fix():
    result = InterventionResult(
        action_type=OpenAction, fix=(ParameterChange("arm", "RIGHT", "LEFT"),)
    )

    assert str(result) == (
        "OpenAction: arm RIGHT -> LEFT fixes it (pre_condition: fail -> pass)"
    )


def test_intervention_result_str_reports_a_multi_field_fix():
    result = InterventionResult(
        action_type=PickUpAction,
        fix=(
            ParameterChange("arm", "RIGHT", "LEFT"),
            ParameterChange("grasp_description.approach_direction", "FRONT", "BACK"),
        ),
    )

    assert str(result) == (
        "PickUpAction: arm RIGHT -> LEFT, grasp_description.approach_direction "
        "FRONT -> BACK fixes it (pre_condition: fail -> pass)"
    )


def test_intervention_result_str_reports_extra_fixes_count():
    result = InterventionResult(
        action_type=OpenAction,
        fix=(ParameterChange("arm", "RIGHT", "LEFT"),),
        other_fixes_count=2,
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
        InterventionResult(
            action_type=OpenAction, fix=(ParameterChange("arm", "RIGHT", "LEFT"),)
        ),
        InterventionResult(action_type=ParkArmsAction, fix=None),
    ]

    assert intervention.format_suggestions(results) == f"{results[0]}\n{results[1]}"
