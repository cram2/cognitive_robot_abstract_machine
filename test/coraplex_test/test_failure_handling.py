import pytest

from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import try_in_order
from coraplex.plans.failures import AllChildrenFailed
from coraplex.robot_plans.actions.core.container import OpenAction


def test_all_children_failed_suggest_correction_diagnoses_the_failed_open_action(
    immutable_model_world,
):
    """
    Same handle ``test_action_belief_query.py`` uses: only the left arm is
    reachable, so a ``TryInOrderNode`` wrapping only the infeasible right-arm
    ``OpenAction`` must fail with ``AllChildrenFailed``, and its
    ``suggest_correction()`` must report switching to the left arm as the verified
    fix -- not a heuristic guess.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")

    plan = try_in_order([OpenAction(handle, Arms.RIGHT)], context).plan

    with pytest.raises(AllChildrenFailed) as excinfo:
        with simulated_robot:
            plan.perform()

    suggestion = excinfo.value.suggest_correction()

    assert suggestion == (
        f"OpenAction: arm {Arms.RIGHT} -> {Arms.LEFT} fixes it "
        f"(pre_condition: fail -> pass)"
    )
