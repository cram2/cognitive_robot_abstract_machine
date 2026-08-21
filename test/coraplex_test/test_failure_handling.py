import pytest

from coraplex.datastructures.enums import Arms
from coraplex.exceptions import ConditionNotSatisfied
from coraplex.execution_environment import simulated_robot
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.executables import ConditionExecutable
from coraplex.plans.factories import execute_single, try_in_order
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


def test_condition_not_satisfied_suggest_correction_diagnoses_the_failed_action(
    immutable_model_world,
):
    """
    Same handle as above: only the left arm is reachable.

    ``ConditionExecutable.execute()`` is the ``ConditionNotSatisfied`` raise site
    that checks the condition before constructing the exception, so it can set
    ``failed_action`` -- exercised directly here, since nothing in the live
    plan-execution path currently constructs a ``ConditionExecutable`` (conditions
    during real execution are instead monitored by Giskard's motion statechart, see
    ``GiskardExecutable._add_condition_monitors``).
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")
    action = OpenAction(handle, Arms.RIGHT)
    node = execute_single(action_like=action, context=context)

    condition_node = ConditionNode(
        condition=action.pre_condition(
            action.bound_variables, context, action.designator_parameter
        ),
        pre_condition=True,
        action_node=node,
    )
    executable = ConditionExecutable(context=context, condition_node=condition_node)

    with pytest.raises(ConditionNotSatisfied) as excinfo:
        executable.execute()

    assert excinfo.value.failed_action is action
    assert excinfo.value.suggest_correction() == (
        f"OpenAction: arm {Arms.RIGHT} -> {Arms.LEFT} fixes it "
        f"(pre_condition: fail -> pass)"
    )
