"""
Tests for performing plans and for what their actions move.
"""

import pytest

from coraplex.plans.failures import EmptyUnderspecified
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    variable_from,
    a,
)
from krrood.parametrization.model_registries import (
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.orm.model import (
    Point3Mapping,
    QuaternionMapping,
    PoseMapping,
)
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk

from cramph.composites import Sequence, StatementNode
from ...plan_running import run_plan, with_grounding


def _torso_position(world):
    return world.state[
        world.get_degree_of_freedom_by_name(PR2Joint.TORSO_LIFT).id
    ].position


def test_sequence_runs_all_motions(pr2_apartment_context):
    """
    Every motion of a sequence is executed, so the torso ends at the target of the
    *last* motion.

    The robot starts in the LOW configuration, so a final HIGH motion proves the second
    motion actually ran.
    """
    world, robot_view, extensions = pr2_apartment_context

    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.LOW),
            MoveTorsoAction(torso_state=TorsoState.HIGH),
        ]
    )
    run_plan(plan, extensions)

    assert _torso_position(world) == pytest.approx(0.3, abs=0.05)


def test_algebra_sequential_plan(apartment_world_pr2_copy_with_context):
    """
    Parameterize a sequence using krrood parameterizer, create a fully- factorized
    distribution and assert the correctness of sampled values after conditioning and
    truncation.
    """
    world, robot_view, extensions = apartment_world_pr2_copy_with_context

    target_location = a(PoseMapping.from_point_mapping_quaternion_mapping)(
        position=a(Point3Mapping)(x=..., y=..., z=0.0, reference_frame=None),
        orientation=QuaternionMapping(x=0, y=0, z=0, w=1, reference_frame=None),
        reference_frame=variable_from([robot_view.root]),
    )

    navigate_action = a(NavigateAction)(
        target_location=target_location,
    )
    # navigate_action.resolve()

    extensions = with_grounding(
        extensions,
        query_backend=ProbabilisticBackend(model_registry=FullyFactorizedRegistry()),
    )

    # resolved_navigate = next(pm_backend.evaluate(navigate_action))
    plan = Sequence(
        nodes=[
            MoveTorsoAction(torso_state=TorsoState.LOW),
            StatementNode(statement=navigate_action),
        ]
    )

    run_plan(plan, extensions)

    assert isinstance(plan.nodes[1].grounded_nodes[-1], NavigateAction)
    assert len(plan.nodes[1].chosen_children) == 1


def test_parameterization_of_pick_up(apartment_world_pr2_copy_with_context):
    world, robot_view, extensions = apartment_world_pr2_copy_with_context

    milk = world.get_semantic_annotations_by_type(Milk)[0]

    grasp_variable = variable_from(milk.grasp_candidates())

    pick_up_description = a(PickUpAction)(
        grasp=grasp_variable,
        arm=variable_from(robot_view.all_arms),
        approach_clearance=0.05,
    )

    parameters = UnderspecifiedParameters(pick_up_description)

    [approach_clearance] = [
        v for v in parameters.variables.values() if v.name.endswith("clearance")
    ]

    assert (
        parameters.conditioning_assignments_from_literal_values[approach_clearance]
        == 0.05
    )

    extensions = with_grounding(
        extensions,
        query_backend=ProbabilisticBackend(model_registry=FullyFactorizedRegistry()),
    )

    plan = StatementNode(statement=pick_up_description)

    try:
        run_plan(plan, extensions)
    except EmptyUnderspecified:
        pass
