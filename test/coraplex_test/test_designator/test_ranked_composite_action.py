from coraplex.action_belief.action_belief_query import ActionBeliefQuery
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import UnderspecifiedNode
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    PickAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose


def test_transport_action_substeps_are_ranked_without_new_code(
    mutable_model_world, rclpy_node, monkeypatch
):
    """
    ``TransportAction`` builds its internal navigate/pick-up/place steps with
    ``a(NavigateAction)(...)``, ``a(PickUpAction)(...)`` and ``a(PlaceAction)(...)`` --
    real underspecified statements that expand into ``UnderspecifiedNode`` instances
    migrated into the outer plan by ``ActionDescription.add_subplan`` (via
    ``Plan._migrate_nodes_from_plan``, which reassigns ``.plan`` -- and therefore
    ``.context`` -- to the outer plan).

    ``UnderspecifiedNode._build_action_iterator`` only reads ``self.designator_type``
    and ``self.plan.context``, with no dependency on nesting depth, so all three
    registered sub-action types must already be ranked by
    ``ActionBeliefQuery.rank_grounded_actions`` with no new code for composites --
    verified here by spying on the real method (still calling through to it), not by
    replacing it.
    """
    world, view, context = mutable_model_world

    ranked_action_types = []
    original_rank_grounded_actions = ActionBeliefQuery.rank_grounded_actions

    def spying_rank_grounded_actions(self, actions):
        ranked_action_types.append(self.action_type)
        return original_rank_grounded_actions(self, actions)

    monkeypatch.setattr(
        ActionBeliefQuery, "rank_grounded_actions", spying_rank_grounded_actions
    )

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            TransportAction(
                world.get_body_by_name("milk.stl"),
                Pose.from_xyz_rpy(2.37, 2.5, 1.05, reference_frame=world.root),
                Arms.RIGHT,
            ),
        ],
        context=context,
    )

    plan.notify()
    executable = plan.parse()
    with simulated_robot:
        executable.execute()

    assert {NavigateAction, PickUpAction, PlaceAction}.issubset(
        set(ranked_action_types)
    )


def test_pick_and_place_action_builds_no_underspecified_substeps(
    immutable_model_world,
):
    """
    Unlike ``TransportAction``, ``PickAndPlaceAction`` builds its internal
    ``PickUpAction``/``PlaceAction`` as concrete instances directly -- ``arm`` and
    ``grasp_description`` come straight from its own already-resolved fields, not from
    an ``a(...)`` underspecified statement.

    There is therefore no
    ``UnderspecifiedNode`` for the ranking wiring to attach to, confirmed here
    structurally (building the sub-plan only, no physical execution).
    """
    world, view, context = immutable_model_world
    action = PickAndPlaceAction(
        object_designator=world.get_body_by_name("milk.stl"),
        target_location=Pose.from_xyz_rpy(2.37, 2.5, 1.05, reference_frame=world.root),
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.right_arm.end_effector,
        ),
    )
    execute_single(action_like=action, context=context)

    sub_plan_root = action.action_plan
    nodes = [sub_plan_root] + sub_plan_root.descendants

    assert not any(isinstance(node, UnderspecifiedNode) for node in nodes)


def test_move_and_place_action_builds_no_underspecified_substeps(
    immutable_model_world,
):
    """
    Same reasoning as ``PickAndPlaceAction``: ``MoveAndPlaceAction`` builds
    ``NavigateAction``, ``FaceAtAction`` and ``PlaceAction`` as concrete instances from
    its own already-resolved fields, so there is nothing for the ranking wiring to rank.
    """
    world, view, context = immutable_model_world
    action = MoveAndPlaceAction(
        standing_position=Pose.from_xyz_rpy(1.0, 1.0, 0.0, reference_frame=world.root),
        object_designator=world.get_body_by_name("milk.stl"),
        target_location=Pose.from_xyz_rpy(2.37, 2.5, 1.05, reference_frame=world.root),
        arm=Arms.RIGHT,
    )
    execute_single(action_like=action, context=context)

    sub_plan_root = action.action_plan
    nodes = [sub_plan_root] + sub_plan_root.descendants

    assert not any(isinstance(node, UnderspecifiedNode) for node in nodes)


def test_move_and_pick_up_action_builds_no_underspecified_substeps(
    immutable_model_world,
):
    """
    Same reasoning as ``PickAndPlaceAction``: ``MoveAndPickUpAction`` builds
    ``NavigateAction``, ``FaceAtAction`` and ``PickUpAction`` as concrete instances from
    its own already-resolved fields, so there is nothing for the ranking wiring to rank.
    """
    world, view, context = immutable_model_world
    action = MoveAndPickUpAction(
        standing_position=Pose.from_xyz_rpy(1.0, 1.0, 0.0, reference_frame=world.root),
        object_designator=world.get_body_by_name("milk.stl"),
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.right_arm.end_effector,
        ),
    )
    execute_single(action_like=action, context=context)

    sub_plan_root = action.action_plan
    nodes = [sub_plan_root] + sub_plan_root.descendants

    assert not any(isinstance(node, UnderspecifiedNode) for node in nodes)
