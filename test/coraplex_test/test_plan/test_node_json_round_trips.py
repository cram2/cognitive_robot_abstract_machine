"""
Every action and perception node class of coraplex sent through JSON inside a
statechart.

An underspecified node is left out: it is grounded by the process that wrote it, which
sends the nodes it chose instead.
"""

from __future__ import annotations

from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from dataclasses import dataclass, replace

import pytest
from typing_extensions import Callable

from coraplex import perception
from coraplex.datastructures.enums import DetectionTechnique
from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.perception import PerceptionQuery, PerceptionSource, PerceptionTask
from coraplex.robot_plans.actions import core
from coraplex.robot_plans.actions.composite import (
    carrying,
    facing,
    tool_based,
    transporting,
)
from coraplex.robot_plans.actions.composite.carrying import CarryAction
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    PickAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.misc import DetectAction, MoveToReach
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    FaceAtAction,
    LookAtAction,
    NavigateAction,
    PathPlanningNavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import (
    FixateObjectAction,
    GraspingAction,
    PickUpAction,
    ReachAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    FollowToolCenterPointPathAction,
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
    MoveFingersToWidthAction,
)
from coraplex.plans.executors import PlanExecutor
from cramph.context import ContextExtension, StatechartContext
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CuttingKnife,
    Door,
    Elevator,
    GroundFloor,
    Handle,
    Milk,
    PouringCup,
    Sponge,
    Tool,
    Whisk,
)
from krrood_spatial_math.spatial_types import Pose, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.specifications.connections import (
    ScrewAxisConnectionSpecification,
)
from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Scale,
    VolumetricBoundingBox,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

from ..test_designator.test_multi_robot_action_designator import _handle_annotation
from ..test_tool_based_actions import _add_box_body, _attach_box_to_gripper
from ..conftest import physically_simulated_executor
from ...plan_running import robot_extensions, simulated_executor
from ...statechart_json_round_trip import (
    NodeRoundTripCase,
    NodeRoundTripTests,
    WorldFixture,
    concrete_node_types,
)

# %% the executor a case is performed by


@dataclass(kw_only=True)
class ActionRoundTripCase(NodeRoundTripCase):
    """
    The nodes that send one action or perception node class through JSON, sent from and
    received by executors of the kind that performs it.
    """

    executor_for: Callable[[list[ContextExtension]], PlanExecutor] = simulated_executor
    """
    Builds an executor performing the nodes, from the context extensions of their plan.
    """


# %% what the nodes of the cases refer to


def _robot(world: World) -> AbstractRobot:
    return world.get_semantic_annotations_by_type(
        semantic_annotation_type=AbstractRobot
    )[0]


def _arm(world: World) -> Arm:
    return _robot(world).right_arm


def _milk(world: World) -> Milk:
    return world.get_semantic_annotations_by_type(semantic_annotation_type=Milk)[0]


def _pose(world: World) -> Pose:
    return Pose.from_xyz_rpy(x=1.5, y=2.0, z=0.9, reference_frame=world.root)


def _handle(world: World) -> Handle:
    return _handle_annotation(world, "handle_cab10_m")


def _container(world: World) -> Body:
    return _add_box_body(world, "container", (0.2, 0.2, 0.1), (2.4, 2.2, 1.0))


def _tool_body(world: World) -> Body:
    """
    :return: A box the right gripper of the robot of `world` holds.
    """
    return _attach_box_to_gripper(world, _robot(world), "tool", (0.04, 0.04, 0.2), 0.1)


def _tool(world: World, tool_type: type[Tool]) -> Tool:
    """
    :return: A tool of `tool_type` that `world` knows.
    """
    tool = tool_type(root=_tool_body(world))
    with world.modify_world():
        world.add_semantic_annotation(semantic_annotation=tool)
    return tool


def _whole_scene(world: World) -> VolumetricBoundingBox:
    return VolumetricBoundingBox(
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
        min_x=-10,
        min_y=-10,
        min_z=-10,
        max_x=10,
        max_y=10,
        max_z=10,
    )


# %% nodes of the cases


def _navigate(world: World) -> NavigateAction:
    """
    :param world: The world the goal is taken from.
    :return: A navigation to the shared test pose.
    """
    return NavigateAction(planar_goal_pose=Pose2D.from_pose(_pose(world)))


def _face_and_look_at(world: World) -> FaceAndLookAtAction:
    """
    :param world: The world the target is taken from.
    :return: Facing and looking at the shared test pose.
    """
    return FaceAndLookAtAction(
        face_at=FaceAtAction(look_target=_pose(world)),
        look_at=LookAtAction(look_target=_pose(world)),
    )


def _pick_up(world: World) -> PickUpAction:
    """
    :param world: The world the milk is taken from.
    :return: A pick-up of the milk with its first grasp.
    """
    return PickUpAction(
        grasp_candidate=_milk(world).grasp_candidates()[0], arm=_arm(world)
    )


def _place(world: World) -> PlaceAction:
    """
    :return: A place of the milk with the arm of `world`, which has to follow
        :func:`_pick_up`.
    """
    return PlaceAction(arm=_arm(world), graspable=_milk(world), goal_pose=_pose(world))


def _move_and_pick_up(world: World) -> MoveAndPickUpAction:
    return MoveAndPickUpAction(
        navigate=_navigate(world),
        face_and_look_at=_face_and_look_at(world),
        pick_up=_pick_up(world),
    )


def _move_and_place(world: World) -> MoveAndPlaceAction:
    """
    :return: A placing whose carry names the arm that picked the object up.
    """
    return MoveAndPlaceAction(
        carry=CarryAction(navigate=_navigate(world), holding_arms=[_arm(world)]),
        face_and_look_at=_face_and_look_at(world),
        place=_place(world),
    )


def _elevator_navigation(world: World) -> list[StatechartNode]:
    """
    :return: The statechart nodes of navigating an elevator with an opening door to a floor.
    """
    with world.modify_world():
        elevator = Elevator.create_with_new_body_in_world(
            name="elevator",
            world=world,
            scale=Scale(x=2, y=2, z=2),
            parent_connection_specification=ScrewAxisConnectionSpecification(
                screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(
                    axis=Vector3.Z()
                ),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    maximum_speed=1.0
                ),
            ),
        )
        door = Door.create_with_new_body_in_world(
            name="door",
            world=world,
            scale=Scale(x=0.05, y=1, z=2),
            parent_connection_specification=ScrewAxisConnectionSpecification(
                screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(
                    axis=Vector3.Y()
                ),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    lower_position=0.0, upper_position=1.0
                ),
            ),
        )
        elevator.add(part=door)
        ground_floor = GroundFloor.create_with_new_region_in_world(
            name="ground_floor", world=world, scale=Scale(x=40, y=40, z=6)
        )
    return [ElevatorNavigation(elevator=elevator, target_floor=ground_floor)]


def _robot_body_cases() -> list[ActionRoundTripCase]:
    """
    :return: The round trip cases of the actions moving the torso, the grippers and the arms of the robot.
    """
    return [
        ActionRoundTripCase(
            node_type=MoveTorsoAction,
            build_nodes=lambda world: [MoveTorsoAction(torso_state=TorsoState.HIGH)],
        ),
        ActionRoundTripCase(
            node_type=SetGripperAction,
            build_nodes=lambda world: [
                SetGripperAction(
                    gripper_configuration=_arm(
                        world
                    ).end_effector.default_configuration(state_type=GripperState.OPEN)
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=MoveFingersToWidthAction,
            build_nodes=lambda world: [
                MoveFingersToWidthAction(
                    two_finger_gripper=_arm(world).end_effector, width=0.04
                )
            ],
            executor_for=physically_simulated_executor,
        ),
        ActionRoundTripCase(
            node_type=ParkArmsAction,
            build_nodes=lambda world: [ParkArmsAction(arms=_robot(world).all_arms)],
        ),
        ActionRoundTripCase(
            node_type=FollowToolCenterPointPathAction,
            build_nodes=lambda world: [
                FollowToolCenterPointPathAction(
                    arm=_arm(world),
                    target_locations=PoseTrajectory(poses=[_pose(world), _pose(world)]),
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=MoveManipulatorAction,
            build_nodes=lambda world: [
                MoveManipulatorAction(
                    goal_pose=_pose(world),
                    arm=_arm(world),
                    allow_gripper_collision=False,
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=MoveToReach,
            build_nodes=lambda world: [
                MoveToReach(
                    target_pose_offset_robot=Pose2D.from_xy_yaw(0.2, -0.55),
                    reference_T_grasp=_pose(world),
                    hip_rotation=0.0,
                    arm=_arm(world),
                )
            ],
        ),
    ]


def _navigation_and_perception_cases() -> list[ActionRoundTripCase]:
    """
    :return: The cases of the actions that drive, look and perceive.
    """
    return [
        ActionRoundTripCase(
            node_type=NavigateAction, build_nodes=lambda world: [_navigate(world)]
        ),
        ActionRoundTripCase(
            node_type=CarryAction,
            build_nodes=lambda world: [
                CarryAction(navigate=_navigate(world), holding_arms=[_arm(world)])
            ],
        ),
        ActionRoundTripCase(
            node_type=PathPlanningNavigateAction,
            build_nodes=lambda world: [
                PathPlanningNavigateAction(
                    planar_goal_pose=Pose2D.from_pose(_pose(world))
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=LookAtAction,
            build_nodes=lambda world: [LookAtAction(look_target=_pose(world))],
        ),
        ActionRoundTripCase(
            node_type=FaceAtAction,
            build_nodes=lambda world: [FaceAtAction(look_target=_pose(world))],
        ),
        ActionRoundTripCase(
            node_type=FaceAndLookAtAction,
            build_nodes=lambda world: [_face_and_look_at(world)],
        ),
        ActionRoundTripCase(
            node_type=ElevatorNavigation, build_nodes=_elevator_navigation
        ),
        ActionRoundTripCase(
            node_type=DetectAction,
            build_nodes=lambda world: [
                DetectAction(
                    technique=DetectionTechnique.TYPES, object_sem_annotation=Milk
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=PerceptionTask,
            build_nodes=lambda world: [
                PerceptionTask(
                    query=PerceptionQuery(
                        semantic_annotation=Milk,
                        region=_whole_scene(world),
                        robot=_robot(world),
                        world=world,
                    ),
                    answered_by=PerceptionSource.WORLD_MODEL,
                )
            ],
        ),
    ]


def _manipulation_cases() -> list[ActionRoundTripCase]:
    """
    :return: The manipulation actions whose JSON round trip is tested.
    """
    return [
        ActionRoundTripCase(
            node_type=ReachAction,
            build_nodes=lambda world: [
                ReachAction(
                    grasp_candidate=_milk(world).grasp_candidates()[0], arm=_arm(world)
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=GraspingAction,
            build_nodes=lambda world: [
                GraspingAction(
                    grasp_candidate=_milk(world).grasp_candidates()[0], arm=_arm(world)
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=PickUpAction, build_nodes=lambda world: [_pick_up(world)]
        ),
        ActionRoundTripCase(
            node_type=PlaceAction,
            build_nodes=lambda world: [_pick_up(world), _place(world)],
        ),
        ActionRoundTripCase(
            node_type=OpenAction,
            build_nodes=lambda world: [
                OpenAction(
                    grasp_candidate=_handle(world).grasp_candidates()[0],
                    arm=_arm(world),
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=CloseAction,
            build_nodes=lambda world: [
                CloseAction(
                    grasp_candidate=_handle(world).grasp_candidates()[0],
                    arm=_arm(world),
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=FixateObjectAction,
            build_nodes=lambda world: [
                FixateObjectAction(end_effector=_arm(world).end_effector)
            ],
        ),
        ActionRoundTripCase(
            node_type=MoveAndPickUpAction,
            build_nodes=lambda world: [_move_and_pick_up(world)],
        ),
        ActionRoundTripCase(
            node_type=MoveAndPlaceAction,
            build_nodes=lambda world: [_pick_up(world), _move_and_place(world)],
        ),
        ActionRoundTripCase(
            node_type=MoveAndOpenAction,
            build_nodes=lambda world: [
                MoveAndOpenAction(
                    navigate=_navigate(world),
                    face_and_look_at=_face_and_look_at(world),
                    open_container=OpenAction(
                        grasp_candidate=_handle(world).grasp_candidates()[0],
                        arm=_arm(world),
                    ),
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=PickAndPlaceAction,
            build_nodes=lambda world: [
                PickAndPlaceAction(pick_up=_pick_up(world), place=_place(world))
            ],
        ),
        ActionRoundTripCase(
            node_type=TransportAction,
            build_nodes=lambda world: [
                TransportAction(
                    pick_up=_move_and_pick_up(world), place=_move_and_place(world)
                )
            ],
        ),
    ]


def _tool_cases() -> list[ActionRoundTripCase]:
    return [
        ActionRoundTripCase(
            node_type=MixingAction,
            build_nodes=lambda world: [
                MixingAction(
                    container=_container(world),
                    arm=_arm(world),
                    tool=_tool(world, Whisk),
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=CuttingAction,
            build_nodes=lambda world: [
                CuttingAction(
                    object_to_cut=_container(world),
                    arm=_arm(world),
                    tool=_tool(world, CuttingKnife),
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=WipingAction,
            build_nodes=lambda world: [
                WipingAction(
                    arm=_arm(world),
                    tool=_tool(world, Sponge),
                    target_pose=_pose(world),
                )
            ],
        ),
        ActionRoundTripCase(
            node_type=PouringAction,
            build_nodes=lambda world: [
                PouringAction(
                    target_container=_container(world),
                    source_container=_tool(world, PouringCup),
                    arm=_arm(world),
                )
            ],
        ),
    ]


def _cases() -> list[ActionRoundTripCase]:
    """
    :return: One case per action and perception node class, each in the apartment of
        the PR2.
    """
    return [
        replace(case, world_fixture=WorldFixture.PR2_APARTMENT_CONTEXT)
        for case in (
            *_robot_body_cases(),
            *_navigation_and_perception_cases(),
            *_manipulation_cases(),
            *_tool_cases(),
        )
    ]


# %% sender and receiver


@pytest.fixture(params=_cases(), ids=lambda case: case.node_type.__name__)
def case(request: pytest.FixtureRequest) -> ActionRoundTripCase:
    return request.param


@pytest.fixture()
def cases() -> list[ActionRoundTripCase]:
    return _cases()


@pytest.fixture()
def node_types_needing_a_case() -> set[type[StatechartNode]]:
    return set().union(
        *(
            concrete_node_types(package)
            for package in (
                core,
                carrying,
                facing,
                tool_based,
                transporting,
                perception,
            )
        )
    )


@pytest.fixture()
def sending_context(
    case: ActionRoundTripCase, request: pytest.FixtureRequest
) -> StatechartContext:
    world, robot, extensions = request.getfixturevalue(case.world_fixture)
    return case.executor_for(extensions).context


@pytest.fixture()
def receiving_executor(
    case: ActionRoundTripCase, request: pytest.FixtureRequest
) -> StatechartExecutor:
    """
    :return: An executor performing plans of the robot of `case` the way `case` is
        performed, with context extensions of its own.
    """
    world, robot, extensions = request.getfixturevalue(case.world_fixture)
    return case.executor_for(robot_extensions(robot))


# %% round trips


class TestActionRoundTrips(NodeRoundTripTests):
    pass
