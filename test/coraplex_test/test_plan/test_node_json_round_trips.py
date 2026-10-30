"""
Every action and perception node class of coraplex sent through JSON inside a
statechart.

An underspecified node is left out: it is grounded by the process that wrote it, which
sends the nodes it chose instead.
"""

from __future__ import annotations

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
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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


@dataclass
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
    return world.get_semantic_annotations_by_type(AbstractRobot)[0]


def _arm(world: World) -> Arm:
    return _robot(world).right_arm


def _milk(world: World) -> Milk:
    return world.get_semantic_annotations_by_type(Milk)[0]


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
        world.add_semantic_annotation(tool)
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
    return NavigateAction(target_location=_pose(world))


def _face_and_look_at(world: World) -> FaceAndLookAtAction:
    return FaceAndLookAtAction(
        face_at=FaceAtAction(target=_pose(world)),
        look_at=LookAtAction(target=_pose(world)),
    )


def _pick_up(world: World) -> PickUpAction:
    return PickUpAction(grasp=_milk(world).grasp_candidates()[0], arm=_arm(world))


def _place(world: World) -> PlaceAction:
    return PlaceAction(object_designator=_milk(world), target_location=_pose(world))


def _move_and_pick_up(world: World) -> MoveAndPickUpAction:
    return MoveAndPickUpAction(
        navigate=_navigate(world),
        face_and_look_at=_face_and_look_at(world),
        pick_up=_pick_up(world),
    )


def _move_and_place(world: World) -> MoveAndPlaceAction:
    return MoveAndPlaceAction(
        carry=CarryAction(navigate=_navigate(world)),
        face_and_look_at=_face_and_look_at(world),
        place=_place(world),
    )


def _elevator_navigation(world: World) -> list[StatechartNode]:
    with world.modify_world():
        elevator = Elevator.create_with_new_body_in_world(
            name="elevator",
            world=world,
            scale=Scale(2, 2, 2),
            parent_connection_specification=PrismaticConnectionSpecification(
                axis=Vector3.Z(),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    maximum_speed=1.0
                ),
            ),
        )
        door = Door.create_with_new_body_in_world(
            name="door",
            world=world,
            scale=Scale(0.05, 1, 2),
            parent_connection_specification=PrismaticConnectionSpecification(
                axis=Vector3.Y(),
                dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    lower_position=0.0, upper_position=1.0
                ),
            ),
        )
        elevator.add(door)
        ground_floor = GroundFloor.create_with_new_region_in_world(
            name="ground_floor", world=world, scale=Scale(40, 40, 6)
        )
    return [ElevatorNavigation(elevator=elevator, target_floor=ground_floor)]


def _robot_body_cases() -> list[ActionRoundTripCase]:
    return [
        ActionRoundTripCase(
            MoveTorsoAction,
            lambda world: [MoveTorsoAction(torso_state=TorsoState.HIGH)],
        ),
        ActionRoundTripCase(
            SetGripperAction,
            lambda world: [
                SetGripperAction(
                    configuration=_arm(world).end_effector.default_configuration(
                        GripperState.OPEN
                    )
                )
            ],
        ),
        ActionRoundTripCase(
            MoveFingersToWidthAction,
            lambda world: [
                MoveFingersToWidthAction(gripper=_arm(world).end_effector, width=0.04)
            ],
            executor_for=physically_simulated_executor,
        ),
        ActionRoundTripCase(
            ParkArmsAction, lambda world: [ParkArmsAction(arms=_robot(world).all_arms)]
        ),
        ActionRoundTripCase(
            FollowToolCenterPointPathAction,
            lambda world: [
                FollowToolCenterPointPathAction(
                    arm=_arm(world),
                    target_locations=PoseTrajectory([_pose(world), _pose(world)]),
                )
            ],
        ),
        ActionRoundTripCase(
            MoveManipulatorAction,
            lambda world: [
                MoveManipulatorAction(
                    target_pose=_pose(world),
                    end_effector=_arm(world).end_effector,
                    allow_gripper_collision=False,
                )
            ],
        ),
        ActionRoundTripCase(
            MoveToReach,
            lambda world: [
                MoveToReach(
                    target_pose_offset_robot=Pose2D.from_xy_yaw(0.2, -0.55),
                    reference_T_grasp=_pose(world),
                    hip_rotation=0.0,
                    end_effector=_arm(world).end_effector,
                )
            ],
        ),
    ]


def _navigation_and_perception_cases() -> list[ActionRoundTripCase]:
    return [
        ActionRoundTripCase(NavigateAction, lambda world: [_navigate(world)]),
        ActionRoundTripCase(
            CarryAction, lambda world: [CarryAction(navigate=_navigate(world))]
        ),
        ActionRoundTripCase(
            PathPlanningNavigateAction,
            lambda world: [PathPlanningNavigateAction(target=_pose(world))],
        ),
        ActionRoundTripCase(
            LookAtAction, lambda world: [LookAtAction(target=_pose(world))]
        ),
        ActionRoundTripCase(
            FaceAtAction, lambda world: [FaceAtAction(target=_pose(world))]
        ),
        ActionRoundTripCase(
            FaceAndLookAtAction, lambda world: [_face_and_look_at(world)]
        ),
        ActionRoundTripCase(ElevatorNavigation, _elevator_navigation),
        ActionRoundTripCase(
            DetectAction,
            lambda world: [
                DetectAction(
                    technique=DetectionTechnique.TYPES, object_sem_annotation=Milk
                )
            ],
        ),
        ActionRoundTripCase(
            PerceptionTask,
            lambda world: [
                PerceptionTask(
                    query=PerceptionQuery(
                        Milk, _whole_scene(world), _robot(world), world
                    ),
                    answered_by=PerceptionSource.WORLD_MODEL,
                )
            ],
        ),
    ]


def _manipulation_cases() -> list[ActionRoundTripCase]:
    return [
        ActionRoundTripCase(
            ReachAction,
            lambda world: [
                ReachAction(grasp=_milk(world).grasp_candidates()[0], arm=_arm(world))
            ],
        ),
        ActionRoundTripCase(
            GraspingAction,
            lambda world: [
                GraspingAction(
                    grasp=_milk(world).grasp_candidates()[0], arm=_arm(world)
                )
            ],
        ),
        ActionRoundTripCase(PickUpAction, lambda world: [_pick_up(world)]),
        ActionRoundTripCase(
            PlaceAction, lambda world: [_pick_up(world), _place(world)]
        ),
        ActionRoundTripCase(
            OpenAction,
            lambda world: [
                OpenAction(grasp=_handle(world).grasp_candidates()[0], arm=_arm(world))
            ],
        ),
        ActionRoundTripCase(
            CloseAction,
            lambda world: [
                CloseAction(grasp=_handle(world).grasp_candidates()[0], arm=_arm(world))
            ],
        ),
        ActionRoundTripCase(
            MoveAndPickUpAction, lambda world: [_move_and_pick_up(world)]
        ),
        ActionRoundTripCase(
            MoveAndPlaceAction,
            lambda world: [_pick_up(world), _move_and_place(world)],
        ),
        ActionRoundTripCase(
            MoveAndOpenAction,
            lambda world: [
                MoveAndOpenAction(
                    navigate=_navigate(world),
                    face_and_look_at=_face_and_look_at(world),
                    open_container=OpenAction(
                        grasp=_handle(world).grasp_candidates()[0], arm=_arm(world)
                    ),
                )
            ],
        ),
        ActionRoundTripCase(
            PickAndPlaceAction,
            lambda world: [
                PickAndPlaceAction(pick_up=_pick_up(world), place=_place(world))
            ],
        ),
        ActionRoundTripCase(
            TransportAction,
            lambda world: [
                TransportAction(
                    pick_up=_move_and_pick_up(world), place=_move_and_place(world)
                )
            ],
        ),
    ]


def _tool_cases() -> list[ActionRoundTripCase]:
    return [
        ActionRoundTripCase(
            MixingAction,
            lambda world: [
                MixingAction(
                    container=_container(world),
                    arm=_arm(world),
                    tool=_tool(world, Whisk),
                )
            ],
        ),
        ActionRoundTripCase(
            CuttingAction,
            lambda world: [
                CuttingAction(
                    object_to_cut=_container(world),
                    arm=_arm(world),
                    tool=_tool(world, CuttingKnife),
                )
            ],
        ),
        ActionRoundTripCase(
            WipingAction,
            lambda world: [
                WipingAction(
                    arm=_arm(world),
                    tool=_tool(world, Sponge),
                    target_pose=_pose(world),
                )
            ],
        ),
        ActionRoundTripCase(
            PouringAction,
            lambda world: [
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
