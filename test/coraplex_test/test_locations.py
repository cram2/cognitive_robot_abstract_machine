from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from typing_extensions import Iterator, List

from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location, PoseGeneratorBackend, PoseValidator
from coraplex.view_manager import ViewManager
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% test doubles


@dataclass
class FixedPoseGenerator(PoseGeneratorBackend):
    """
    Yields predetermined candidates, so a location's placement can be asserted exactly.
    """

    poses: List[Pose]
    """
    The candidates to yield, in order.
    """

    def __iter__(self) -> Iterator[Pose]:
        return iter(self.poses)


@dataclass
class RecordsEvaluatedRobot(PoseValidator):
    """
    Accepts every candidate and records the robot it was evaluated against.
    """

    evaluated_robots: List[AbstractRobot] = field(default_factory=list)
    """
    The robot annotation each candidate was evaluated against, in evaluation order.
    """

    evaluated_root_poses: List[Pose] = field(default_factory=list)
    """
    Where the evaluated robot's root stood in the world frame, in evaluation order.
    """

    def __call__(self, *args, **kwargs) -> bool:
        self.evaluated_robots.append(self.robot)
        self.evaluated_root_poses.append(self.robot.root.global_pose)
        return True


@dataclass
class MotionlessExecutor:
    """
    Stands in for a Giskard executor and leaves the world exactly as it found it.
    """

    def tick_until_end(self, *args, **kwargs) -> None:
        pass


@dataclass
class UnsolvableMotionExecutor:
    """
    Stands in for a Giskard executor whose motion never reaches its goal.
    """

    outcome: Exception
    """
    The exception the motion ends with instead of reaching its goal.
    """

    def tick_until_end(self, *args, **kwargs) -> None:
        raise self.outcome


@dataclass
class RecordsTickBudget:
    """
    Stands in for a Giskard executor and records how many ticks it was allowed.
    """

    granted_budgets: List[int] = field(default_factory=list)
    """
    The tick budget each solve was started with, in solve order.
    """

    def tick_until_end(self, tick_budget: int) -> None:
        self.granted_budgets.append(tick_budget)


# %% specification-built worlds whose odom is displaced

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom offsets stay
# in that plane. The environment holds nothing but the robots, so a candidate is never
# rejected for collision and each test fails only for the behaviour it names.
_FIRST_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(0.5, 0.5, 0, yaw=np.pi / 2)
_SECOND_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(
    -2.0, 1.0, 0, yaw=-np.pi / 4
)


def _world_with_robots_behind_displaced_odoms(
    *world_T_odoms: HomogeneousTransformationMatrix,
) -> World:
    """
    A world holding nothing but PR2s, each reached through its own displaced odom.
    """
    specification = WorldSpecification(
        world_parser=None,
        robots=[
            RobotSpecification(semantic_annotation_type=PR2, world_T_odom=world_T_odom)
            for world_T_odom in world_T_odoms
        ],
    )
    try:
        return specification.to_domain_object()
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")


@pytest.fixture(scope="session")
def _single_robot_world_setup() -> World:
    return _world_with_robots_behind_displaced_odoms(_FIRST_ODOM)


@pytest.fixture
def single_robot_world(_single_robot_world_setup):
    world = deepcopy(_single_robot_world_setup)
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    return world, robot, Context(world, robot)


@pytest.fixture(scope="session")
def _two_robot_world_setup() -> World:
    return _world_with_robots_behind_displaced_odoms(_FIRST_ODOM, _SECOND_ODOM)


@pytest.fixture
def two_robot_world(_two_robot_world_setup):
    return deepcopy(_two_robot_world_setup)


def _candidate(world: World) -> Pose:
    return Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)


_OBSTACLE_SCALE = Scale(0.3, 0.3, 0.4)
"""
The extents of the obstacle the collision tests place in front of the robot.

As tall as the robot's base and no taller, so the arms above it are never what the
candidate is judged by.
"""

_CLEAR_OF_THE_BASE = 0.08
"""
A gap, in meters, that the robot's base is reported close to but not in collision with.

Wider than the margin the PR2's own rules keep around its base, narrower than the
distance at which the collision detector stops reporting the pair at all -- which is
what makes a candidate at this distance tell a proximity check apart from a collision
check.
"""


def _box_at(world: World, position: Point3, scale: Scale) -> Body:
    """
    Put a box into the world, standing on its own in the world frame.

    :param world: The world the box is added to.
    :param position: Where the box's center goes.
    :param scale: The extents of the box.
    :return: The box, which stands in either as an obstacle or as something to reach
        for.
    """
    coordinates = np.asarray(position.to_np()).flatten()[:3]
    box = Body(
        name=PrefixedName(f"box_{world.bodies.__len__()}", "coraplex_test"),
        collision=ShapeCollection([Box(scale=scale)]),
    )
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *coordinates
                ),
            )
        )
    return box


def _box_in_front_of(world: World, robot: AbstractRobot, gap: float) -> Body:
    """
    Put a box in front of the world origin, ``gap`` meters clear of the base a robot
    standing there would occupy.

    :param world: The world the box is added to.
    :param robot: The robot whose base decides how far out the box goes.
    :param gap: The clearance between the base and the box, negative to overlap.
    :return: The box, which stands in either as an obstacle or as something to reach
        for.
    """
    base_box = robot.mobile_base.bounding_box
    return _box_at(
        world,
        Point3.from_iterable(
            [
                base_box.depth / 2 + _OBSTACLE_SCALE.x / 2 + gap,
                0.0,
                _OBSTACLE_SCALE.z / 2,
            ]
        ),
        _OBSTACLE_SCALE,
    )


def _candidate_at_the_origin(world: World) -> Pose:
    """
    :return: A candidate that puts the robot's root on the world origin, facing the
        box :func:`_box_in_front_of` places.
    """
    return Pose.from_xyz_rpy(0.0, 0.0, 0.0, reference_frame=world.root)


# %% a location evaluates candidates where the world frame says they are


def test_location_places_the_robot_at_the_candidate_in_the_world_frame(
    single_robot_world,
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    recorder = RecordsEvaluatedRobot()

    list(Location(context, candidate, FixedPoseGenerator([candidate]), [recorder]))

    np.testing.assert_allclose(
        recorder.evaluated_root_poses[0].to_np(), candidate.to_np(), atol=1e-9
    )


def test_location_yields_the_pose_it_evaluated(single_robot_world):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    recorder = RecordsEvaluatedRobot()

    yielded_poses = list(
        Location(context, candidate, FixedPoseGenerator([candidate]), [recorder])
    )

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(
        yielded_poses[0].to_np(),
        recorder.evaluated_root_poses[0].to_np(),
        atol=1e-9,
    )


# %% a location evaluates the robot of its context


def test_location_evaluates_the_robot_of_its_context(two_robot_world):
    world = two_robot_world
    second_robot = world.get_semantic_annotations_by_type(PR2)[1]
    context = Context(world, second_robot)
    recorder = RecordsEvaluatedRobot()
    candidate = _candidate(world)

    list(Location(context, candidate, FixedPoseGenerator([candidate]), [recorder]))

    assert recorder.evaluated_robots[0].id == second_robot.id


# %% the giskard backend reports the pose it placed the robot at


def test_giskard_backend_yields_the_candidate_it_placed_the_robot_at(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)
    backend = GiskardLocationBackend(
        target=candidate,
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
        ),
        robot=robot,
        world=world,
    )
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: MotionlessExecutor(),
    )

    yielded_poses = list(backend)

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(yielded_poses[0].to_np(), candidate.to_np(), atol=1e-9)


# %% a location rejects what execution would cancel on, and nothing else


def test_location_keeps_a_candidate_that_only_stands_close_to_an_obstacle(
    single_robot_world,
):
    world, robot, context = single_robot_world
    _box_in_front_of(world, robot, gap=_CLEAR_OF_THE_BASE)
    candidate = _candidate_at_the_origin(world)

    yielded_poses = list(
        Location(context, candidate, FixedPoseGenerator([candidate]), [])
    )

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(yielded_poses[0].to_np(), candidate.to_np(), atol=1e-9)


def test_location_rejects_a_candidate_that_stands_inside_an_obstacle(
    single_robot_world,
):
    world, robot, context = single_robot_world
    _box_in_front_of(world, robot, gap=-_OBSTACLE_SCALE.x)
    candidate = _candidate_at_the_origin(world)

    yielded_poses = list(
        Location(context, candidate, FixedPoseGenerator([candidate]), [])
    )

    assert yielded_poses == []


# %% a location leaves the world it was built on alone


def _backend_reaching_for(
    target: Pose | Body, robot: AbstractRobot, world: World
) -> GiskardLocationBackend:
    """
    :return: A backend that reaches for ``target`` with the robot's right arm.
    """
    end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)
    return GiskardLocationBackend(
        target=target,
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
        ),
        robot=robot,
        world=world,
    )


def test_location_leaves_the_context_world_untouched(single_robot_world, monkeypatch):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = _backend_reaching_for(candidate, robot, world)
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: MotionlessExecutor(),
    )
    pose_before = robot.root.global_pose.to_np()

    list(Location(context, candidate, backend, []))

    np.testing.assert_allclose(robot.root.global_pose.to_np(), pose_before, atol=1e-9)


# %% the giskard backend only reports poses it solved for


@pytest.mark.parametrize(
    "outcome",
    [TimeoutError(), CollisionViolatedError(violated_collisions=[], thresholds=[])],
    ids=["timeout", "collision"],
)
def test_giskard_backend_skips_a_candidate_whose_motion_does_not_reach_the_target(
    single_robot_world, monkeypatch, outcome
):
    world, robot, context = single_robot_world
    unreachable, reachable = _candidate(world), _candidate_at_the_origin(world)
    backend = _backend_reaching_for(unreachable, robot, world)
    executors = iter([UnsolvableMotionExecutor(outcome), MotionlessExecutor()])
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_costmap",
        lambda self, pose: [unreachable, reachable],
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: next(executors),
    )

    yielded_poses = list(backend)

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(yielded_poses[0].to_np(), reachable.to_np(), atol=1e-9)


def test_giskard_backend_builds_an_executor_for_every_candidate(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidates = [_candidate(world), _candidate_at_the_origin(world)]
    backend = _backend_reaching_for(candidates[0], robot, world)
    built_executors = []
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: candidates
    )

    def build(self, *args, **kwargs):
        built_executors.append(MotionlessExecutor())
        return built_executors[-1]

    monkeypatch.setattr(GiskardLocationBackend, "setup_giskard_executor", build)

    list(backend)

    assert len(built_executors) == len(candidates)


def test_giskard_backend_solves_the_reach_the_grasp_performs(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    target = _box_in_front_of(world, robot, gap=0.5)
    backend = _backend_reaching_for(target, robot, world)
    solved_sequences = []
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )

    def record(self, pose_sequence, *args, **kwargs):
        solved_sequences.append(pose_sequence)
        return MotionlessExecutor()

    monkeypatch.setattr(GiskardLocationBackend, "setup_giskard_executor", record)

    list(backend)

    pre_pose, grasp_pose, _ = backend.grasp_description.grasp_pose_sequence(target)
    assert [pose.to_np().tolist() for pose in solved_sequences[0]] == [
        pre_pose.to_np().tolist(),
        grasp_pose.to_np().tolist(),
    ]


# %% the backend searches within the budget it was given


def test_giskard_backend_solves_a_candidate_within_the_tick_budget_it_was_given(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = replace(
        _backend_reaching_for(candidate, robot, world), candidate_tick_budget=7
    )
    executor = RecordsTickBudget()
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: executor,
    )

    list(backend)

    assert executor.granted_budgets == [backend.candidate_tick_budget]


def test_giskard_backend_draws_a_seed_per_candidate_it_may_try(single_robot_world):
    world, robot, context = single_robot_world
    target = _candidate(world)
    backend = replace(_backend_reaching_for(target, robot, world), candidate_seeds=3)

    costmap = backend.setup_costmap(target)

    assert costmap.number_of_samples == backend.candidate_seeds


# %% the backend solves the reach the action will perform, in the world it was given


def test_giskard_backend_moves_into_the_world_it_is_given(single_robot_world):
    world, robot, context = single_robot_world
    target = _box_in_front_of(world, robot, gap=0.5)
    backend = _backend_reaching_for(target, robot, world)
    other_world = deepcopy(world)

    moved = backend.copy_for_world(other_world)

    assert moved.world is other_world
    assert moved.robot is other_world.get_semantic_annotation_by_id(robot.id)
    assert moved.target is other_world.get_world_entity_with_id_by_id(target.id)
    assert moved.grasp_description.end_effector.tool_frame._world is other_world


def test_giskard_backend_keeps_its_search_budget_when_it_moves_world(
    single_robot_world,
):
    world, robot, context = single_robot_world
    target = _box_in_front_of(world, robot, gap=0.5)
    backend = replace(
        _backend_reaching_for(target, robot, world),
        candidate_seeds=3,
        candidate_tick_budget=11,
    )

    moved = backend.copy_for_world(deepcopy(world))

    assert moved.candidate_seeds == backend.candidate_seeds
    assert moved.candidate_tick_budget == backend.candidate_tick_budget


def test_giskard_backend_reaches_for_a_pose_with_what_the_gripper_holds(
    single_robot_world,
):
    """
    Placing puts the *held body* on the target, so the hand has to stop short of it by
    that body -- which is the reach
    :class:`~coraplex.robot_plans.actions.core.placing.PlaceAction` performs.
    """
    world, robot, context = single_robot_world
    end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)
    held = _box_at(
        world, end_effector.tool_frame.global_pose.to_position(), Scale(0.05, 0.05, 0.2)
    )
    with world.modify_world():
        world.remove_connection(held.parent_connection)
        world.add_connection(
            FixedConnection(parent=end_effector.tool_frame, child=held)
        )
    target = Pose.from_xyz_rpy(1.0, 0.5, 0.8, reference_frame=world.root)
    backend = _backend_reaching_for(target, robot, world)

    sequence = backend.reach_sequence(backend.grasp_description)

    transport, placing, _ = backend.grasp_description.place_pose_sequence(target)
    assert [pose.to_np().tolist() for pose in sequence] == [
        transport.to_np().tolist(),
        placing.to_np().tolist(),
    ]


def test_giskard_backend_turns_a_point_target_the_way_the_robot_stands(
    single_robot_world,
):
    """
    A point says where something has to end up but not which way it ends up turned, so
    the reach takes the robot's own heading -- which is the heading it would put the
    object down at from where it stands.
    """
    world, robot, context = single_robot_world
    standing_yaw = 0.7
    with world.modify_world():
        robot.set_root_pose(
            Pose.from_xyz_rpy(
                0.3, -0.2, 0.0, yaw=standing_yaw, reference_frame=world.root
            )
        )
    point = Point3(1.0, 0.5, 0.8, reference_frame=world.root)
    backend = _backend_reaching_for(point, robot, world)

    reached_for = backend.target_facing_the_robot()

    assert isinstance(reached_for, Pose)
    assert [float(reached_for.x), float(reached_for.y), float(reached_for.z)] == [
        pytest.approx(float(point.x)),
        pytest.approx(float(point.y)),
        pytest.approx(float(point.z)),
    ]
    assert float(reached_for.yaw) == pytest.approx(float(robot.root.global_pose.yaw))


def test_giskard_backend_follows_the_robot_when_it_stands_somewhere_else(
    single_robot_world,
):
    """
    The heading is read off the robot every time it is asked for, so a point target
    moves with it rather than keeping the heading of wherever it was first asked.
    """
    world, robot, context = single_robot_world
    backend = _backend_reaching_for(
        Point3(1.0, 0.5, 0.8, reference_frame=world.root), robot, world
    )

    headings = []
    for standing_yaw in (0.0, 1.2):
        with world.modify_world():
            robot.set_root_pose(
                Pose.from_xyz_rpy(
                    0.0, 0.0, 0.0, yaw=standing_yaw, reference_frame=world.root
                )
            )
        headings.append(float(backend.target_facing_the_robot().yaw))

    assert headings == [pytest.approx(0.0), pytest.approx(1.2)]


def test_giskard_backend_offers_every_side_when_it_is_given_no_grasp(
    single_robot_world,
):
    """
    Which side the gripper comes from depends on where the robot ends up standing, so a
    backend given no grasp keeps all of them open rather than settling one up front.
    """
    world, robot, context = single_robot_world
    backend = GiskardLocationBackend(
        target=Pose.from_xyz_rpy(1.0, 0.5, 0.8, reference_frame=world.root),
        arm=Arms.RIGHT,
        grasp_description=None,
        robot=robot,
        world=world,
    )

    grasps = backend.grasps_to_try()

    assert [grasp.approach_direction for grasp in grasps] == list(ApproachDirection)
    assert {grasp.vertical_alignment for grasp in grasps} == {
        VerticalAlignment.NoAlignment
    }
    assert {grasp.end_effector for grasp in grasps} == {
        ViewManager.get_end_effector_view(Arms.RIGHT, robot)
    }


def test_giskard_backend_keeps_to_the_grasp_it_was_given(single_robot_world):
    """
    A caller that knows which side it comes from is not second-guessed.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(1.0, 0.5, 0.8, reference_frame=world.root)
    backend = _backend_reaching_for(target, robot, world)

    assert backend.grasps_to_try() == [backend.grasp_description]


def test_giskard_backend_leaves_a_pose_target_alone(single_robot_world):
    """
    A pose already says which way it faces, so the robot's heading does not touch it.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(1.0, 0.5, 0.8, yaw=2.0, reference_frame=world.root)

    assert (
        _backend_reaching_for(target, robot, world).target_facing_the_robot() is target
    )


def test_giskard_backend_reaches_for_a_pose_directly_with_an_empty_hand(
    single_robot_world,
):
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(1.0, 0.5, 0.8, reference_frame=world.root)
    backend = _backend_reaching_for(target, robot, world)

    assert backend.reach_sequence(backend.grasp_description) == [target]


def test_giskard_backend_demands_the_accuracy_the_motions_demand(single_robot_world):
    """
    A pose the backend calls reachable at a looser tolerance than the motion insists on
    is a pose the motion can stall at.
    """
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = _backend_reaching_for(candidate, robot, world)
    end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)

    executor = backend.setup_giskard_executor([candidate], world, robot, end_effector)

    tolerances = MotionToleranceConfig()
    goals = [
        node
        for node in executor.motion_statechart.nodes
        if isinstance(node, CartesianPose)
    ]
    assert goals
    assert all(
        goal.translation_threshold == tolerances.default_tcp_position_threshold
        and goal.orientation_threshold == tolerances.tool_orientation_threshold
        for goal in goals
    )


def test_giskard_backend_judges_a_reach_the_way_execution_runs_it(single_robot_world):
    """
    The reach the backend tries out has to be judged under the rules the drive there
    obeys: it keeps to its buffer zones like any other reach, and it ends on a violated
    collision the way execution does. Weighted above the avoidance instead, it would
    drive into what it should keep clear of and then reject the candidate over the
    contact it made itself.
    """
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = _backend_reaching_for(candidate, robot, world)
    end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)

    executor = backend.setup_giskard_executor([candidate], world, robot, end_effector)

    goals = [
        node
        for node in executor.motion_statechart.nodes
        if isinstance(node, CartesianPose)
    ]
    assert goals
    assert all(
        goal.weight == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE for goal in goals
    )
    avoidance = [
        node
        for node in executor.motion_statechart.nodes
        if isinstance(node, ExternalCollisionAvoidance)
    ]
    assert len(avoidance) == 1
    assert avoidance[0].cancel_if_collision_violated


# %% a search that finds nothing draws again


def test_giskard_backend_gives_up_when_no_candidate_works(
    single_robot_world, monkeypatch
):
    """
    A target that cannot be reached from anywhere has to be reported as such, once each
    candidate has been given its turn: searching for ever would hang a plan that should
    fail.
    """
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = _backend_reaching_for(candidate, robot, world)
    attempted_reaches = []
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )

    def build(self, *args, **kwargs):
        attempted_reaches.append(candidate)
        return UnsolvableMotionExecutor(TimeoutError())

    monkeypatch.setattr(GiskardLocationBackend, "setup_giskard_executor", build)

    assert list(backend) == []
    assert attempted_reaches == [candidate]


def test_giskard_backend_does_not_reach_from_a_pose_it_cannot_stand_at(
    single_robot_world, monkeypatch
):
    """
    Simulating a reach costs hundreds of control ticks, and no arm movement makes a pose
    the robot does not even fit at into one it can work from.
    """
    world, robot, context = single_robot_world
    _box_in_front_of(world, robot, gap=-_OBSTACLE_SCALE.x)
    candidate = _candidate_at_the_origin(world)
    backend = _backend_reaching_for(candidate, robot, world)
    attempted_reaches = []
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )

    def build(self, *args, **kwargs):
        attempted_reaches.append(candidate)
        return MotionlessExecutor()

    monkeypatch.setattr(GiskardLocationBackend, "setup_giskard_executor", build)

    assert list(backend) == []
    assert attempted_reaches == []


def test_giskard_backend_skips_a_candidate_that_cannot_see_the_target(
    single_robot_world, monkeypatch
):
    """
    A body has to be seen to be worked with, and looking costs a ray where trying costs
    hundreds of control ticks, so a candidate the target is hidden from is dropped
    before a reach is simulated from it.
    """
    world, robot, context = single_robot_world
    target = _box_at(
        world, Point3.from_iterable([1.0, 0.5, 0.8]), Scale(0.05, 0.05, 0.2)
    )
    candidate = _candidate_at_the_origin(world)
    backend = _backend_reaching_for(target, robot, world)
    attempted_reaches = []
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: [candidate]
    )
    monkeypatch.setattr(PR2, "can_see_body", lambda self, body: False)

    def build(self, *args, **kwargs):
        attempted_reaches.append(candidate)
        return MotionlessExecutor()

    monkeypatch.setattr(GiskardLocationBackend, "setup_giskard_executor", build)

    assert list(backend) == []
    assert attempted_reaches == []
