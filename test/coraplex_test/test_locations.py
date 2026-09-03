from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice

import numpy as np
import pytest
from typing_extensions import Iterator, List

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.locations.backends import GiskardLocationBackend
from coraplex.locations.base import Location, PoseGeneratorBackend, PoseValidator
from coraplex.locations.factories import (
    reachability_location,
)
from coraplex.view_manager import ViewManager
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowSelfCollisions,
    CollisionRule,
)
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

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
class RecordsCollisionRules(PoseValidator):
    """
    Accepts every candidate and records the temporary collision rules in force while it
    was evaluated.
    """

    temporary_rules_seen: List[List[CollisionRule]] = field(default_factory=list)
    """
    The world's temporary collision rules at each evaluation, in evaluation order.
    """

    def __call__(self, *args, **kwargs) -> bool:
        self.temporary_rules_seen.append(
            list(self.world.collision_manager.temporary_rules)
        )
        return True


@dataclass
class MotionlessExecutor:
    """
    Stands in for a Giskard executor and leaves the world exactly as it found it.
    """

    def tick_until_end(self, *args, **kwargs) -> None:
        pass


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


# %% how far a reachability location stands from its target


REACHABILITY_TARGET_POSITION = (2.0, 2.0, 0.9)
"""
Position of the target a reachability location is built around, clear of the robot.
"""

STANDING_DISTANCE_TOLERANCE = 0.05
"""
Tolerance of a sampled standing distance, in meter.

Candidates land on the cell centres of a 0.02 m costmap grid, so a sample sits a
fraction of a cell off the ring it was drawn from.
"""

CANDIDATES_TO_SAMPLE = 20
"""
Number of candidates whose distance to the target is asserted.
"""


def test_reachability_location_stands_at_the_arm_length_fraction_from_its_target(
    single_robot_world,
):
    """
    The standing distance follows the constant, so tuning it moves the robot.

    Standing too close puts the arms inside whatever the target rests on, which the
    collision check on candidate poses then rejects.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    # approximate_length returns a symbolic scalar, which compares as unequal to a float
    # under pytest.approx no matter the tolerance.
    expected_distance = (
        float(ViewManager.get_arm_view(Arms.RIGHT, robot).approximate_length()) * 0.66
    )
    target_position = target.to_position().to_np()[:2]

    candidates = list(
        islice(
            reachability_location(target, context, Arms.RIGHT).generator,
            CANDIDATES_TO_SAMPLE,
        )
    )

    assert len(candidates) == CANDIDATES_TO_SAMPLE
    distances = [
        np.linalg.norm(candidate.to_position().to_np()[:2] - target_position)
        for candidate in candidates
    ]
    assert distances == pytest.approx(
        [expected_distance] * len(distances), abs=STANDING_DISTANCE_TOLERANCE
    )


# %% the giskard backend reports the pose it placed the robot at


def test_giskard_backend_yields_the_candidate_it_placed_the_robot_at(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = GiskardLocationBackend(
        target=candidate,
        arm=Arms.RIGHT,
        grasp_pose=candidate,
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


def test_location_validates_against_the_rules_the_plan_runs_with(single_robot_world):
    """
    Deciding whether a standing pose is already in collision needs collision rules of
    its own, but they are the wrong ones for the reachability simulation that follows:

    left in place they override the distances the robot actually has to keep, and a pose
    validates against clearances the executed motion is never given.
    """
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    recorder = RecordsCollisionRules()

    list(Location(context, candidate, FixedPoseGenerator([candidate]), [recorder]))

    assert recorder.temporary_rules_seen
    assert not any(
        isinstance(rule, AllowSelfCollisions)
        for rule in recorder.temporary_rules_seen[0]
    )
