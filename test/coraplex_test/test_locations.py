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
from coraplex.locations.costmaps import Costmap, RingCostmap
from coraplex.locations.sampling import CandidateDraw
from coraplex.config.action_conf import ActionConfig
from coraplex.locations import factories
from coraplex.locations.factories import accessing_location, reachability_location
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.view_manager import ViewManager
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowSelfCollisions,
    CollisionRule,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
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

    def candidates(self, draw: CandidateDraw) -> Iterator[Pose]:
        """
        Offers every candidate it was given, in its own order, whatever terms it is
        drawn on: it rates none of them, and a test says exactly which it expects.
        """
        return iter(self.poses)


@dataclass
class OffersFixedCandidates(Costmap):
    """
    Stands in for a map that has already decided what it offers, so a backend's own
    handling of the candidates can be asserted without building a map to produce them.
    """

    offered: List[Pose] = field(default_factory=list)
    """
    The candidates to offer, in order.
    """

    def candidates(self, draw: CandidateDraw) -> Iterator[Pose]:
        return iter(self.offered)


def _offering(poses: List[Pose]) -> OffersFixedCandidates:
    """
    :return: A map that offers exactly ``poses``.
    """
    return OffersFixedCandidates(
        resolution=0.02, world=poses[0].reference_frame._world, offered=poses
    )


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
class RefusesEveryCandidate(PoseValidator):
    """
    Refuses every candidate and counts how often it was asked, so the number of
    candidates a location spends on validation can be asserted.
    """

    times_asked: int = 0
    """
    How often this validator was called.
    """

    def __call__(self, *args, **kwargs) -> bool:
        self.times_asked += 1
        return False


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


DRIVEN_DISTANCE = 0.4
"""
How far the executor standing in for a drive moves the robot, in meter.
"""


@dataclass
class DrivesOnce:
    """
    Stands in for a Giskard executor that can only run its motion once.

    A compiled statechart ends and cleans up when it is ticked to the end, so ticking
    the same one again returns without moving anything.
    """

    robot: AbstractRobot
    """
    The robot the drive moves.
    """

    driven: bool = False
    """
    Whether this executor has already run its motion.
    """

    def tick_until_end(self, *args, **kwargs) -> None:
        if self.driven:
            return
        self.driven = True
        reached = self.robot.root.global_pose
        self.robot.set_root_pose(
            Pose.from_xyz_rpy(
                float(reached.to_position().x) + DRIVEN_DISTANCE,
                float(reached.to_position().y),
                0.0,
                reference_frame=reached.reference_frame,
            )
        )


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

    list(Location(context, candidate, FixedPoseGenerator([candidate]), recorder))

    np.testing.assert_allclose(
        recorder.evaluated_root_poses[0].to_np(), candidate.to_np(), atol=1e-9
    )


def test_location_yields_the_pose_it_evaluated(single_robot_world):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    recorder = RecordsEvaluatedRobot()

    yielded_poses = list(
        Location(context, candidate, FixedPoseGenerator([candidate]), recorder)
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

    list(Location(context, candidate, FixedPoseGenerator([candidate]), recorder))

    assert recorder.evaluated_robots[0].id == second_robot.id


# %% how far a reachability location stands from its target


REACHABILITY_TARGET_POSITION = (2.0, 2.0, 0.9)
"""
Position of the target a reachability location is built around, clear of the robot.
"""

REACH_FRACTION = 0.5
"""
The fraction of the arm's length the sampled ring is asked to stand off by, chosen away
from the default so the parameter is what the sampling follows.
"""


def test_a_ring_from_the_arm_reach_distance_stands_off_by_the_reach_fraction(
    single_robot_world,
):
    """
    The standing distance follows the reach fraction, so tuning it moves the robot.

    Standing too close puts the arms inside whatever the target rests on, which the
    collision check on candidate poses then rejects.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    # approximate_length returns a symbolic scalar, and so does the distance derived
    # from it, which compares as unequal to a float under pytest.approx no matter the
    # tolerance.
    arm = ViewManager.get_arm_view(Arms.RIGHT, robot)
    expected_distance = float(arm.approximate_length()) * REACH_FRACTION

    ring = RingCostmap.from_arm_reach_distance(
        context, arm, target, reach_fraction=REACH_FRACTION
    )

    assert float(ring.distance) == pytest.approx(expected_distance)


# %% how many candidates a location spends on validation


CANDIDATES_IN_COLLISION = 4
"""
How many candidates of the budget test stand inside the box, so are thrown out before
any validator sees them.
"""

CANDIDATES_BEYOND_THE_BUDGET = 5
"""
How many candidates past the budget are offered, so that running out of budget is what
ends the search rather than running out of candidates.
"""


def _pose_at(world: World, x: float, y: float) -> Pose:
    """
    :param world: The world the pose is expressed in.
    :param x: Where the robot stands along the world's x-axis.
    :param y: Where it stands along the y-axis.
    :return: A standing pose there.
    """
    return Pose.from_xyz_rpy(x, y, 0.0, reference_frame=world.root)


def _poses_clear_of_everything(world: World, count: int) -> List[Pose]:
    """
    :param world: The world the poses are expressed in.
    :param count: How many to lay out.
    :return: Standing poses spaced along a line, none of them touching anything.
    """
    return [_pose_at(world, 2.0 + 0.5 * step, 2.0) for step in range(count)]


def test_a_location_validates_no_more_candidates_than_its_budget(single_robot_world):
    """
    Validation drives the robot to see whether it arrives, which is the expensive part
    of judging a standing pose, so a location only spends its budget of them before it
    gives up.
    """
    world, robot, context = single_robot_world
    validator = RefusesEveryCandidate(context=context)
    budget = Location.candidates_to_validate
    location = Location(
        context,
        _pose_at(world, 0.0, 0.0),
        FixedPoseGenerator(
            _poses_clear_of_everything(world, budget + CANDIDATES_BEYOND_THE_BUDGET)
        ),
        validator,
    )

    assert list(location) == []
    assert validator.times_asked == location.candidates_to_validate


def test_a_location_does_not_spend_its_budget_on_candidates_it_never_validates(
    single_robot_world,
):
    """
    A candidate thrown out for standing in collision costs nothing to judge, so it must
    not use up one of the validations the location is allowed.

    Otherwise a target hemmed in by furniture exhausts the budget before a single pose
    is ever tried.
    """
    world, robot, context = single_robot_world
    box = _box_in(world)
    box_position = box.global_pose.to_position().to_np()[:2].ravel()
    validator = RefusesEveryCandidate(context=context)
    budget = Location.candidates_to_validate
    inside_the_box = [
        _pose_at(world, float(box_position[0]), float(box_position[1]))
    ] * CANDIDATES_IN_COLLISION
    location = Location(
        context,
        _pose_at(world, 0.0, 0.0),
        FixedPoseGenerator(inside_the_box + _poses_clear_of_everything(world, budget)),
        validator,
    )

    assert list(location) == []
    assert validator.times_asked == location.candidates_to_validate


# %% a reachability location for a body that is going to be somewhere else


def _box_in(world: World) -> Body:
    """
    A box with collision geometry, standing away from the robot.
    """
    body = Body(
        name=PrefixedName("box"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.2))]),
    )
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.0, y=-1.0, z=0.9
                ),
            )
        )
    return body


def test_a_reachability_location_for_a_body_stands_around_its_destination(
    single_robot_world,
):
    world, robot, context = single_robot_world
    body = _box_in(world)
    destination = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )

    location = reachability_location(
        body,
        context,
        ViewManager.get_arm_view(Arms.RIGHT, robot),
        destination=destination,
    )

    assert location.target_pose is destination


def test_a_reachability_location_for_a_body_reaches_the_grasp_at_its_destination(
    single_robot_world,
):
    """
    The grasp is carried to the destination with the body, and the approach still clears
    the body itself.

    A body reached at a destination is released there, and a release runs the sequence
    backwards, so the check has to run it backwards too.
    """
    world, robot, context = single_robot_world
    body = _box_in(world)
    destination = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    grasp = Pose.from_xyz_rpy(z=0.05, reference_frame=body)

    arm = ViewManager.get_arm_view(Arms.RIGHT, robot)

    validator = reachability_location(
        body, context, arm, grasp_pose=grasp, destination=destination
    ).validator

    expected_sequence = HasApproachesGraspPoses().grasp_pose_sequence(
        destination.to_homogeneous_matrix() @ grasp,
        arm.end_effector,
        grasp,
        reverse=True,
    )
    np.testing.assert_allclose(
        [pose.to_np() for pose in validator.pose_sequence],
        [pose.to_np() for pose in expected_sequence],
        atol=1e-9,
    )


def test_a_reachability_location_for_a_body_where_it_is_reaches_the_grasp_onto_it(
    single_robot_world,
):
    """
    Without a destination the body is picked up where it stands, which approaches the
    grasp rather than withdrawing from it.
    """
    world, robot, context = single_robot_world
    body = _box_in(world)
    grasp = Pose.from_xyz_rpy(z=0.05, reference_frame=body)

    arm = ViewManager.get_arm_view(Arms.RIGHT, robot)

    validator = reachability_location(body, context, arm, grasp_pose=grasp).validator

    expected_sequence = HasApproachesGraspPoses().grasp_pose_sequence(
        body.global_pose.to_homogeneous_matrix() @ grasp,
        arm.end_effector,
        grasp,
    )
    np.testing.assert_allclose(
        [pose.to_np() for pose in validator.pose_sequence],
        [pose.to_np() for pose in expected_sequence],
        atol=1e-9,
    )


# %% opening a container is reached for by its own standing distance


def test_an_accessing_location_stands_off_by_the_accessing_reach_fraction(
    single_robot_world, monkeypatch
):
    """
    Opening a container is reached for by its own standing distance rather than the one
    used for a grasp.
    """
    world, robot, context = single_robot_world
    handle_body = _box_in(world)
    container = Drawer(root=handle_body, handle=Handle(root=handle_body))
    asked_for = {}
    monkeypatch.setattr(
        factories,
        "reachability_location",
        lambda *args, **kwargs: asked_for.update(kwargs),
    )

    accessing_location(container, context, ViewManager.get_arm_view(Arms.RIGHT, robot))

    assert asked_for["reach_fraction"] == ActionConfig.accessing_reach_fraction


# %% the giskard backend reports the pose it placed the robot at


def test_giskard_backend_yields_the_candidate_it_placed_the_robot_at(
    single_robot_world, monkeypatch
):
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = GiskardLocationBackend(
        target_pose=candidate,
        arm=ViewManager.get_arm_view(Arms.RIGHT, robot),
        grasp_pose=candidate,
        robot=robot,
        world=world,
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_costmap",
        lambda self, pose: _offering([candidate]),
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: MotionlessExecutor(),
    )

    yielded_poses = list(backend.candidates(CandidateDraw()))

    assert len(yielded_poses) == 1
    np.testing.assert_allclose(yielded_poses[0].to_np(), candidate.to_np(), atol=1e-9)


def test_giskard_backend_solves_the_reach_its_location_validates(
    single_robot_world, monkeypatch
):
    """
    The backend steers the robot onto the same approach the location's validator then
    checks, including how far that approach stays off the grasped body.
    """
    world, robot, context = single_robot_world
    body = _box_in(world)
    grasp = Pose.from_xyz_rpy(z=0.05, reference_frame=body)
    grasp_frame = body.global_pose.to_homogeneous_matrix() @ grasp
    backend = GiskardLocationBackend(
        target_pose=body.global_pose,
        arm=ViewManager.get_arm_view(Arms.RIGHT, robot),
        grasp_pose=grasp_frame,
        robot=robot,
        world=world,
        body_T_grasp=grasp,
    )
    solved_sequences = []

    def record_the_solved_sequence(self, pose_sequence, *args, **kwargs):
        solved_sequences.append(pose_sequence)
        return MotionlessExecutor()

    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_costmap",
        lambda self, pose: _offering([_candidate(world)]),
    )
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_giskard_executor", record_the_solved_sequence
    )

    list(backend.candidates(CandidateDraw()))

    expected_sequence = HasApproachesGraspPoses().grasp_pose_sequence(
        grasp_frame, ViewManager.get_end_effector_view(Arms.RIGHT, robot), grasp
    )
    np.testing.assert_allclose(
        [pose.to_np() for pose in solved_sequences[0]],
        [pose.to_np() for pose in expected_sequence],
        atol=1e-9,
    )


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

    list(Location(context, candidate, FixedPoseGenerator([candidate]), recorder))

    assert recorder.temporary_rules_seen
    assert not any(
        isinstance(rule, AllowSelfCollisions)
        for rule in recorder.temporary_rules_seen[0]
    )


def test_location_validates_with_the_motion_policy_of_its_own_context(
    single_robot_world,
):
    """
    Validators run against a copy of the world and so are handed a context of their own.

    That context has to carry the tolerances of the run, or a candidate is judged by
    defaults the plan itself is never held to.
    """
    world, robot, context = single_robot_world
    context.motion_tolerances.default_tcp_position_threshold = 0.123
    candidate = _candidate(world)
    recorder = RecordsEvaluatedRobot()

    list(Location(context, candidate, FixedPoseGenerator([candidate]), recorder))

    assert recorder.context.motion_tolerances is context.motion_tolerances


def test_the_giskard_backend_drives_to_every_candidate(single_robot_world, monkeypatch):
    """
    Each candidate is a different place to stand, so each has to be driven from.

    An executor that has already run its motion returns without moving anything, which
    would leave every candidate after the first wherever it was dropped.
    """
    world, robot, context = single_robot_world
    candidates = [
        Pose.from_xyz_rpy(1.3, 2.0 + offset, 0.0, reference_frame=world.root)
        for offset in (0.0, 0.5, 1.0)
    ]
    backend = GiskardLocationBackend(
        target_pose=candidates[0],
        arm=ViewManager.get_arm_view(Arms.RIGHT, robot),
        grasp_pose=candidates[0],
        robot=robot,
        world=world,
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_costmap",
        lambda self, pose: _offering(candidates),
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, pose_sequence, world, robot, end_effector: DrivesOnce(robot=robot),
    )

    reached = list(backend.candidates(CandidateDraw(number_of_samples=len(candidates))))

    assert [float(pose.to_position().x) for pose in reached] == [
        pytest.approx(float(candidate.to_position().x) + DRIVEN_DISTANCE)
        for candidate in candidates
    ]


# %% a location decides how its candidates are drawn


@dataclass
class RecordsHowItWasDrawn(PoseGeneratorBackend):
    """
    Yields one candidate and records the terms the draw was asked for on.
    """

    pose: Pose
    """
    The single candidate to yield.
    """

    asked_for: List[CandidateDraw] = field(default_factory=list)
    """
    One entry per draw: the terms it was asked on.
    """

    def candidates(self, draw: CandidateDraw) -> Iterator[Pose]:
        self.asked_for.append(draw)
        return iter([self.pose])


def test_a_location_draws_on_the_terms_it_was_given(single_robot_world):
    """
    How candidates are drawn belongs to the location rather than to any of the maps that
    constrain it, so what it was given is what reaches the draw.
    """
    world, robot, context = single_robot_world
    generator = RecordsHowItWasDrawn(pose=_candidate(world))
    draw = CandidateDraw(number_of_samples=17, seed=3)
    location = Location(context, _candidate(world), generator, None, draw=draw)

    list(islice(iter(location), 1))

    assert generator.asked_for == [draw]


def test_a_backend_that_does_not_rate_its_candidates_offers_its_own_order(
    single_robot_world,
):
    """
    Only a backend that rates its candidates has anything to draw weighted, so one that
    does not offers them as it always would.
    """
    world, robot, context = single_robot_world
    first, second = _candidate(world), _candidate(world)
    generator = FixedPoseGenerator([first, second])

    drawn = list(generator.candidates(CandidateDraw(number_of_samples=1, seed=1)))

    assert drawn == [first, second]


# %% a standing pose is drawn seeded by the plan it belongs to


def test_a_giskard_backend_drives_to_no_more_candidates_than_it_says(
    single_robot_world, monkeypatch
):
    """
    Each candidate costs a full simulated run, so the backend draws far fewer of them
    than a map is usually offered for.
    """
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    backend = GiskardLocationBackend(
        target_pose=candidate,
        arm=ViewManager.get_arm_view(Arms.RIGHT, robot),
        grasp_pose=candidate,
        robot=robot,
        world=world,
        number_of_candidates=2,
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_costmap",
        lambda self, pose: _offering([candidate] * 9),
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: MotionlessExecutor(),
    )

    assert len(list(backend.candidates(CandidateDraw()))) == 2


def test_a_reachability_location_takes_its_seed_from_the_context(single_robot_world):
    """
    A demonstration is only worth running as a regression test if it runs the same way
    twice, so a plan can fix the draws made anywhere inside it.
    """
    world, robot, context = single_robot_world
    context.sampling_seed = 5

    location = reachability_location(
        _box_in(world), context, ViewManager.get_arm_view(Arms.RIGHT, robot)
    )

    assert location.draw.seed == 5


def test_a_reachability_location_draws_afresh_without_one(single_robot_world):
    """
    Left unseeded a plan explores the region differently each run, which is what makes
    drawing from the map worth more than ranking it.
    """
    world, robot, context = single_robot_world

    location = reachability_location(
        _box_in(world), context, ViewManager.get_arm_view(Arms.RIGHT, robot)
    )

    assert location.draw.seed is None


# %% what a backend must say about the terms it is drawn on


def test_a_backend_that_does_not_say_how_it_draws_cannot_be_built():
    """
    A backend inherits no draw of its own, so one that leaves the terms unanswered is
    refused where it is defined rather than silently dropping them at runtime.
    """

    @dataclass
    class SaysNothingAboutTheTerms(PoseGeneratorBackend):
        pass

    with pytest.raises(TypeError):
        SaysNothingAboutTheTerms()


def test_a_giskard_backend_draws_its_map_on_the_seed_it_was_given(
    single_robot_world, monkeypatch
):
    """
    The backend picks its base poses from a map of its own, so the seed it is drawn on
    has to reach that map rather than being replaced by a draw of the backend's
    choosing.
    """
    world, robot, context = single_robot_world
    candidate = _candidate(world)
    drawn_map = RecordsHowItWasDrawn(pose=candidate)
    backend = GiskardLocationBackend(
        target_pose=candidate,
        arm=ViewManager.get_arm_view(Arms.RIGHT, robot),
        grasp_pose=candidate,
        robot=robot,
        world=world,
    )
    monkeypatch.setattr(
        GiskardLocationBackend, "setup_costmap", lambda self, pose: drawn_map
    )
    monkeypatch.setattr(
        GiskardLocationBackend,
        "setup_giskard_executor",
        lambda self, *args, **kwargs: MotionlessExecutor(),
    )

    list(backend.candidates(CandidateDraw(seed=4)))

    assert [terms.seed for terms in drawn_map.asked_for] == [4]
