from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice

import numpy as np
import pytest
from typing_extensions import Iterator, List

from coraplex.datastructures.dataclasses import Context
from coraplex.locations.base import Location
from coraplex.locations.costmaps import RingCostmap
from coraplex.locations.sampling import CandidateDraw
from coraplex.locations.locations import ReachabilityLocation, VisibilityLocation
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% test doubles


@dataclass
class RecordsHowItWasDrawn(Location):
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


# %% a specification-built world whose odom is displaced

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom offset stays
# in that plane.
_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(0.5, 0.5, 0, yaw=np.pi / 2)


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
    return _world_with_robots_behind_displaced_odoms(_ODOM)


@pytest.fixture
def single_robot_world(_single_robot_world_setup):
    world = deepcopy(_single_robot_world_setup)
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    return world, robot, Context(world, robot)


def _candidate(world: World) -> Pose:
    return Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)


# %% a location draws its candidates on its own terms


def test_a_location_draws_on_the_terms_it_was_given(single_robot_world):
    world, robot, context = single_robot_world
    draw = CandidateDraw(number_of_samples=17, seed=3)
    location = RecordsHowItWasDrawn(pose=_candidate(world), draw=draw)

    list(islice(iter(location), 1))

    assert location.asked_for == [draw]


def test_a_location_draws_nothing_before_it_is_consumed(single_robot_world):
    """
    A location handed to a plan as a domain is only drawn from once the plan asks for a
    pose, so it reflects the world at that moment.
    """
    world, robot, context = single_robot_world
    location = RecordsHowItWasDrawn(pose=_candidate(world))

    candidates = iter(location)
    assert location.asked_for == []

    next(candidates)
    assert location.asked_for == [location.draw]


def test_a_location_grounds_to_its_first_candidate(single_robot_world):
    world, robot, context = single_robot_world
    location = RecordsHowItWasDrawn(pose=_candidate(world))

    assert location.ground() is location.pose


def test_a_location_that_does_not_say_how_it_draws_cannot_be_built():
    """
    A location inherits no draw of its own, so one that leaves the terms unanswered is
    refused where it is defined rather than silently offering nothing at runtime.
    """

    @dataclass
    class SaysNothingAboutTheTerms(Location):
        pass

    with pytest.raises(TypeError):
        SaysNothingAboutTheTerms()


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
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    # approximate_length returns a symbolic scalar, and so does the distance derived
    # from it, which compares as unequal to a float under pytest.approx no matter the
    # tolerance.
    arm = context.robot.right_arm
    expected_distance = float(arm.approximate_length()) * REACH_FRACTION

    ring = RingCostmap.from_arm_reach_distance(
        context, arm, target, reach_fraction=REACH_FRACTION
    )

    assert float(ring.distance) == pytest.approx(expected_distance)


# %% a reachability location stands around the target it is given


def _box_in(world: World) -> Milk:
    """
    A graspable box with collision geometry, standing away from the robot.
    """
    body = Body(
        name=PrefixedName("box"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.2))]),
    )
    graspable = Milk(root=body)
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
        world.add_semantic_annotation(graspable)
    return graspable


def test_a_reachability_location_is_drawn_around_its_target(single_robot_world):
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )

    location = ReachabilityLocation(
        target, context.robot.right_arm, context=context
    )

    np.testing.assert_allclose(
        location.costmap().origin.to_position().to_np()[:2],
        target.to_position().to_np()[:2],
    )


def test_a_reachability_location_takes_its_seed_from_the_context(single_robot_world):
    """
    A demonstration is only worth running as a regression test if it runs the same way
    twice, so a plan can fix the draws made anywhere inside it.
    """
    world, robot, context = single_robot_world
    context.sampling_seed = 5

    location = ReachabilityLocation(
        _box_in(world).root.global_pose,
        context.robot.right_arm,
        context=context,
    )

    assert location.draw.seed == context.sampling_seed


def test_a_reachability_location_draws_afresh_without_one(single_robot_world):
    """
    Left unseeded a plan explores the region differently each run, which is what makes
    drawing from the map worth more than ranking it.
    """
    world, robot, context = single_robot_world

    location = ReachabilityLocation(
        _box_in(world).root.global_pose,
        context.robot.right_arm,
        context=context,
    )

    assert location.draw.seed is None


# %% a location reflects the world when it is drawn from


def test_a_costmap_location_builds_its_costmap_only_when_drawn_from(
    single_robot_world, monkeypatch
):
    """
    Handing a location to a plan must not build its costmap, so the map describes the
    world as the plan finds it when it gets there.
    """
    world, robot, context = single_robot_world
    location = ReachabilityLocation(
        _box_in(world).root.global_pose,
        context.robot.right_arm,
        context=context,
    )
    built = []
    build_costmap = ReachabilityLocation.costmap
    monkeypatch.setattr(
        ReachabilityLocation,
        "costmap",
        lambda self: built.append(True) or build_costmap(self),
    )

    candidates = iter(location)
    assert built == []

    next(candidates)
    assert built == [True]


def test_a_target_given_in_a_body_frame_follows_the_body(single_robot_world):
    """
    A target named relative to a body is where that body is when the location is drawn
    from, not where it was when the location was made.
    """
    world, robot, context = single_robot_world
    box = _box_in(world).root
    location = ReachabilityLocation(
        Pose(reference_frame=box),
        context.robot.right_arm,
        context=context,
    )
    with world.modify_world():
        box.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(*REACHABILITY_TARGET_POSITION)
        )

    np.testing.assert_allclose(
        location.costmap().origin.to_position().to_np()[:2],
        box.global_pose.to_position().to_np()[:2],
    )


# %% seeing a target


def test_a_visibility_location_takes_its_seed_from_the_context(single_robot_world):
    world, robot, context = single_robot_world
    context.sampling_seed = 5

    location = VisibilityLocation(
        Pose.from_xyz_rpy(*REACHABILITY_TARGET_POSITION, reference_frame=world.root),
        context=context,
    )

    assert location.draw.seed == context.sampling_seed
