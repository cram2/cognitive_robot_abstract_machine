from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice

import numpy as np
import pytest
from typing_extensions import Iterator, List, Optional, Tuple

from coraplex.datastructures.dataclasses import Context
from coraplex.locations.base import Location
from coraplex.locations.costmaps import RingCostmap
from coraplex.locations.locations import ReachabilityLocation, VisibilityLocation
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
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
class RecordsHowItWasSampled(Location):
    """
    Yields one candidate and records the terms it sampled with.
    """

    pose: Pose
    """
    The single candidate to yield.
    """

    sampled_with: List[Tuple[int, Optional[int]]] = field(default_factory=list)
    """
    One entry per call: the number of samples and the seed it sampled with.
    """

    def candidates(self) -> Iterator[Pose]:
        self.sampled_with.append((self.number_of_samples, self.seed))
        return iter([self.pose])


# %% a specification-built world whose odom is displaced

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom offset stays
# in that plane.
_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(0.5, 0.5, 0, yaw=np.pi / 2)


def _world_with_a_robot_behind_a_displaced_odom(
    world_T_odom: HomogeneousTransformationMatrix,
) -> World:
    """
    A world holding nothing but a PR2, reached through a displaced odom.
    """
    specification = WorldSpecification(
        world_parser=None,
        robots=[
            RobotSpecification(semantic_annotation_type=PR2, world_T_odom=world_T_odom)
        ],
    )
    try:
        return specification.to_domain_object()
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")


@pytest.fixture(scope="session")
def _single_robot_world_setup() -> World:
    return _world_with_a_robot_behind_a_displaced_odom(_ODOM)


@pytest.fixture
def single_robot_world(_single_robot_world_setup):
    world = deepcopy(_single_robot_world_setup)
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    return world, robot, Context(world, robot)


def _candidate(world: World) -> Pose:
    return Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)


# %% a location samples its candidates on its own terms


def test_a_location_samples_on_the_terms_it_was_given(single_robot_world):
    world, robot, context = single_robot_world
    location = RecordsHowItWasSampled(
        pose=_candidate(world), number_of_samples=17, seed=3
    )

    list(islice(iter(location), 1))

    assert location.sampled_with == [(17, 3)]


def test_a_location_samples_nothing_before_it_is_consumed(single_robot_world):
    """
    A location handed to a plan as a domain is only sampled from once the plan asks for
    a pose, so it reflects the world at that moment.
    """
    world, robot, context = single_robot_world
    location = RecordsHowItWasSampled(pose=_candidate(world))

    candidates = iter(location)
    assert location.sampled_with == []

    next(candidates)
    assert location.sampled_with == [(location.number_of_samples, location.seed)]


def test_a_location_grounds_to_its_first_candidate(single_robot_world):
    world, robot, context = single_robot_world
    location = RecordsHowItWasSampled(pose=_candidate(world))

    assert location.ground() is location.pose


def test_a_location_that_does_not_say_how_it_samples_cannot_be_built():
    """
    A location inherits no sampling of its own, so one that leaves the terms unanswered
    is refused where it is defined rather than silently offering nothing at runtime.
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
    location = ReachabilityLocation(target, arm, context=context)

    ring = RingCostmap.from_arm_reach_distance(
        context,
        arm,
        target,
        reach_fraction=REACH_FRACTION,
        resolution=location.map_resolution,
        cells=location.map_cells,
        standard_deviation=location.ring_standard_deviation,
    )

    assert float(ring.distance) == pytest.approx(expected_distance)


POSES_CHECKED = 400
"""
How many of the standing poses a reachability location offers are checked.
"""


def _horizontal_distance(pose: Pose, target: Pose) -> float:
    """
    :return: How far `pose` stands from `target` along the floor.
    """
    return float(
        np.linalg.norm(pose.position.to_np()[:2] - target.position.to_np()[:2])
    )


def test_a_reachability_location_stands_around_the_reach_fraction_of_the_arm(
    single_robot_world,
):
    """
    The ring the candidates are sampled from is centred on the stand-off distance, so
    the poses offered stand that far off on the whole.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    arm = context.robot.right_arm
    location = ReachabilityLocation(
        target, arm, context=context, seed=0, reach_fraction=REACH_FRACTION
    )
    ring = RingCostmap.from_arm_reach_distance(
        context,
        arm,
        target,
        reach_fraction=REACH_FRACTION,
        resolution=location.map_resolution,
        cells=location.map_cells,
        standard_deviation=location.ring_standard_deviation,
    )

    stand_offs = [
        _horizontal_distance(pose, target)
        for pose in islice(location.candidates(), POSES_CHECKED)
    ]

    assert float(np.median(stand_offs)) == pytest.approx(
        float(ring.distance), abs=ring.standard_deviation * ring.resolution
    )


def test_a_reachability_location_offers_no_standing_pose_farther_than_the_arm_is_long(
    single_robot_world,
):
    """
    A target farther from where the robot stands than its arm is long cannot be reached
    from there, so such a standing pose is not worth trying.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    arm = context.robot.right_arm
    location = ReachabilityLocation(target, arm, context=context, seed=0)
    sampled = islice(
        location.costmap().sample(location.number_of_samples, location.seed),
        POSES_CHECKED,
    )
    assert any(
        _horizontal_distance(pose, target) > float(arm.approximate_length())
        for pose in sampled
    ), "the map has to offer a pose out of reach for leaving it out to show"

    offered = islice(location.candidates(), POSES_CHECKED)

    assert max(_horizontal_distance(pose, target) for pose in offered) <= float(
        arm.approximate_length()
    )


def test_a_reachability_location_offers_the_poses_in_reach_in_the_order_sampled(
    single_robot_world,
):
    """
    Leaving out what is out of reach does not change which of the remaining poses come
    first, so a plan that found a standing pose before still finds the same one.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    arm = context.robot.right_arm
    location = ReachabilityLocation(target, arm, context=context, seed=0)
    in_reach = [
        pose.position.to_np()[:2]
        for pose in islice(
            location.costmap().sample(location.number_of_samples, location.seed),
            POSES_CHECKED,
        )
        if _horizontal_distance(pose, target) <= float(arm.approximate_length())
    ]

    offered = [
        pose.position.to_np()[:2]
        for pose in islice(location.candidates(), len(in_reach))
    ]

    np.testing.assert_allclose(offered, in_reach)


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


def test_a_reachability_location_is_sampled_around_its_target(single_robot_world):
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )

    location = ReachabilityLocation(target, context.robot.right_arm, context=context)

    np.testing.assert_allclose(
        location.costmap().origin.position.to_np()[:2],
        target.position.to_np()[:2],
    )


def test_a_reachability_location_takes_its_seed_from_the_context(single_robot_world):
    """
    A demonstration is only worth running as a regression test if it runs the same way
    twice, so a plan can fix the samples made anywhere inside it.
    """
    world, robot, context = single_robot_world
    context.sampling_seed = 5

    location = ReachabilityLocation(
        _box_in(world).root.global_pose,
        context.robot.right_arm,
        context=context,
    )

    assert location.seed == context.sampling_seed


def test_a_location_keeps_a_seed_of_its_own_over_the_contexts(single_robot_world):
    world, robot, context = single_robot_world
    context.sampling_seed = 5

    location = ReachabilityLocation(
        _box_in(world).root.global_pose,
        context.robot.right_arm,
        context=context,
        seed=7,
    )

    assert location.seed == 7


def test_a_reachability_location_samples_afresh_without_one(single_robot_world):
    """
    Left unseeded a plan explores the region differently each run, which is what makes
    sampling from the map worth more than ranking it.
    """
    world, robot, context = single_robot_world

    location = ReachabilityLocation(
        _box_in(world).root.global_pose,
        context.robot.right_arm,
        context=context,
    )

    assert location.seed is None


# %% a location reflects the world when it is sampled from


def test_a_costmap_location_builds_its_costmap_only_when_sampled_from(
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
    A target named relative to a body is where that body is when the location is sampled
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
        location.costmap().origin.position.to_np()[:2],
        box.global_pose.position.to_np()[:2],
    )


# %% seeing a target


def test_a_visibility_location_takes_its_seed_from_the_context(single_robot_world):
    world, robot, context = single_robot_world
    context.sampling_seed = 5

    location = VisibilityLocation(
        Pose.from_xyz_rpy(*REACHABILITY_TARGET_POSITION, reference_frame=world.root),
        context=context,
    )

    assert location.seed == context.sampling_seed


def test_a_visibility_location_offers_poses_facing_its_target(single_robot_world):
    """
    A pose to see a target from is only any use looking at it.
    """
    world, robot, context = single_robot_world
    target = Pose.from_xyz_rpy(
        *REACHABILITY_TARGET_POSITION, reference_frame=world.root
    )
    location = VisibilityLocation(target, context=context, seed=0)

    for pose in islice(location.candidates(), POSES_CHECKED):
        heading = pose.rotation_matrix.to_np()[:2, 0]
        offset = target.position.to_np()[:2] - pose.position.to_np()[:2]
        np.testing.assert_allclose(heading, offset / np.linalg.norm(offset), atol=1e-6)
