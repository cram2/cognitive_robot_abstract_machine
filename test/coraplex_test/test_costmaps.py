from copy import deepcopy
from itertools import islice

import numpy as np
from numpy.typing import NDArray
import pytest

from coraplex.locations.costmaps import (
    Costmap,
    OccupancyCostmap,
    GaussianCostmap,
    OrientationGenerator,
    RingCostmap,
)
from coraplex.locations.sampling import (
    CostmapSamplingStrategy,
    HighestRatedFirst,
    UniformlyAtRandom,
    WeightedByRating,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3

# ---- Occupancy locations tests ----


def test_attachment_exclusion(immutable_model_world, rclpy_node):

    world, robot_view, context = immutable_model_world

    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            -1.5, 1, 0, reference_frame=world.root
        )
    )
    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            -1.8, 1, 1, reference_frame=world.root
        )
    )

    test_world = deepcopy(world)
    with test_world.modify_world():
        test_world.move_branch(
            test_world.get_body_by_name("milk.stl"),
            test_world.get_body_by_name("r_gripper_tool_frame"),
        )
    o = OccupancyCostmap(
        distance_to_obstacle=0.2,
        height=200,
        width=200,
        resolution=0.02,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(-1.5, 1, 0, 0, 0, 0, 1, test_world.root),
        world=test_world,
    )

    assert 400 == np.sum(o.map[90:110, 90:110])
    assert np.sum(o.map[80:90, 90:110]) != 0


def test_merge_costmap(immutable_model_world):
    world, robot_view, context = immutable_model_world
    o = OccupancyCostmap(
        distance_to_obstacle=0.2,
        height=200,
        width=200,
        resolution=0.02,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        world=world,
    )
    o2 = OccupancyCostmap(
        distance_to_obstacle=0.2,
        height=200,
        width=200,
        resolution=0.02,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        world=world,
    )
    o3 = o + o2
    assert np.all(o.map == o3.map)
    o2.map[100:120, 100:120] = 0
    o3 = o + o2
    assert np.all(o3.map[100:120, 100:120] == 0)
    assert np.all(o3.map[0:100, 0:100] == o.map[0:100, 0:100])
    o2.map = np.zeros_like(o2.map)
    o3 = o + o2
    assert np.all(o3.map == o2.map)


def test_occupancy_robot_exclusion(immutable_model_world):
    world, robot_view, context = immutable_model_world
    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(10, 10)
    )
    occupancy_map = OccupancyCostmap(
        resolution=0.02,
        height=400,
        width=400,
        world=world,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(10, 10, 0, 0, 0, 0, 1, world.root),
        distance_to_obstacle=0.3,
    )
    assert np.sum(occupancy_map.map) == 137641


def test_gaussian_costmap(immutable_model_world):

    world, robot_view, context = immutable_model_world
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3.1, 2.2, 0, 0, 0, 1, 0, world.root),
        mean=400,
        sigma=150,  # Change back
        world=world,
    )

    # Checks that 5% of the size around the middle is cut out
    assert np.sum(gaussian_map.map == 0) == (400 * 0.05 * 2) ** 2


def test_sample_reachability(immutable_model_world):
    world, robot_view, context = immutable_model_world
    occupancy_map = OccupancyCostmap(
        resolution=0.02,
        height=400,
        width=400,
        world=world,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(3.0, 2.2, 0, 0, 0, 1, 0, world.root),
        distance_to_obstacle=0.3,
    )

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3.0, 2.2, 0, 0, 0, 1, 0, world.root),
        mean=400,
        sigma=15,  # Change back
        world=world,
    )

    reach_map = occupancy_map + gaussian_map

    assert np.sum(reach_map.map[:200, :]) < 5

    for pose in reach_map.candidates(HighestRatedFirst()):
        assert pose.to_position().x > 3


# ----- Sampling test ---------------


def test_position_generation(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[90:110, 90:110] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(1, 1, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,  # Change back
        world=world,
    )
    gaussian_map.map = np_map

    for pose in gaussian_map.candidates(HighestRatedFirst()):
        assert 0.8 <= pose.to_position().x <= 1.2
        assert 0.8 <= pose.to_position().y <= 1.2


def test_segment_map(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[90:110, 90:110] = 1
    np_map[20:40, 20:40] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(1, 1, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    seg_maps = gaussian_map.segment_map()

    assert len(seg_maps) == 2
    map_2 = seg_maps[0] if np.sum(seg_maps[0][20:40, 20:40]) > 1 else seg_maps[1]
    map_1 = seg_maps[1] if np.sum(seg_maps[1][90:110, 90:110]) > 1 else seg_maps[0]

    assert np.sum(map_2[20:40, 20:40]) == 20**2 and np.sum(map_2[90:110, 90:110]) == 0
    assert np.sum(map_1[90:110, 90:110]) == 20**2 and np.sum(map_1[20:40, 20:40]) == 0


def test_orientation_generation(immutable_model_world):
    world, robot_view, context = immutable_model_world

    orientation = OrientationGenerator.generate_origin_orientation(
        Point3(0, 1, 0),
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
    )

    assert orientation.to_list() == pytest.approx([0, 0, -0.707, 0.707], abs=0.001)

    orientation = OrientationGenerator.generate_origin_orientation(
        Point3(0, -1, 0),
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
    )

    assert orientation.to_list() == pytest.approx([0, 0, 0.707, 0.707], abs=0.001)


def test_sample_x_axis(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[:, 99:101] = 1

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    for pose in gaussian_map.candidates(HighestRatedFirst()):
        assert -0.05 < pose.to_position().y < 0.05


def test_sample_x_axis_offset(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:140, 90:110] = 1

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    for pose in gaussian_map.candidates(HighestRatedFirst()):
        assert -0.2 <= pose.to_position().y <= 0.2
        assert 0.4 <= pose.to_position().x <= 0.8


def test_sample_x_axis_offset_non_id(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:140, 90:110] = 1

    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3, 2, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    tolerance = 0.01
    for pose in gaussian_map.candidates(HighestRatedFirst()):
        assert 1.8 <= pose.to_position().y <= 2.2 + tolerance
        assert 3.4 <= pose.to_position().x <= 3.8 + tolerance


def test_sample_to_pose_gau(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:140, 90:110] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3, 2, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    gaussian_map2 = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3, 2, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    final_map = gaussian_map + gaussian_map2

    # The merge keeps only the cells both maps cover, which is the box the first one was
    # given: rows 120:140 and columns 90:110 of a 0.02 m grid centred on the origin.
    tolerance = 0.01
    for pose in final_map.candidates(HighestRatedFirst()):
        assert 1.8 <= pose.to_position().y <= 2.2 + tolerance
        assert 3.4 <= pose.to_position().x <= 3.8 + tolerance


def test_sample_y_axis(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[99:101, :] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map
    for pose in gaussian_map.candidates(HighestRatedFirst()):
        assert -0.05 < pose.to_position().x < 0.05


def test_sample_rotated(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[120:121, 99:101] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map
    assert len(list(gaussian_map.candidates(HighestRatedFirst()))) == 2

    for pose in gaussian_map.candidates(HighestRatedFirst()):
        assert -0.05 < pose.to_position().y < 0.05
        assert 0.4 <= pose.to_position().x <= 0.45

    gaussian_map.origin = Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 1, 1, world.root)

    assert len(list(gaussian_map.candidates(HighestRatedFirst()))) == 2

    for pose in gaussian_map.candidates(HighestRatedFirst()):
        assert -0.05 < pose.to_position().y < 0.05
        assert 0.4 <= pose.to_position().x <= 0.45


def test_sample_to_pose(immutable_model_world):
    world, robot_view, context = immutable_model_world

    np_map = np.zeros((200, 200))
    np_map[130, 160] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(1, 1, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    pose = list(gaussian_map.candidates(HighestRatedFirst()))[0]

    assert pose.to_position().x == 1.6
    assert pose.to_position().y == 2.2
    assert pose.to_position().z == 0


def test_sample_highest_first(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[40, 40] = 1
    np_map[80, 80] = 2
    np_map[120, 120] = 3
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )

    gaussian_map.map = np_map

    poses = list(gaussian_map.candidates(HighestRatedFirst()))

    assert len(poses) == 3

    assert (
        poses[2].to_position().x < poses[1].to_position().x < poses[0].to_position().x
    )
    assert (
        poses[2].to_position().y < poses[1].to_position().y < poses[0].to_position().y
    )


def test_segment_highest_first(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    np_map[40:45, 40:45] = 1
    np_map[80:85, 80:85] = 3
    np_map[120:125, 120:125] = 2
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    segmented_maps = gaussian_map.segment_map()

    assert len(segmented_maps) == 3
    assert np.max(segmented_maps[0]) == 3
    assert np.max(segmented_maps[1]) == 2
    assert np.max(segmented_maps[2]) == 1


def test_segment_empty_map(immutable_model_world):
    world, robot_view, context = immutable_model_world
    np_map = np.zeros((200, 200))
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose(reference_frame=world.root),
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    segmented_maps = gaussian_map.segment_map()

    assert len(segmented_maps) == 1
    assert np.sum(segmented_maps[0]) == 0


def test_orientation_generator_by_axis_y(immutable_model_world):
    world, robot_view, context = immutable_model_world

    ori_gen = OrientationGenerator.orientation_generator_for_axis(
        Vector3.from_iterable([0, 1, 0])
    )

    origin_pose = Pose(reference_frame=world.root)
    target_position = Point3.from_iterable([1, 0, 0])

    generated_orientation = ori_gen(target_position, origin_pose)

    assert generated_orientation.to_list() == pytest.approx(
        [0, 0, 0.7071, 0.7071], abs=0.001
    )


def test_orientation_generator_by_axis_minus_y(immutable_model_world):
    world, robot_view, context = immutable_model_world

    ori_gen = OrientationGenerator.orientation_generator_for_axis(
        Vector3.from_iterable([0, -1, 0])
    )

    origin_pose = Pose(reference_frame=world.root)
    target_position = Point3.from_iterable([1, 0, 0])

    generated_orientation = ori_gen(target_position, origin_pose)

    assert generated_orientation.to_list() == pytest.approx(
        [0, 0, -0.7071, 0.7071], abs=0.001
    )


def test_orientation_generator_by_axis_x(immutable_model_world):
    world, robot_view, context = immutable_model_world

    ori_gen = OrientationGenerator.orientation_generator_for_axis(
        Vector3.from_iterable([1, 0, 0])
    )

    origin_pose = Pose(reference_frame=world.root)
    target_position = Point3.from_iterable([1, 0, 0])

    generated_orientation = ori_gen(target_position, origin_pose)

    assert generated_orientation.to_list() == pytest.approx([0, 0, 1, 0], abs=0.001)


# %% how a map's ratings decide which candidates it offers


def _ring_map(world) -> RingCostmap:
    """
    :return: A ring a metre across, the shape a standing pose is drawn from.
    """
    return RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        std=15,
        distance=0.6,
        world=world,
        origin=Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root),
    )


def _sparsely_rated_map(world) -> RingCostmap:
    """
    :return: A map rating three entries apart from one another, the rest of it zero.
    """
    costmap = _ring_map(world)
    costmap.map = np.zeros((200, 200))
    costmap.map[40, 40] = 1
    costmap.map[80, 80] = 2
    costmap.map[120, 120] = 3
    return costmap


def _stand_off_distances(
    costmap: Costmap, sampling_strategy: CostmapSamplingStrategy, count: int
) -> NDArray[np.float64]:
    """
    :return: How far the first ``count`` candidates stand from the map's origin.
    """
    origin = costmap.origin.to_position().to_np()[:3]
    return np.array(
        [
            float(np.linalg.norm(pose.to_position().to_np()[:3] - origin))
            for pose in islice(costmap.candidates(sampling_strategy), count)
        ]
    )


def test_highest_rated_candidates_come_first(immutable_model_world):
    """
    Ranking a map is what lets a caller take the first candidate that passes its own
    checks, so the highest rated entry has to be offered before any lower one.
    """
    world, _, _ = immutable_model_world
    costmap = _sparsely_rated_map(world)

    poses = list(costmap.candidates(HighestRatedFirst()))

    assert len(poses) == 3
    assert (
        poses[2].to_position().x < poses[1].to_position().x < poses[0].to_position().x
    )


def test_weighted_sampling_reaches_the_whole_ring(immutable_model_world):
    """
    A ring says a stand-off distance is likely, not that it is the only one worth
    trying.

    Ranking offers a caller the ring's own radius over and over, one angle at a time, so
    a pose that needs a few centimetres more never comes up inside the budget a caller
    can afford to simulate.
    """
    world, _, _ = immutable_model_world
    budget = 50

    ring = _ring_map(world)
    ranked_spread = np.ptp(_stand_off_distances(ring, HighestRatedFirst(), budget))
    weighted_spread = np.ptp(
        _stand_off_distances(ring, WeightedByRating(seed=0), budget)
    )

    assert ranked_spread < 0.05
    assert weighted_spread > 0.2


def test_weighted_sampling_still_favours_what_the_map_rates_highest(
    immutable_model_world,
):
    """
    Weighting has to follow the map rather than ignore it, or the ring stops meaning
    anything and the robot is as likely to stand anywhere.

    A ring in the plane holds more entries the further out they sit, so the draw is
    pulled outwards whatever the ratings say. What the rating buys is how much closer to
    the ring the draw stays than it would without one.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    weighted_median = float(
        np.median(_stand_off_distances(ring, WeightedByRating(seed=0), 400))
    )
    ignored_median = float(
        np.median(_stand_off_distances(ring, UniformlyAtRandom(seed=0), 400))
    )

    assert abs(weighted_median - ring.distance) < abs(ignored_median - ring.distance)


def test_uniform_sampling_ignores_what_the_map_rates(immutable_model_world):
    """
    Uniform sampling is the deliberate opposite of following the map, for a caller that
    wants the region covered rather than its best part, so it spreads further than a
    draw the ratings steer.
    """
    world, _, _ = immutable_model_world

    ring = _ring_map(world)

    assert np.ptp(_stand_off_distances(ring, UniformlyAtRandom(seed=0), 400)) > np.ptp(
        _stand_off_distances(ring, WeightedByRating(seed=0), 400)
    )


def test_a_seeded_draw_repeats(immutable_model_world):
    """
    A run has to be reproducible to be debugged, so a caller can fix the draw.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    first = _stand_off_distances(ring, WeightedByRating(seed=7), 40)
    again = _stand_off_distances(ring, WeightedByRating(seed=7), 40)
    different = _stand_off_distances(ring, WeightedByRating(seed=8), 40)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, different)


def test_an_unseeded_draw_varies(immutable_model_world):
    """
    Without a seed each run explores the region afresh, which is the point of drawing
    from the map rather than ranking it.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    first = _stand_off_distances(ring, WeightedByRating(), 40)
    again = _stand_off_distances(ring, WeightedByRating(), 40)

    assert not np.array_equal(first, again)


def test_how_many_candidates_to_draw_is_the_callers_to_say(immutable_model_world):
    """
    How many candidates a map offers belongs to whoever draws from it, not to the map.

    A map built once is drawn from by callers that can afford to judge different numbers
    of them.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    few = list(ring.candidates(HighestRatedFirst(), number_of_samples=12))
    many = list(ring.candidates(HighestRatedFirst(), number_of_samples=300))

    assert len(few) == 12
    assert len(many) == 300


def test_which_way_a_candidate_faces_is_the_callers_to_say(immutable_model_world):
    """
    The orientation a candidate is offered with is part of drawing from the map, so it
    is chosen where the draw is, and survives however the map was built up.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)
    facing_y = OrientationGenerator.orientation_generator_for_axis(
        Vector3.from_iterable([0, 1, 0])
    )

    merged = ring & _everywhere_map(world)
    drawn = next(
        iter(
            merged.candidates(
                HighestRatedFirst(),
                number_of_samples=10,
                orientation_generator=facing_y,
            )
        )
    )

    expected = facing_y(drawn.to_position(), merged.origin)
    assert drawn.to_quaternion().to_list() == pytest.approx(
        expected.to_list(), abs=1e-6
    )


def _everywhere_map(world) -> RingCostmap:
    """
    :return: A map that rates everywhere alike, for merging without changing an order.
    """
    everywhere = _ring_map(world)
    everywhere.map = np.ones((200, 200))
    return everywhere


# %% asking a map for more candidates than it can offer


def _sparsely_rated_entries() -> NDArray[np.float64]:
    """
    :return: A hundred entries, all but three of them rated zero.
    """
    ratings = np.zeros(100)
    ratings[[7, 13, 61]] = [1.0, 2.0, 3.0]
    return ratings


def test_weighted_draw_offers_every_rated_entry_when_asked_for_more():
    """
    A drawn entry has to be one the map rates, so a map rating fewer entries than a
    caller asks for offers the ones it rates rather than refusing the draw.
    """
    ratings = _sparsely_rated_entries()

    drawn = WeightedByRating(seed=0).choose(ratings, 50)

    assert sorted(drawn.tolist()) == np.flatnonzero(ratings).tolist()


def test_uniform_draw_offers_every_entry_when_asked_for_more():
    """
    Drawing without repeating runs out at the size of the map, so asking for more than
    it holds offers all of it.
    """
    ratings = _sparsely_rated_entries()

    drawn = UniformlyAtRandom(seed=0).choose(ratings, 2 * ratings.size)

    assert sorted(drawn.tolist()) == list(range(ratings.size))


def test_ranking_offers_every_entry_when_asked_for_more():
    """
    Ranking runs out at the size of the map too, the highest rated entries still coming
    first.
    """
    ratings = _sparsely_rated_entries()

    ranked = HighestRatedFirst().choose(ratings, 2 * ratings.size)

    assert len(ranked) == ratings.size
    assert ranked[:3].tolist() == np.argsort(ratings)[::-1][:3].tolist()


def test_asking_for_no_candidates_offers_none():
    """
    A caller that asks for no candidates gets none, rather than the whole map or an
    error.
    """
    ratings = _sparsely_rated_entries()

    assert WeightedByRating(seed=0).choose(ratings, 0).size == 0
    assert UniformlyAtRandom(seed=0).choose(ratings, 0).size == 0
    assert HighestRatedFirst().choose(ratings, 0).size == 0


def test_a_sparsely_rated_map_is_drawn_from_within_its_rated_entries(
    immutable_model_world,
):
    """
    A map whose rated region is a fraction of its extent is the ordinary case, and the
    default sample budget far exceeds what such a map rates.
    """
    world, _, _ = immutable_model_world
    costmap = _sparsely_rated_map(world)

    poses = list(costmap.candidates(WeightedByRating(seed=0)))

    assert len(poses) == int(np.count_nonzero(costmap.map))
