from copy import deepcopy
import operator
from itertools import islice
from typing_extensions import Optional

import numpy as np
from numpy.typing import NDArray
import pytest

from coraplex.locations.costmaps import (
    Costmap,
    OccupancyCostmap,
    GaussianCostmap,
    RingCostmap,
)
from krrood.entity_query_language.exceptions import NonPositiveLimitValue
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

# ---- Occupancy locations tests ----


def test_attachment_exclusion(pr2_apartment_context, rclpy_node):

    world, robot_view, context = pr2_apartment_context

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


def test_merge_costmap(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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


@pytest.mark.parametrize("merge", [operator.and_, operator.add], ids=["&", "+"])
def test_a_merged_map_samples_on_the_terms_of_the_map_it_was_merged_into(
    pr2_apartment_context, merge
):
    world, robot_view, context = pr2_apartment_context
    origin = Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 0, 1, world.root)
    first = GaussianCostmap(
        resolution=0.02,
        height=200,
        width=200,
        mean=200,
        sigma=15,
        origin=origin,
        world=world,
        number_of_samples=17,
        seed=3,
    )
    second = GaussianCostmap(
        resolution=0.02,
        height=200,
        width=200,
        mean=200,
        sigma=15,
        origin=origin,
        world=world,
    )

    merged = merge(first, second)

    assert merged.number_of_samples == first.number_of_samples
    assert merged.seed == first.seed


def test_occupancy_robot_exclusion(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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


def test_occupancy_leaves_the_floor_free(pr2_apartment_context):
    """
    The ground the robot drives on is not an obstacle: over a patch of open floor every
    cell stays free, and only what stands on the floor occupies anything.
    """
    world, robot_view, context = pr2_apartment_context

    occupancy_map = OccupancyCostmap(
        resolution=0.02,
        height=50,
        width=50,
        world=world,
        robot_view=robot_view,
        origin=Pose.from_xyz_quaternion(1.5, 2, 0, 0, 0, 0, 1, world.root),
        distance_to_obstacle=0.1,
    )

    assert np.all(occupancy_map.create_ray_mask_around_origin() == 1)


def test_gaussian_costmap(pr2_apartment_context):

    world, robot_view, context = pr2_apartment_context
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=Pose.from_xyz_quaternion(3.1, 2.2, 0, 0, 0, 1, 0, world.root),
        mean=400,
        sigma=150,  # Change back
        world=world,
    )

    # Checks that 5% of the size around the middle is cut out
    assert np.sum(gaussian_map.map == 0) == (400 * 0.05 * 2) ** 2


STRAY_CANDIDATE_SHARE = 0.05
"""
The largest share of candidates a map may offer from a region it barely rates.
"""


def test_a_reachability_map_rates_only_the_side_the_robot_can_reach_from(
    pr2_apartment_context,
):
    """
    Merged with an occupancy map, the reachability map barely rates the far side of what
    the target rests against, and few candidates come from there.
    """
    world, robot_view, context = pr2_apartment_context
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
    poses = list(reach_map.sample(reach_map.number_of_samples, context.sampling_seed))
    from_the_far_side = sum(1 for pose in poses if pose.position.x < 3.0)
    assert from_the_far_side < STRAY_CANDIDATE_SHARE * len(poses)


# ----- Sampling test ---------------


def test_position_generation(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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

    poses = list(gaussian_map.candidates())
    assert poses
    for pose in poses:
        assert 0.8 <= pose.position.x <= 1.2
        assert 0.8 <= pose.position.y <= 1.2


def test_segment_map(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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


def test_sample_x_axis(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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

    poses = list(gaussian_map.candidates())
    assert poses
    for pose in poses:
        assert -0.05 < pose.position.y < 0.05


def test_sample_x_axis_offset(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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

    poses = list(gaussian_map.candidates())
    assert poses
    for pose in poses:
        assert -0.2 <= pose.position.y <= 0.2
        assert 0.4 <= pose.position.x <= 0.8


def test_sample_x_axis_offset_non_id(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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
    poses = list(gaussian_map.candidates())
    assert poses
    for pose in poses:
        assert 1.8 <= pose.position.y <= 2.2 + tolerance
        assert 3.4 <= pose.position.x <= 3.8 + tolerance


def test_sample_to_pose_gau(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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
    poses = list(final_map.candidates())
    assert poses
    for pose in poses:
        assert 1.8 <= pose.position.y <= 2.2 + tolerance
        assert 3.4 <= pose.position.x <= 3.8 + tolerance


def test_sample_y_axis(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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
    poses = list(gaussian_map.candidates())
    assert poses
    for pose in poses:
        assert -0.05 < pose.position.x < 0.05


def test_sample_rotated(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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
    assert len(list(gaussian_map.candidates())) == 2

    for pose in gaussian_map.candidates():
        assert -0.05 < pose.position.y < 0.05
        assert 0.4 <= pose.position.x <= 0.45

    gaussian_map.origin = Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 1, 1, world.root)

    assert len(list(gaussian_map.candidates())) == 2

    for pose in gaussian_map.candidates():
        assert -0.05 < pose.position.y < 0.05
        assert 0.4 <= pose.position.x <= 0.45


def test_sample_to_pose(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context

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

    pose = list(gaussian_map.candidates())[0]

    assert pose.position.x == 1.6
    assert pose.position.y == 2.2
    assert pose.position.z == 0


def test_sample_highest_first(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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

    poses = list(gaussian_map.candidates())

    assert len(poses) == 3

    assert poses[2].position.x < poses[1].position.x < poses[0].position.x
    assert poses[2].position.y < poses[1].position.y < poses[0].position.y


def test_segment_highest_first(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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


def test_segment_empty_map(pr2_apartment_context):
    world, robot_view, context = pr2_apartment_context
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


# %% how a map's ratings decide which candidates it offers


def _ring_map(world) -> RingCostmap:
    """
    :return: A ring a metre across, the shape a standing pose is sampled from.
    """
    return RingCostmap(
        resolution=0.02,
        width=200,
        height=200,
        standard_deviation=15,
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
    costmap: Costmap, seed: Optional[int], count: int
) -> NDArray[np.float64]:
    """
    :return: How far the first ``count`` candidates stand from the map's origin.
    """
    origin = costmap.origin.position.to_np()[:3]
    return np.array(
        [
            float(np.linalg.norm(pose.position.to_np()[:3] - origin))
            for pose in islice(
                costmap.sample(costmap.number_of_samples, seed),
                count,
            )
        ]
    )


def test_weighted_sampling_reaches_the_whole_ring(pr2_apartment_context):
    """
    Sampling reaches stand-off distances off the ring's peak within an affordable
    budget.
    """
    world, _, _ = pr2_apartment_context
    budget = 50

    ring = _ring_map(world)

    assert np.ptp(_stand_off_distances(ring, 0, budget)) > 0.2


def test_weighted_sampling_still_favours_what_the_map_rates_highest(
    pr2_apartment_context,
):
    """
    Weighted sampling stays closer to the ring's stand-off distance than sampling an
    evenly rated map does.
    """
    world, _, _ = pr2_apartment_context
    ring = _ring_map(world)

    weighted_median = float(np.median(_stand_off_distances(ring, 0, 400)))
    ignored_median = float(
        np.median(_stand_off_distances(_everywhere_map(world), 0, 400))
    )

    assert abs(weighted_median - ring.distance) < abs(ignored_median - ring.distance)


def test_a_seeded_sampling_repeats(pr2_apartment_context):
    """
    A run has to be reproducible to be debugged, so a caller can fix the sampling.
    """
    world, _, _ = pr2_apartment_context
    ring = _ring_map(world)

    first = _stand_off_distances(ring, 7, 40)
    again = _stand_off_distances(ring, 7, 40)
    different = _stand_off_distances(ring, 8, 40)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, different)


def test_an_unseeded_sampling_varies(pr2_apartment_context):
    """
    Without a seed each run explores the region afresh, which is the point of sampling
    from the map rather than ranking it.
    """
    world, _, _ = pr2_apartment_context
    ring = _ring_map(world)

    first = _stand_off_distances(ring, None, 40)
    again = _stand_off_distances(ring, None, 40)

    assert not np.array_equal(first, again)


def test_how_many_candidates_to_sample_is_the_callers_to_say(pr2_apartment_context):
    """
    How many candidates a map offers belongs to whoever samples from it, not to the map.

    A map built once is sampled from by callers that can afford to judge different
    numbers of them.
    """
    world, _, _ = pr2_apartment_context
    ring = _ring_map(world)

    few = list(ring.sample(12, None))
    many = list(ring.sample(300, None))

    assert len(few) == 12
    assert len(many) == 300


def test_a_sampled_candidate_faces_the_maps_origin(pr2_apartment_context):
    """
    A candidate says where to stand and which way to look, and looking at the origin is
    what puts whatever the map was built around in front of the robot.
    """
    world, _, _ = pr2_apartment_context
    ring = _ring_map(world)

    sampled = list(
        islice(
            ring.sample(10, None),
            5,
        )
    )

    assert len(sampled) == 5
    for candidate in sampled:
        facing = RotationMatrix.from_quaternion(candidate.quaternion) @ Vector3.X()
        to_origin = ring.origin.position - candidate.position
        assert float(facing.angle_between(to_origin)) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    "x, y", [(1, 0), (-1, 0), (0, 1), (0, -1)], ids=["front", "behind", "left", "right"]
)
def test_a_candidate_faces_the_maps_origin_from_every_side(pr2_apartment_context, x, y):
    world, _, _ = pr2_apartment_context
    ring = _ring_map(world)
    position = Point3(x, y, 0, reference_frame=world.root)

    facing = (
        RotationMatrix.from_quaternion(ring._orientation_facing_origin(position))
        @ Vector3.X()
    )

    to_origin = ring.origin.position - position
    assert float(facing.angle_between(to_origin)) == pytest.approx(0, abs=1e-6)


def test_a_candidate_at_the_maps_origin_still_has_an_orientation(
    pr2_apartment_context,
):
    """
    There is no direction towards the origin from the origin itself, but a candidate
    there still needs one to stand in.
    """
    world, _, _ = pr2_apartment_context
    ring = _ring_map(world)

    orientation = ring._orientation_facing_origin(ring.origin.position)

    assert np.all(np.isfinite(orientation.to_np()))


def _everywhere_map(world) -> RingCostmap:
    """
    :return: A map that rates everywhere alike, for merging without changing an order.
    """
    everywhere = _ring_map(world)
    everywhere.map = np.ones((200, 200))
    return everywhere


# %% asking a map for more candidates than it can offer


def test_a_sparsely_rated_map_is_sampled_from_within_its_rated_entries(
    pr2_apartment_context,
):
    """
    A map whose rated region is a fraction of its extent is the ordinary case, and the
    default sample budget far exceeds what such a map rates.
    """
    world, _, _ = pr2_apartment_context
    costmap = _sparsely_rated_map(world)

    poses = list(costmap.sample(costmap.number_of_samples, 0))

    assert len(poses) == int(np.count_nonzero(costmap.map))


# %% how a sample budget is spread over a map's segments


def test_a_budget_smaller_than_the_segment_count_still_offers_candidates(
    pr2_apartment_context,
):
    """
    A budget smaller than the number of segments is still spent in full.
    """
    world, _, _ = pr2_apartment_context
    costmap = _sparsely_rated_map(world)
    asked_for = len(costmap.segment_map()) - 1

    poses = list(costmap.sample(asked_for, None))

    assert len(poses) == asked_for


def test_a_segment_is_sampled_from_as_much_as_it_is_rated(pr2_apartment_context):
    """
    A segment the map barely rates has to be sampled from barely, or a region worth
    standing in and one worth avoiding are offered alike however the map rates them.
    """
    world, _, _ = pr2_apartment_context
    costmap = _ring_map(world)
    costmap.map = np.zeros((200, 200))
    costmap.map[20:40, 20:40] = 1.0
    costmap.map[120:140, 120:140] = 0.25
    budget = 100

    poses = list(costmap.sample(budget, 0))

    preferred_share = costmap.map[20:40, 20:40].sum() / costmap.map.sum()
    sampled_from_preferred = sum(
        1 for pose in poses if pose.position.x < costmap.origin.position.x
    )
    assert sampled_from_preferred == round(budget * float(preferred_share))


def test_what_a_full_segment_cannot_take_goes_to_the_next(pr2_apartment_context):
    """
    A segment rated far above its size cannot take its whole share, and what it leaves
    over is spent on the next segment rather than lost.
    """
    world, _, _ = pr2_apartment_context
    costmap = _ring_map(world)
    small_and_highly_rated = np.full(4, 1.0)
    large_and_barely_rated = np.full(100, 0.25)

    budgets = costmap._budget_per_segment(
        [small_and_highly_rated, large_and_barely_rated], 50
    )

    assert budgets == [small_and_highly_rated.size, 50 - small_and_highly_rated.size]


def test_a_map_offers_no_more_candidates_than_it_holds(pr2_apartment_context):
    """
    An entry is offered once, so a budget beyond what the map holds is capped at its
    extent rather than running up to the budget over its segments.
    """
    world, _, _ = pr2_apartment_context
    costmap = _ring_map(world)
    costmap.map = np.ones((3, 3))

    poses = list(costmap.sample(100 * costmap.map.size, None))

    assert len(poses) == costmap.map.size


def test_a_map_that_rates_nothing_offers_nothing(pr2_apartment_context):
    """
    An entry rated zero is never a candidate, so a map that rates nothing, such as a
    ring lying wholly inside the furniture, has no pose to offer at all.
    """
    world, _, _ = pr2_apartment_context
    costmap = _ring_map(world)
    costmap.map = np.zeros((200, 200))

    assert list(costmap.sample(costmap.number_of_samples, 0)) == []


@pytest.mark.parametrize("asked_for", [0, -1, -5])
def test_a_map_asked_for_no_candidates_says_so(pr2_apartment_context, asked_for):
    """
    No sampling satisfies a request for fewer than one candidate, so it is refused where
    it is asked for rather than quietly answered with an empty map.

    Refused at the call rather than once the sampling is iterated, so a caller that
    hands the candidates on is told where the mistake is.
    """
    world, _, _ = pr2_apartment_context
    costmap = _ring_map(world)

    with pytest.raises(NonPositiveLimitValue):
        costmap.sample(asked_for, None)
