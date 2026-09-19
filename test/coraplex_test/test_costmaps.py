from copy import deepcopy
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
from coraplex.exceptions import NonPositiveNumberOfSamples
from coraplex.locations.sampling import CandidateDraw
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

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


def test_occupancy_leaves_the_floor_free(immutable_model_world):
    """
    The ground the robot drives on is not an obstacle: over a patch of open floor every
    cell stays free, and only what stands on the floor occupies anything.
    """
    world, robot_view, context = immutable_model_world

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


def test_a_reachability_map_rates_only_the_side_the_robot_can_reach_from(
    immutable_model_world,
):
    """
    Merging an occupancy map into a gaussian one is what keeps a target's standing poses
    off the far side of whatever it rests against.

    Which of the rated entries a draw then offers is the draw's business; a stray
    candidate is what the collision and reachability checks on a location are for.
    """
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

    for pose in gaussian_map.candidates(CandidateDraw()):
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

    for pose in gaussian_map.candidates(CandidateDraw()):
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

    for pose in gaussian_map.candidates(CandidateDraw()):
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
    for pose in gaussian_map.candidates(CandidateDraw()):
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
    for pose in final_map.candidates(CandidateDraw()):
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
    for pose in gaussian_map.candidates(CandidateDraw()):
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
    assert len(list(gaussian_map.candidates(CandidateDraw()))) == 2

    for pose in gaussian_map.candidates(CandidateDraw()):
        assert -0.05 < pose.to_position().y < 0.05
        assert 0.4 <= pose.to_position().x <= 0.45

    gaussian_map.origin = Pose.from_xyz_quaternion(0, 0, 0, 0, 0, 1, 1, world.root)

    assert len(list(gaussian_map.candidates(CandidateDraw()))) == 2

    for pose in gaussian_map.candidates(CandidateDraw()):
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

    pose = list(gaussian_map.candidates(CandidateDraw()))[0]

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

    poses = list(gaussian_map.candidates(CandidateDraw()))

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


# %% how a map's ratings decide which candidates it offers


def _ring_map(world) -> RingCostmap:
    """
    :return: A ring a metre across, the shape a standing pose is drawn from.
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
    origin = costmap.origin.to_position().to_np()[:3]
    return np.array(
        [
            float(np.linalg.norm(pose.to_position().to_np()[:3] - origin))
            for pose in islice(
                costmap.candidates(CandidateDraw(seed=seed)),
                count,
            )
        ]
    )


def test_weighted_sampling_reaches_the_whole_ring(immutable_model_world):
    """
    A ring says a stand-off distance is likely, not that it is the only one worth
    trying, so a pose needing a few centimetres more has to come up inside the budget a
    caller can afford to simulate.
    """
    world, _, _ = immutable_model_world
    budget = 50

    ring = _ring_map(world)

    assert np.ptp(_stand_off_distances(ring, 0, budget)) > 0.2


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

    weighted_median = float(np.median(_stand_off_distances(ring, 0, 400)))
    ignored_median = float(
        np.median(_stand_off_distances(_everywhere_map(world), 0, 400))
    )

    assert abs(weighted_median - ring.distance) < abs(ignored_median - ring.distance)


def test_a_seeded_draw_repeats(immutable_model_world):
    """
    A run has to be reproducible to be debugged, so a caller can fix the draw.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    first = _stand_off_distances(ring, 7, 40)
    again = _stand_off_distances(ring, 7, 40)
    different = _stand_off_distances(ring, 8, 40)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, different)


def test_an_unseeded_draw_varies(immutable_model_world):
    """
    Without a seed each run explores the region afresh, which is the point of drawing
    from the map rather than ranking it.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    first = _stand_off_distances(ring, None, 40)
    again = _stand_off_distances(ring, None, 40)

    assert not np.array_equal(first, again)


def test_how_many_candidates_to_draw_is_the_callers_to_say(immutable_model_world):
    """
    How many candidates a map offers belongs to whoever draws from it, not to the map.

    A map built once is drawn from by callers that can afford to judge different numbers
    of them.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    few = list(ring.candidates(CandidateDraw(number_of_samples=12)))
    many = list(ring.candidates(CandidateDraw(number_of_samples=300)))

    assert len(few) == 12
    assert len(many) == 300


def test_a_drawn_candidate_faces_the_maps_origin(immutable_model_world):
    """
    A candidate says where to stand and which way to look, and looking at the origin is
    what puts whatever the map was built around in front of the robot.
    """
    world, _, _ = immutable_model_world
    ring = _ring_map(world)

    drawn = list(
        islice(
            ring.candidates(CandidateDraw(number_of_samples=10)),
            5,
        )
    )

    assert len(drawn) == 5
    for candidate in drawn:
        facing = RotationMatrix.from_quaternion(candidate.to_quaternion()) @ Vector3.X()
        to_origin = ring.origin.to_position() - candidate.to_position()
        assert float(facing.angle_between(to_origin)) == pytest.approx(0, abs=1e-6)


def _everywhere_map(world) -> RingCostmap:
    """
    :return: A map that rates everywhere alike, for merging without changing an order.
    """
    everywhere = _ring_map(world)
    everywhere.map = np.ones((200, 200))
    return everywhere


# %% asking a map for more candidates than it can offer


def test_a_sparsely_rated_map_is_drawn_from_within_its_rated_entries(
    immutable_model_world,
):
    """
    A map whose rated region is a fraction of its extent is the ordinary case, and the
    default sample budget far exceeds what such a map rates.
    """
    world, _, _ = immutable_model_world
    costmap = _sparsely_rated_map(world)

    poses = list(costmap.candidates(CandidateDraw(seed=0)))

    assert len(poses) == int(np.count_nonzero(costmap.map))


# %% how a sample budget is spread over a map's segments


def test_a_budget_smaller_than_the_segment_count_still_offers_candidates(
    immutable_model_world,
):
    """
    A budget is spread over the segments a map falls into, and a caller that can only
    afford a handful still has to be offered that handful.

    Sharing the budget out by rating leaves nothing for any segment once it is smaller
    than the number of them, which would have the map offer nothing at all.
    """
    world, _, _ = immutable_model_world
    costmap = _sparsely_rated_map(world)
    asked_for = len(costmap.segment_map()) - 1

    poses = list(costmap.candidates(CandidateDraw(number_of_samples=asked_for)))

    assert len(poses) == asked_for


def test_a_segment_is_drawn_from_as_much_as_it_is_rated(immutable_model_world):
    """
    A segment the map barely rates has to be drawn from barely, or a region worth
    standing in and one worth avoiding are offered alike however the map rates them.
    """
    world, _, _ = immutable_model_world
    costmap = _ring_map(world)
    costmap.map = np.zeros((200, 200))
    costmap.map[20:40, 20:40] = 1.0
    costmap.map[120:140, 120:140] = 0.25
    budget = 100

    poses = list(costmap.candidates(CandidateDraw(number_of_samples=budget, seed=0)))

    preferred_share = costmap.map[20:40, 20:40].sum() / costmap.map.sum()
    drawn_from_preferred = sum(
        1 for pose in poses if pose.to_position().x < costmap.origin.to_position().x
    )
    assert drawn_from_preferred == round(budget * float(preferred_share))


def test_a_map_offers_no_more_candidates_than_it_holds(immutable_model_world):
    """
    An entry is offered once, so a budget beyond what the map holds is capped at its
    extent rather than running up to the budget over its segments.
    """
    world, _, _ = immutable_model_world
    costmap = _ring_map(world)
    costmap.map = np.ones((3, 3))

    poses = list(
        costmap.candidates(CandidateDraw(number_of_samples=100 * costmap.map.size))
    )

    assert len(poses) == costmap.map.size


@pytest.mark.parametrize("asked_for", [0, -1, -5])
def test_a_map_asked_for_no_candidates_says_so(immutable_model_world, asked_for):
    """
    No draw satisfies a request for fewer than one candidate, so it is refused where it
    is asked for rather than quietly answered with an empty map.

    Refused at the call rather than once the draw is iterated, so a caller that hands
    the candidates on is told where the mistake is.
    """
    world, _, _ = immutable_model_world
    costmap = _ring_map(world)

    with pytest.raises(NonPositiveNumberOfSamples):
        costmap.candidates(CandidateDraw(number_of_samples=asked_for))
