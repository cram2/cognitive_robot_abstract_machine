from dataclasses import replace

import numpy as np

from coraplex.locations.base import Location
from coraplex.locations.costmaps import GaussianCostmap
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose


def _three_isolated_candidates_location(world, robot_view, context) -> Location:
    """
    A :class:`Location` with three well-separated, collision-free candidates, centred
    in open space away from the apartment geometry (same off-map coordinate
    ``test_costmaps.py``'s ``test_occupancy_robot_exclusion`` uses for the same
    reason). Cell values (3, 2, 1) are placed on a diagonal of increasing x, so the
    generator's own (unscored) order -- highest map value first -- comes out in
    ascending x, giving a known, non-trivial order to assert against.
    """
    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(10, 10)
    )
    origin = Pose.from_xyz_rpy(10, 10, 0, reference_frame=world.root)

    np_map = np.zeros((200, 200))
    np_map[80, 80] = 3
    np_map[100, 100] = 2
    np_map[120, 120] = 1
    gaussian_map = GaussianCostmap(
        resolution=0.02,
        origin=origin,
        mean=200,
        sigma=15,
        world=world,
    )
    gaussian_map.map = np_map

    return Location(context, origin, gaussian_map, [])


def test_iter_without_scorer_preserves_generator_order(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location = _three_isolated_candidates_location(world, robot_view, context)

    poses = list(location)

    assert len(poses) == 3
    xs = [pose.to_position().x for pose in poses]
    assert xs[0] < xs[1] < xs[2]
    assert location.scorer is None


def test_iter_with_scorer_ranks_candidates_by_score(immutable_model_world):
    world, robot_view, context = immutable_model_world
    unscored_location = _three_isolated_candidates_location(world, robot_view, context)

    scored_location = replace(
        unscored_location, scorer=lambda pose, collisions: pose.to_position().x
    )

    ranked_poses = list(scored_location)

    assert len(ranked_poses) == 3
    ranked_xs = [pose.to_position().x for pose in ranked_poses]
    assert ranked_xs[0] > ranked_xs[1] > ranked_xs[2]
    # the scorer reversed the generator's own order, proving it actually re-ranked
    # rather than passing generator order through unchanged
    unscored_xs = [pose.to_position().x for pose in list(unscored_location)]
    assert ranked_xs == list(reversed(unscored_xs))


def test_iter_with_scorer_ranks_only_the_buffered_prefix_and_appends_the_rest(
    immutable_model_world,
):
    """
    ``max_candidates_before_rank`` bounds how many candidates are ranked, not how
    many are yielded: with it set below the number of real candidates, only the
    first that many are buffered and sorted by score -- the rest are still yielded
    afterward, unranked, in their original generator order. Nothing is dropped,
    matching ``ActionBeliefQuery.rank_grounded_actions``'s sibling contract.
    """
    world, robot_view, context = immutable_model_world
    unscored_location = _three_isolated_candidates_location(world, robot_view, context)

    scored_location = replace(
        unscored_location,
        scorer=lambda pose, collisions: pose.to_position().x,
        max_candidates_before_rank=2,
    )

    ranked_poses = list(scored_location)
    ranked_xs = [pose.to_position().x for pose in ranked_poses]
    unscored_xs = [pose.to_position().x for pose in list(unscored_location)]

    assert len(ranked_poses) == 3
    # the first two candidates in generator order were buffered and ranked
    # (descending by score); the third was never buffered, so it is appended
    # afterward unranked, in its original generator-order position
    assert ranked_xs == [unscored_xs[1], unscored_xs[0], unscored_xs[2]]


def test_merge_preserves_scorer_and_max_candidates_before_rank(immutable_model_world):
    """
    ``Location.merge``/``__and__`` must carry the ranking configuration through to the
    merged location -- previously it silently reverted to the defaults.
    """
    world, robot_view, context = immutable_model_world
    location = _three_isolated_candidates_location(world, robot_view, context)
    other = _three_isolated_candidates_location(world, robot_view, context)

    def scorer(pose, collisions):
        return pose.to_position().x

    scored_location = replace(location, scorer=scorer, max_candidates_before_rank=2)

    merged = scored_location & other

    assert merged.scorer is scorer
    assert merged.max_candidates_before_rank == 2
