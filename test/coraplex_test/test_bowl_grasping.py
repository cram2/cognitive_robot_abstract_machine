from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
import trimesh
from trimesh.proximity import closest_point

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.part_finding.locating import LocatedFacesFinder
from semantic_digital_twin.part_finding.splitting import PartSplitter
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Rim
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from ..conftest import SAMPLING_SEED

# %% fixtures

BOWL_MESH = str(
    Path(__file__).resolve().parents[2]
    / "coraplex"
    / "resources"
    / "objects"
    / "bowl.stl"
)
"""
The bowl the demos transport, an irregular scan rather than a turned shape.
"""

GRIPPABLE_DISTANCE = 0.005
"""
How far from the bowl's surface the fingers may close and still find material, in
meters.
"""


def split_off_rim(bowl: Bowl, rim_depth: float = 0.02) -> Rim:
    """
    Split the bowl's rim off its shape, at the faces whose centers lie within
    ``rim_depth`` of its highest point.

    :param bowl: The bowl, its rim still part of its body's shape.
    :param rim_depth: How far below the bowl's highest point its rim reaches, in meters.
    :return: The rim.
    """
    mesh = bowl.root.collision.combined_mesh
    centers = mesh.triangles_center
    rim_centers = centers[centers[:, 2] > mesh.bounds[1][2] - rim_depth]
    return PartSplitter(
        part_type=Rim, finder=LocatedFacesFinder(face_centers=rim_centers)
    ).split(bowl)


@pytest.fixture
def bowl() -> Bowl:
    world = STLParser(BOWL_MESH).parse()
    annotation = Bowl(root=world.get_body_by_name("bowl.stl"))
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    return annotation


@pytest.fixture
def bowl_with_rim(bowl) -> Bowl:
    """
    The bowl with its rim split off as a part of its own.
    """
    split_off_rim(bowl)
    return bowl


def distances_to_surface(
    bowl: Bowl, positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    :param bowl: The bowl whose surface to measure against.
    :param positions: Points in the bowl's own frame.
    :return: Each point's distance to the nearest point of the bowl's surface: that of
        its body and of its rim, if split off, which shares the body's frame.
    """
    surface = trimesh.util.concatenate([body.combined_mesh for body in bowl.bodies])
    _, distances, _ = closest_point(surface, positions)
    return distances


# %% grasping the bowl


def test_bowl_grasps_close_on_the_bowls_wall(bowl):
    """
    The fingers must meet material.

    Grasping the bowl at its own origin -- which is what an object without grasp
    candidates of its own offers -- closes them in mid air inside the bowl.
    """
    positions = np.array(
        [grasp.grasp_pose.to_np()[:3, 3] for grasp in bowl.grasp_candidates()]
    )

    assert len(positions) == bowl.grasp_candidate_count
    assert np.all(distances_to_surface(bowl, positions) < GRIPPABLE_DISTANCE)


def test_grasps_of_a_split_off_rim_close_on_the_bowls_wall(bowl_with_rim):
    positions = np.array(
        [grasp.grasp_pose.to_np()[:3, 3] for grasp in bowl_with_rim.grasp_candidates()]
    )

    assert len(positions) == bowl_with_rim.rim.grasp_candidate_count
    assert np.all(distances_to_surface(bowl_with_rim, positions) < GRIPPABLE_DISTANCE)


def test_the_bowls_origin_is_not_grippable(bowl):
    """
    Guards the premise of the test above: the origin really does sit in mid air, so
    grasping there is a bug rather than a matter of taste.
    """
    [distance] = distances_to_surface(bowl, np.zeros((1, 3)))

    assert distance > GRIPPABLE_DISTANCE


# %% transporting a bowl


@pytest.fixture
def pr2_and_bowl(simple_pr2_context):
    """
    A PR2 in a world with the demos' bowl standing on the counter.
    """
    world, robot, _ = simple_pr2_context
    bowl_world = STLParser(BOWL_MESH).parse()
    with world.modify_world():
        world.merge_world_at_pose(
            bowl_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                2.4, 2.2, 1, reference_frame=world.root
            ),
        )
    annotation = Bowl(root=world.get_body_by_name("bowl.stl"))
    with world.modify_world():
        world.add_semantic_annotations([annotation])
    return world, robot, annotation


def test_transporting_a_bowl_grasps_it_at_its_rim(pr2_and_bowl):
    """
    An object that says where it may be grasped has to actually be grasped there.

    Naming a grasp explicitly is what a caller does for an object with nothing better to
    offer; doing it for a bowl silently puts the fingers back in its middle, which is
    the whole reason a bowl generates its own grasps.
    """
    world, robot, bowl = pr2_and_bowl
    context = Context(world, robot, sampling_seed=SAMPLING_SEED)
    context.evaluate_conditions = False
    transport = TransportAction.from_graspable_by_closest_grasps(
        bowl,
        Pose.from_xyz_rpy(5.0, 3.3, 0.75, reference_frame=world.root),
        context.robot.left_arm,
        context,
    )

    sequential([transport], context=context)

    first_pick_up = next(iter(context.query_backend.evaluate(transport.pick_up)))
    grasp_position = first_pick_up.pick_up.grasp.grasp_pose.to_np()[:3, 3]
    assert distances_to_surface(bowl, grasp_position[None, :])[0] < GRIPPABLE_DISTANCE
