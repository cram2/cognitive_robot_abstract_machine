import os

import numpy as np
import pytest
from trimesh.proximity import closest_point

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% fixtures

BOWL_MESH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "coraplex",
    "resources",
    "objects",
    "bowl.stl",
)
"""
The bowl the demos transport, an irregular scan rather than a turned shape.
"""

GRIPPABLE_DISTANCE = 0.005
"""
How far from the bowl's surface the fingers may close and still find material, in
meters.
"""


@pytest.fixture
def bowl() -> Bowl:
    world = STLParser(BOWL_MESH).parse()
    annotation = Bowl(root=world.get_body_by_name("bowl.stl"))
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    return annotation


def distances_to_surface(bowl: Bowl, positions: np.ndarray) -> np.ndarray:
    """
    :param bowl: The bowl whose surface to measure against.
    :param positions: Points in the bowl's own frame.
    :return: Each point's distance to the nearest point of the bowl's surface.
    """
    _, distances, _ = closest_point(bowl.root.combined_mesh, positions)
    return distances


# %% grasping the bowl


def test_bowl_grasps_close_on_the_bowls_wall(bowl):
    """
    The fingers must meet material. Grasping the bowl at its own origin -- which is
    what an object without grasp poses of its own offers -- closes them in mid air
    inside the bowl.
    """
    positions = np.array([pose.to_np()[:3, 3] for pose in bowl.grasp_poses()])

    assert len(positions) == bowl.grasp_pose_count
    assert np.all(distances_to_surface(bowl, positions) < GRIPPABLE_DISTANCE)


def test_the_bowls_origin_is_not_grippable(bowl):
    """
    Guards the premise of the test above: the origin really does sit in mid air, so
    grasping there is a bug rather than a matter of taste.
    """
    [distance] = distances_to_surface(bowl, np.zeros((1, 3)))

    assert distance > GRIPPABLE_DISTANCE


# %% transporting a bowl


@pytest.fixture
def pr2_and_bowl(mutable_simple_pr2_world):
    """
    A PR2 in a world with the demos' bowl standing on the counter.
    """
    world, robot, _ = mutable_simple_pr2_world
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
    context = Context(world, robot)
    context.evaluate_conditions = False
    transport = TransportAction(
        bowl,
        Pose.from_xyz_rpy(5.0, 3.3, 0.75, reference_frame=world.root),
        Arms.LEFT,
    )

    sequential([transport], context=context)
    transport._action_plan

    grasp_position = transport.grasp_pose.to_np()[:3, 3]
    assert distances_to_surface(bowl, grasp_position[None, :])[0] < GRIPPABLE_DISTANCE
