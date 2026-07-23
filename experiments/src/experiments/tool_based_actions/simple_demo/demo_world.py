"""
Shared world and tool setup for the tool-based action demos (cutting, pouring, mixing,
wiping).
"""

import math
import os

from typing_extensions import Optional

import coraplex
from semantic_digital_twin.adapters.mesh import MeshParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager

OBJECTS_DIRECTORY = os.path.join(
    os.path.dirname(coraplex.__file__), "..", "..", "resources", "objects"
)

TARGET_POSITION_XYZ = (2.4, 2.2, 1.0)
"""
Position of the manipulated object on the apartment kitchen counter.
"""

BASE_POSITION_XYZ = (1.85, 2.2, 0.0)
"""
Base position in front of the kitchen counter, facing the target.
"""

CUT_MOUNT = {"z": 0.08, "pitch": -math.pi / 2}
"""
Knife mount transform on the right gripper's tool frame.
"""

MIX_MOUNT = {"z": -0.08, "pitch": math.pi / 2}
"""
Whisk mount transform on the right gripper's tool frame.
"""

POUR_MOUNT = {"z": -0.08}
"""
Cup mount transform on the right gripper's tool frame.
"""

WIPE_MOUNT = {"pitch": math.pi / 2}
"""
Sponge mount transform on the right gripper's tool frame.
"""

BREAD_COLOR = Color(0.55, 0.35, 0.17)
"""
Brownish color for the bread.
"""

CUP_COLOR = Color(0.3, 0.5, 0.9)
"""
Blueish color for the cup.
"""

BOWL_COLOR = Color(0.8, 0.2, 0.2)
"""
Redish color for the bowl.
"""

SPONGE_COLOR = Color(0.95, 0.85, 0.3)
"""
Yellowish color for the sponge.
"""


def parse_object(mesh_file_name: str, color: Optional[Color] = None) -> World:
    """
    :param mesh_file_name: Name of the mesh file in the demo resources, in any format
        :class:`~semantic_digital_twin.adapters.mesh.MeshParser` understands.
    :param color: Color to dye the mesh's visual shapes with. Keeps the mesh's own
        appearance if None.
    :return: A world containing the mesh from the demo resources.
    """
    object_world = MeshParser(os.path.join(OBJECTS_DIRECTORY, mesh_file_name)).parse()
    if color is not None:
        object_world.root.visual.dye_shapes(color)
    return object_world


def spawn_mesh_body(
    world: World,
    mesh_file_name: str,
    transform: HomogeneousTransformationMatrix,
    color: Optional[Color] = None,
    name: Optional[str] = None,
    scale: Optional[Scale] = None,
) -> Body:
    """
    Spawn a mesh from the demo resources into the world at a pose.

    :param world: The world to spawn into.
    :param mesh_file_name: Mesh file in the demo resources.
    :param transform: Pose of the spawned body in the world frame.
    :param color: Color the mesh's visual shapes are dyed with, or None to keep them.
    :param name: Name for the spawned body, or None to keep the mesh's own name.
    :param scale: Uniform scale applied to the mesh's shapes, or None to keep them.
    :return: The spawned body inside ``world``.
    """
    object_world = parse_object(mesh_file_name, color=color)
    if name is not None:
        object_world.root.name = PrefixedName(name)
    if scale is not None:
        for shape in object_world.root.visual.shapes:
            shape.scale = scale
        for shape in object_world.root.collision.shapes:
            shape.scale = scale
    spawned_body_name = object_world.root.name.name
    with world.modify_world():
        world.merge_world_at_pose(object_world, transform)
    return world.get_body_by_name(spawned_body_name)


def attach_sponge(world: World, robot: AbstractRobot, arm: Arms) -> Body:
    """
    Attach a primitive box sponge to the arm's tool frame.

    :return: The sponge body inside ``world``.
    """
    sponge_body = Body(
        name=PrefixedName("sponge"),
        collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
        visual=ShapeCollection(
            [Box(scale=Scale(0.05, 0.05, 0.05), color=SPONGE_COLOR)]
        ),
    )
    tool_frame = ViewManager.get_end_effector_view(arm, robot).tool_frame
    connection = FixedConnection(
        parent=tool_frame,
        child=sponge_body,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=tool_frame, **WIPE_MOUNT
        ),
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(sponge_body)
        world.add_connection(connection)
    return sponge_body
