from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_digital_twin.api import RevoluteConnectionSpecification
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Floor,
    Handle,
    Wall,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass
class SingleDoorRoom:
    """
    World-independent description of a room: a floor with four walls, one of which
    carries a hinged door with a handle, whose entry way cuts a doorway into that wall.
    """

    scale: Scale = field(default_factory=lambda: Scale(2, 2, 2))
    """
    Floor extents and wall height.
    """

    wall_thickness: float = 0.05
    """
    Thickness of each wall.
    """

    door_scale: Scale = field(default_factory=lambda: Scale(0.05, 1, 1.8))
    """
    Extents of the door mounted into the room's last wall.
    """

    door_color: Color = field(default_factory=lambda: Color.RED())
    """
    Color of the door to to visually distinguish it from the wall
    """

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Floor:
        """
        Materialize this room in ``world``: a floor, four walls, and a hinged door with
        a handle mounted into the last wall, cutting a doorway into it.

        :param world: The world the room is added to.
        :param name: Prefix for the names of every entity this room creates.
        :param parent: The entity the room's floor attaches to. If None, ``world.root``
            is used.
        :param parent_T_self: Placement of the room's floor in ``parent``'s frame.
            Identity if None.
        :return: The room's floor annotation.
        """
        floor = Floor.get_annotation_specification(
            f"{name}_floor", Floor.get_default_root_kinematic_structure_entity_specification(scale=self.scale.xy)
        ).spawn(world, parent=parent, parent_T_self=parent_T_self)

        wall = None
        for i in range(4):
            yaw = (np.pi / 2) * i
            wall_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                (self.scale.x / 2) * np.cos(yaw),
                (self.scale.y / 2) * np.sin(yaw),
                0,
                0,
                0,
                yaw,
            )
            wall = Wall.get_annotation_specification(
                f"{name}_wall_{i}",
                Wall.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(self.wall_thickness, self.scale.x, self.scale.z)
                ),
            ).spawn(world, parent=floor.root, parent_T_self=wall_pose)

        handle_root_specification = Handle.get_default_root_kinematic_structure_entity_specification()
        handle_root_specification.parent_T_self = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                y=self.door_scale.y / 2 * 0.9, yaw=np.pi
            )
        )
        door = Door.get_annotation_specification(
            f"{name}_door",
            Door.get_default_root_kinematic_structure_entity_specification(
                scale=self.door_scale,
                connection_specification=RevoluteConnectionSpecification(
                    axis=Vector3.Z()
                ),
            ),
            part_specifications={
                "handle": Handle.get_annotation_specification(
                    f"{name}_handle", handle_root_specification
                )
            },
        ).spawn(
            world,
            parent=wall.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.door_scale.z / 2
            ),
        )
        door.root.collision.shapes[0].color = self.door_color
        with world.modify_world():
            wall.add(door.entry_way)

        return floor


@dataclass
class BuildingFloor:
    """
    World-independent description of a building floor: a floor slab carrying a room in
    each of its four corners.
    """

    floor_scale: Scale = field(default_factory=lambda: Scale(8, 8, 0))
    """
    Extents of the floor slab.
    """

    room: SingleDoorRoom = field(default_factory=SingleDoorRoom)
    """
    The room spawned into each of the floor's four corners.
    """

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Floor:
        """
        Materialize this building floor in ``world``: a floor slab with a room in each
        corner.

        :param world: The world the building floor is added to.
        :param name: Prefix for the names of every entity this building floor creates.
        :param parent: The entity the floor slab attaches to. If None, ``world.root``
            is used.
        :param parent_T_self: Placement of the floor slab in ``parent``'s frame.
            Identity if None.
        :return: The floor slab's annotation.
        """
        floor = Floor.get_annotation_specification(
            f"{name}_floor",
            Floor.get_default_root_kinematic_structure_entity_specification(scale=self.floor_scale),
        ).spawn(world, parent=parent, parent_T_self=parent_T_self)

        room_poses = [
            HomogeneousTransformationMatrix.from_xyz_rpy(-1.0, 2.5, 0),
            HomogeneousTransformationMatrix.from_xyz_rpy(-1.0, -2.5, 0, yaw=np.pi),
            HomogeneousTransformationMatrix.from_xyz_rpy(1.0, 2.5, 0),
            HomogeneousTransformationMatrix.from_xyz_rpy(1.0, -2.5, 0, yaw=np.pi),
        ]
        for i, room_pose in enumerate(room_poses):
            self.room.spawn(
                world, f"{name}_room_{i}", parent=floor.root, parent_T_self=room_pose
            )

        return floor
