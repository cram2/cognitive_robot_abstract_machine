import json
import logging
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Tuple, Union, Set, Optional, List, Any, Self

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from sqlalchemy.orm import Session

from ...datastructures.prefixed_name import PrefixedName
from ...orm.ormatic_interface import *
from ...semantic_annotations.factories import (
    DoorFactory,
    RoomFactory,
    WallFactory,
    HandleFactory,
    Direction,
    DoubleDoorFactory,
    HorizontalSemanticDirection,
    SemanticPositionDescription,
    VerticalSemanticDirection,
)
from ...spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from ...world import World
from ...world_description.connections import FixedConnection
from ...world_description.geometry import Scale
from ...world_description.world_entity import Body


@dataclass
class ProcthorDoor:
    """
    Processes a door dictionary from Procthor, extracting the door's hole polygon and computing its scale and
    transformation matrix relative to the parent wall's horizontal center.
    """

    door_dict: dict
    """
    Dictionary representing a door from Procthors' JSON format
    """

    parent_wall_width: float
    """
    Width of the parent wall, since we define the door relative to the wall's horizontal center.
    """

    thickness: float = 0.02
    """
    Thickness of the door, since the door dictionary only provides a 2d polygon.
    """

    name: PrefixedName = field(init=False)
    """
    Name of the door, constructed from the assetId and room numbers.
    """

    min_x: float = field(init=False)
    """
    Minimum x-coordinate of the door's hole polygon.
    """

    min_y: float = field(init=False)
    """
    Minimum y-coordinate of the door's hole polygon.
    """

    max_x: float = field(init=False)
    """
    Maximum x-coordinate of the door's hole polygon.
    """

    max_y: float = field(init=False)
    """
    Maximum y-coordinate of the door's hole polygon.
    """

    def __post_init__(self):
        """
        Extracts the hole polygon, and preprocesses the name and min/max coordinates of the door's hole polygon.
        """
        asset_id = self.door_dict["assetId"]
        room_numbers = self.door_dict["id"].split("|")[1:]

        self.name = PrefixedName(
            f"{asset_id}_room{room_numbers[0]}_room{room_numbers[1]}"
        )

        hole_polygon = self.door_dict["holePolygon"]

        x0, y0 = float(hole_polygon[0]["x"]), float(hole_polygon[0]["y"])
        x1, y1 = float(hole_polygon[1]["x"]), float(hole_polygon[1]["y"])

        self.x_min, self.x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        self.y_min, self.y_max = (y0, y1) if y0 <= y1 else (y1, y0)

    @cached_property
    def scale(self) -> Scale:
        """
        Computes the door scale from the door's hole polygon. Converts the scale from Unity's left-handed Y-up, Z-forward
        convention to the semantic digital twin's right-handed Z-up, X-forward convention.

        :return: Scale representing the door's geometry.
        """
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return Scale(self.thickness, width, height)

    @cached_property
    def wall_T_door(self) -> HomogeneousTransformationMatrix:
        """
        Computes the door position from the wall's horizontal center. Converts the Unity's left-handed Y-up, Z-forward
        convention to the semantic digital twin's right-handed Z-up, X-forward convention.

        :return: TransformationMatrix representing the door's transform from the wall's perspective.
        """
        # Door center origin expressed from the wall's horizontal center. Unity's wall origin is in one of the corners
        width_origin_wall_corner = 0.5 * (self.x_min + self.x_max)
        height_origin_center = 0.5 * (self.y_min + self.y_max)
        width_origin_center = width_origin_wall_corner - 0.5 * self.parent_wall_width

        # In unity, doors are defined as holes in the wall, so we express them as children of walls.
        # This means we just need to translate them, and can assume no rotation
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            Point3(0, -width_origin_center, height_origin_center)
        )

    def _get_double_door_factory(self) -> DoubleDoorFactory:
        """
        Parses the parameters according to the double door assumptions, and returns a double door factory.
        """
        one_door_scale = Scale(self.thickness, self.scale.y * 0.5, self.scale.z)
        x_direction: float = one_door_scale.x / 2
        y_direction: float = one_door_scale.y / 2
        handle_directions = [Direction.Y, Direction.NEGATIVE_Y]

        door_factories = []
        door_transforms = []

        for index, direction in enumerate(handle_directions):
            single_door_name = PrefixedName(
                f"{self.name.name}_{index}", self.name.prefix
            )

            horizontal_direction = (
                HorizontalSemanticDirection.RIGHT
                if direction == Direction.Y
                else HorizontalSemanticDirection.LEFT
            )
            semantic_position = SemanticPositionDescription(
                horizontal_direction_chain=[
                    horizontal_direction,
                    HorizontalSemanticDirection.FULLY_CENTER,
                ],
                vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
            )

            door_factory = self._get_single_door_factory(
                semantic_position, single_door_name, one_door_scale
            )

            door_factories.append(door_factory)

            parent_T_door = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=x_direction,
                y=(-y_direction) if direction == Direction.Y else y_direction,
            )
            door_transforms.append(parent_T_door)

        double_door_factory = DoubleDoorFactory(
            name=self.name,
            door_factories=door_factories,
            door_transforms=door_transforms,
        )
        return double_door_factory

    def _get_single_door_factory(
        self,
        semantic_position: SemanticPositionDescription,
        name: Optional[PrefixedName] = None,
        scale: Optional[Scale] = None,
    ) -> DoorFactory:
        """
        Parses the parameters according to the single door assumptions, and returns a single door factory.
        """
        name = self.name if name is None else name
        scale = self.scale if scale is None else scale
        handle_name = PrefixedName(f"{name.name}_handle", name.prefix)
        door_factory = DoorFactory(
            name=name,
            scale=scale,
            handle_factory=HandleFactory(name=handle_name),
            semantic_position=semantic_position,
        )
        return door_factory

    def get_factory(self) -> Union[DoorFactory, DoubleDoorFactory]:
        """
        Returns a Factory for the door, either a DoorFactory or a DoubleDoorFactory,
        depending on its name. If the door's name contains "double", it is treated as a double door.
        """

        if "double" in self.name.name.lower():
            return self._get_double_door_factory()
        else:
            semantic_position = SemanticPositionDescription(
                horizontal_direction_chain=[
                    HorizontalSemanticDirection.RIGHT,
                    HorizontalSemanticDirection.FULLY_CENTER,
                ],
                vertical_direction_chain=[VerticalSemanticDirection.FULLY_CENTER],
            )
            return self._get_single_door_factory(semantic_position=semantic_position)


@dataclass
class ProcthorWall:
    """
    Processes a wall dictionary from Procthor, extracting the wall's polygon and computing its scale and
    transformation matrix. Its center will be at the horizontal center of its polygon, at height 0.
     It also processes any doors associated with the wall, creating ProcthorDoor instances for each door.
    The wall is defined by two polygons, one for each side of the physical wall, and the door is defined as a hole in
    the wall polygon.
    """

    wall_dicts: List[dict] = field(default_factory=list)
    """
    List of dictionaries, where each dictionary represents one wall polygon in procthor
    """

    door_dicts: List[dict] = field(default_factory=list)
    """
    List of dictionaries, where each dictionary represents one door hole in the wall polygon
    """

    wall_thickness: float = 0.02
    """
    Thickness of the wall, since the wall dictionary only provides a 2d polygon.
    """

    name: PrefixedName = field(init=False)
    """
    Name of the wall, constructed from the corners of the wall polygon and the room numbers associated with the wall.
    """

    x_coords: List[float] = field(init=False)
    """
    List of unique X-coordinates of the wall polygon, extracted in order from the wall dictionary. 
    """

    y_coords: List[float] = field(init=False)
    """
    List of unique Y-coordinates of the wall polygon, extracted in order from the wall dictionary.
    """

    z_coords: List[float] = field(init=False)
    """
    List of unique Z-coordinates of the wall polygon, extracted in order from the wall dictionary.
    """

    delta_x: float = field(init=False)
    """
    Difference between the first and last X-coordinates of the wall polygon.
    """

    delta_z: float = field(init=False)
    """
    Difference between the first and last Z-coordinates of the wall polygon.
    """

    def __post_init__(self):
        """
        Processes the wall polygons and doors, extracting the min/max coordinates and computing the name of the wall.
        If no doors are present, it uses the first wall polygon as the reference for min/max coordinates.
        If doors are present, it uses the wall polygon that corresponds to the first door's 'wall0' reference.
         This is because the door hole is defined relative to that wall polygon and using the other wall would result
         in the hole being on the wrong side of the wall.
        """
        if self.door_dicts:
            used_wall = (
                self.wall_dicts[0]
                if self.wall_dicts[0]["id"] == self.door_dicts[0]["wall0"]
                else self.wall_dicts[1]
            )
        else:
            used_wall = self.wall_dicts[0]

        polygon = used_wall["polygon"]

        def unique_in_order(seq):
            return list(dict.fromkeys(seq))

        self.x_coords = unique_in_order(float(p["x"]) for p in polygon)
        self.y_coords = unique_in_order(float(p["y"]) for p in polygon)
        self.z_coords = unique_in_order(float(p["z"]) for p in polygon)

        # ProcTHOR wall polygons always have exactly four corners, and are perfectly vertical, so we have at
        # most two unique x and two unique z coordinates. If they line up perfectly, we may have only one unique
        # x or z coordinate, which is why we need to access -1 for generality.
        self.delta_x, self.delta_z = (
            self.x_coords[0] - self.x_coords[-1],
            self.z_coords[0] - self.z_coords[-1],
        )

        room_numbers = [w["id"].split("|")[1] for w in self.wall_dicts]
        corners = used_wall["id"].split("|")[2:]
        self.name = PrefixedName(
            f"wall_{corners[0]}_{corners[1]}_{corners[2]}_{corners[3]}_room{room_numbers[0]}_room{room_numbers[1]}"
        )

    @cached_property
    def scale(self) -> Scale:
        """
        Computes the wall scale from the first wall polygon. Converts the scale from Unity's left-handed Y-up, Z-forward
        convention to the semantic digital twin's right-handed Z-up, X-forward convention.

        :return: Scale representing the wall's geometry.
        """
        width = math.hypot(self.delta_x, self.delta_z)
        min_y, max_y = min(self.y_coords), max(self.y_coords)

        height = max_y - min_y

        return Scale(x=self.wall_thickness, y=width, z=height)

    @cached_property
    def world_T_wall(self) -> HomogeneousTransformationMatrix:
        """
        Computes the wall's world position matrix from the wall's x and z coordinates.
        Calculates the yaw angle using the atan2 function based on the wall's width and depth.
        The wall is artificially set to height=0, because
        1. as of now, procthor house floors have the same floor value at 0
        2. Since doors origins are in 3d center, positioning the door correctly at the floor given potentially varying
           wall heights is unnecessarily complex given the assumption stated in 1.
        """

        yaw = math.atan2(self.delta_z, -self.delta_x)
        x_center = (self.x_coords[0] + self.x_coords[-1]) * 0.5
        z_center = (self.z_coords[0] + self.z_coords[-1]) * 0.5

        world_T_wall = HomogeneousTransformationMatrix.from_xyz_rpy(
            x_center, 0, z_center, 0.0, yaw, 0
        )

        return unity_to_semantic_digital_twin_transform(world_T_wall)

    def get_world(self) -> World:
        """
        Returns a World instance with this wall at its root.
        """
        door_factories = []
        list_wall_T_door = []

        for door in self.door_dicts:
            door = ProcthorDoor(door_dict=door, parent_wall_width=self.scale.y)
            door_factories.append(door.get_factory())
            list_wall_T_door.append(door.wall_T_door)

        wall_factory = WallFactory(
            name=self.name,
            scale=self.scale,
            door_factories=door_factories,
            door_transforms=list_wall_T_door,
        )

        return wall_factory.create()


@dataclass
class ProcthorRoom:
    """
    Processes a room dictionary from Procthor, extracting the room's floor polygon and computing its center.
    """

    room_dict: dict
    """
    Dictionary representing a room from Procthor's JSON format.
    """

    name: PrefixedName = field(init=False)
    """
    Name of the room, constructed from the room type and room ID.
    """

    centered_polytope: List[Point3] = field(init=False)
    """
    Polytope representing the room's floor polygon, centered around its local 0, 0, 0 coordinate
    """

    def __post_init__(self):
        """
        Extracts the room's floor polygon, computes its center, and constructs the centered polytope.
        """
        room_polytope = self.room_dict["floorPolygon"]

        polytope_length = len(room_polytope)
        coords = ((v["x"], v["y"], v["z"]) for v in room_polytope)
        x_coords, y_coords, z_coords = zip(*coords)
        self.x_center = sum(x_coords) / polytope_length
        self.y_center = sum(y_coords) / polytope_length
        self.z_center = sum(z_coords) / polytope_length

        self.centered_polytope = [
            Point3(
                v["z"] - self.z_center,
                -(v["x"] - self.x_center),
                v["y"] - self.y_center,
            )
            for v in room_polytope
        ]

        room_id = self.room_dict["id"].split("|")[-1]
        self.name = PrefixedName(f"{self.room_dict['roomType']}_{room_id}")

    @cached_property
    def world_T_room(self) -> HomogeneousTransformationMatrix:
        """
        Computes the room's world transform
        """

        world_P_room = Point3(self.z_center, -self.x_center, self.y_center)

        return HomogeneousTransformationMatrix.from_point_rotation_matrix(world_P_room)

    def get_world(self) -> World:
        """
        Returns a World instance with this room as a Region at its root.
        """

        return RoomFactory(
            name=self.name, floor_polytope=self.centered_polytope
        ).create()


@dataclass
class ProcthorObject:
    """
    Processes an object dictionary from Procthor, extracting the object's position and rotation,
    and computing its world transformation matrix. It also handles the import of child objects recursively.
    """

    object_dict: dict
    """
    Dictionary representing an object from Procthor's JSON format.
    """

    session: Session
    """
    SQLAlchemy session to interact with the database to import objects.
    """

    @cached_property
    def world_T_obj(self) -> HomogeneousTransformationMatrix:
        """
        Computes the object's world transformation matrix from its position and rotation. Converts Unity's
        left-handed Y-up, Z-forward convention to the right-handed Z-up, X-forward convention.
        """
        obj_position = self.object_dict["position"]
        obj_rotation = self.object_dict["rotation"]
        world_T_obj = HomogeneousTransformationMatrix.from_xyz_rpy(
            obj_position["x"],
            obj_position["y"],
            obj_position["z"],
            math.radians(obj_rotation["x"]),
            math.radians(obj_rotation["y"]),
            math.radians(obj_rotation["z"]),
        )

        return HomogeneousTransformationMatrix(
            unity_to_semantic_digital_twin_transform(world_T_obj)
        )

    def get_world(self) -> Optional[World]:
        """
        Returns a World instance with this object at its root, importing it from the database using its assetId.
        If the object has children, they are imported recursively and connected to the parent object.
        If the object cannot be found in the database, it's children are skipped as well.
        """
        asset_id = self.object_dict["assetId"]
        body_world: World = get_world_by_asset_id(self.session, asset_id=asset_id)

        if body_world is None:
            logging.error(
                f"Could not find asset {asset_id} in the database. Skipping object and its children."
            )
            return None

        with body_world.modify_world():

            for child in self.object_dict.get("children", {}):
                child_object = ProcthorObject(child, self.session)
                world_T_child = child_object.world_T_obj
                child_world = child_object.get_world()
                if child_world is None:
                    continue
                obj_T_child = self.world_T_obj.inverse() @ world_T_child
                child_connection = FixedConnection(
                    parent=body_world.root,
                    child=child_world.root,
                    parent_T_connection_expression=obj_T_child,
                )
                body_world.merge_world(child_world, child_connection)

            return body_world


def unity_to_semantic_digital_twin_transform(
    unity_transform_matrix: HomogeneousTransformationMatrix,
) -> HomogeneousTransformationMatrix:
    """
    Convert a left-handed Y-up, Z-forward Unity transform to the right-handed Z-up, X-forward convention used in the
    semantic digital twin.

    :param unity_transform_matrix:  The transformation matrix in Unity coordinates.
    :return: TransformationMatrix in semantic digital twin coordinates.
    """

    unity_transform_matrix = unity_transform_matrix.to_np()

    permutation_matrix = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )

    reflection_vector = np.diag([1, -1, 1])
    R = reflection_vector @ permutation_matrix
    conjugation_matrix = np.eye(4)
    conjugation_matrix[:3, :3] = R
    inverse_conjugation_matrix = conjugation_matrix.T

    unity_transform_matrix = np.asarray(unity_transform_matrix, float).reshape(4, 4)

    return HomogeneousTransformationMatrix(
        data=conjugation_matrix @ unity_transform_matrix @ inverse_conjugation_matrix
    )


@dataclass
class ProcTHORParser:
    """
    Parses a Procthor JSON file into a semantic digital twin World.
    """

    name: str
    """
    The name of the world that is extracted from the house."""

    house: Dict[str, Any]
    """
    The house as JSON.
    """

    session: Optional[Session] = field(default=None)
    """
    SQLAlchemy session to interact with the database to import objects.
    """

    @classmethod
    def from_file(cls, file_path: str, session: Optional[Session] = None) -> Self:
        return cls(
            name=Path(file_path).stem, house=json.load(open(file_path)), session=session
        )

    def parse(self) -> World:
        """
        Parses a JSON file from procthor into a world.
        Room floor areas are constructed from the supplied polygons
        Walls and doors are constructed from the supplied polygons
        Objects are imported from the database
        """

        house_name = self.name
        world = World(name=house_name)
        with world.modify_world():
            world_root = Body(name=PrefixedName(house_name))
            world.add_kinematic_structure_entity(world_root)

            self.import_rooms(world, self.house["rooms"])

            if self.session is not None:
                self.import_objects(world, self.house["objects"])
            else:
                logging.warning("No database session provided, skipping object import.")

            self.import_walls_and_doors(world, self.house["walls"], self.house["doors"])

            return world

    @staticmethod
    def import_rooms(world: World, rooms: List[Dict]):
        """
        Imports rooms from the Procthor JSON file into ProcthorRoom instances.

        :param world: The World instance to which the rooms will be added.
        :param rooms: List of room dictionaries from the Procthor JSON file.
        """
        for room in rooms:
            procthor_room = ProcthorRoom(room_dict=room)
            room_world = procthor_room.get_world()
            room_connection = FixedConnection(
                parent=world.root,
                child=room_world.root,
                parent_T_connection_expression=procthor_room.world_T_room,
            )
            world.merge_world(room_world, room_connection)

    def import_objects(self, world: World, objects: List[Dict]):
        """
        Imports objects from the Procthor JSON file into ProcthorObject instances.

        :param world: The World instance to which the objects will be added.
        :param objects: List of object dictionaries from the Procthor JSON file.
        """
        for index, obj in enumerate(objects):
            procthor_object = ProcthorObject(object_dict=obj, session=self.session)
            obj_world = procthor_object.get_world()
            if obj_world is None:
                continue
            # for kse in obj_world.kinematic_structure_entities:
            #     kse.name.name += f"_{id(obj)}"
            obj_connection = FixedConnection(
                parent=world.root,
                child=obj_world.root,
                parent_T_connection_expression=procthor_object.world_T_obj,
            )
            world.merge_world(obj_world, obj_connection)

    def import_walls_and_doors(
        self, world: World, walls: List[Dict], doors: List[Dict]
    ):
        """
        Imports walls from the Procthor JSON file into ProcthorWall instances.

        :param world: The World instance to which the walls will be added.
        :param walls: List of wall dictionaries from the Procthor JSON file.
        :param doors: List of door dictionaries from the Procthor JSON file.
        """
        procthor_walls = self._build_procthor_walls(walls, doors)

        for procthor_wall in procthor_walls:
            wall_world = procthor_wall.get_world()
            wall_connection = FixedConnection(
                parent=world.root,
                child=wall_world.root,
                parent_T_connection_expression=procthor_wall.world_T_wall,
            )
            world.merge_world(wall_world, wall_connection)

    @staticmethod
    def _build_procthor_wall_from_polygon(
        walls: List[Dict],
    ) -> List[ProcthorWall]:
        """
        Groups walls by their polygon and creates ProcthorWall instances for each group.

        :param walls: List of walls without doors

        :return: List of ProcthorWall instances, each representing a pair of walls with the same polygon.
        :raises AssertionError: If the number of walls is not even, as we assume that walls are always paired.
        """

        assert len(walls) % 2 == 0, (
            f"Expected an even number of walls, but found {len(walls)}. "
            f"We assumed that this is never the case, this case may need to be handled now."
        )

        def _polygon_key(poly):
            return frozenset((p["x"], p["y"], p["z"]) for p in poly)

        groups = {}
        for wall in walls:
            key = _polygon_key(wall.get("polygon", []))
            groups.setdefault(key, []).append(wall)

        procthor_walls = [
            ProcthorWall(wall_dicts=matched_walls) for matched_walls in groups.values()
        ]

        return procthor_walls

    @staticmethod
    def _build_procthor_wall_from_door(
        walls: List[Dict], doors: List[Dict]
    ) -> Tuple[List[ProcthorWall], Set[str]]:
        """
        Builds ProcthorWall instances from the provided walls and doors, associating each door with its corresponding walls.

        :param walls: List of wall dictionaries
        :param doors: List of door dictionaries

        :returns: Tuple containing a list of ProcthorWall instances and a set of used wall IDs.
        :raises AssertionError: If a door does not have exactly two walls associated with it.
        """
        walls_by_id = {wall["id"]: wall for wall in walls}
        used_wall_ids = set()
        procthor_walls = []

        for door in doors:
            wall_ids = [door.get("wall0"), door.get("wall1")]
            found_walls = []
            for wall_id in wall_ids:
                wall = walls_by_id[wall_id]
                found_walls.append(wall)
                used_wall_ids.add(wall_id)

            assert (
                len(found_walls) == 2
            ), f"Door {door['id']} should have two walls, but found {len(found_walls)}."

            procthor_walls.append(
                ProcthorWall(door_dicts=[door], wall_dicts=found_walls)
            )

        return procthor_walls, used_wall_ids

    def _build_procthor_walls(
        self, walls: List[Dict], doors: List[Dict]
    ) -> List[ProcthorWall]:
        """
        Builds ProcthorWall instances from the provided walls and doors.

        :param doors: List of door dictionaries
        :param walls: List of wall dictionaries

        :returns: List of ProcthorWall
        """

        procthor_walls, used_wall_ids = self._build_procthor_wall_from_door(
            walls, doors
        )
        remaining_walls = [wall for wall in walls if wall["id"] not in used_wall_ids]
        paired_walls = self._build_procthor_wall_from_polygon(remaining_walls)

        procthor_walls.extend(paired_walls)

        return procthor_walls


def get_world_by_asset_id(session: Session, asset_id: str) -> Optional[World]:
    """
    Queries the database for a WorldMapping with the given asset_id provided by the procthor file.
    """
    asset_id = asset_id.lower()
    other_possible_name = "_".join(asset_id.split("_")[:-1])

    expr = select(WorldMappingDAO).where(WorldMappingDAO.name == asset_id)
    expr2 = select(WorldMappingDAO).where(WorldMappingDAO.name == other_possible_name)
    logging.info(f"Querying name: {asset_id}")
    try:
        world_mapping = session.scalars(expr).one()
    except (NoResultFound, MultipleResultsFound):
        try:
            logging.info(f"Querying name: {other_possible_name}")
            world_mapping = session.scalars(expr2).one()
        except (NoResultFound, MultipleResultsFound):
            world_mapping = None
            logging.warning(
                f"Could not find world with name {asset_id} or {other_possible_name}; Skipping."
            )

    return world_mapping.from_dao() if world_mapping else None
