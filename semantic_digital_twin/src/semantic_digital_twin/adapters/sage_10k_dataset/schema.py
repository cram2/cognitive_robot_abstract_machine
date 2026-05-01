from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Self

import numpy as np
from typing_extensions import Optional, Tuple, assert_never

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json
from krrood.ripple_down_rules.utils import recursive_subclasses
from krrood.utils import get_full_class_name
from semantic_digital_twin.adapters.sage_10k_dataset.semantic_annotations import (
    Sage10kTypeNameCleaner,
    NaturalLanguageDescriptionWithTypeDescription,
    RoomWithWallsAndDoors,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Wall,
    Door,
    Handle,
    Hinge,
    Room,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale, Box
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    WorldEntity,
    KinematicStructureEntity,
)


@dataclass
class Sage10kBase(SubclassJSONSerializer):
    """
    Base class for all classes of the Sage 10k dataset.

    These objects are serialized with JSON in a layout*.json and behave a bit different than the SubClassJSONSerializer
    of KRROOD asserts. The data of Sage10k does not support polymorphism and hence does not give any column that
    specifies their type. The JSON interface therefore just hard codes the type the referenced data should have.
    Use with care.

    .. important::

        All subclasses of Sage10kBase are only to load the data from the Sage 10k dataset.
        Do not use them for anything else.
    """


@dataclass
class Sage10kWithID(Sage10kBase):
    """
    Base class for Sage10k classes that have an id.
    """

    id: str
    """
    Unique identifier used to reference this object in many-to-one like relationships.
    """

    def create_in_world(
        self,
        world: World,
        directory: Path,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> WorldEntity:
        """
        Create the object in the world by getting its geometry from the provided information.
        Spawn bodies, regions, connections, and semantic annotations.

        :param world: The world to create the instances in.
        :param directory: The directory where the `layout*.json` and all its referenced files are found.
        :param parent: The parent of the newly created entities

        :return: The relevant created body
        """


@dataclass
class HasXYZ(Sage10kBase):
    """
    Parent for Sage10ks position and orientation to deduplicate the code.
    """

    x: float
    y: float
    z: float

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(x=data["x"], y=data["y"], z=data["z"])


@dataclass
class Sage10kRotation(HasXYZ):
    """
    Rotations in the Sage 10k world.
    The format is roll(x), pitch (y), and yaw (z).
    They are given in degrees.
    """

    def as_roll_pitch_yaw_in_radians(self) -> Tuple[float, float, float]:
        conversion_factor = math.pi / 180
        return (
            self.x * conversion_factor,
            self.y * conversion_factor,
            self.z * conversion_factor,
        )


@dataclass
class Sage10kPosition(HasXYZ):
    """
    Position of an entity in a Sage10k scene.
    It seems to always be global
    """


@dataclass
class Sage10kSize(Sage10kBase):
    """
    The scale of an object.
    """

    height: float
    """
    Scale in z
    """

    length: float
    """
    Scale in y
    """

    width: float
    """
    Scale in x
    """

    @property
    def x(self) -> float:
        return self.width

    @property
    def y(self) -> float:
        return self.length

    @property
    def z(self) -> float:
        return self.height

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "height": self.height,
            "length": self.length,
            "width": self.width,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(height=data["height"], length=data["length"], width=data["width"])


@dataclass
class Sage10kPhysicallyBasedRendering(SubclassJSONSerializer):
    """
    Parameters for super realistic renderers.
    Currently, we have no use of this in CRAM, but the information is provided by the dataset anyway.
    This data is ignored when `Sage10kScene.create_world` is called but parsed from the JSON information.
    """

    metallic: float
    roughness: float

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "metallic": self.metallic,
            "roughness": self.roughness,
        }

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], **kwargs
    ) -> Sage10kPhysicallyBasedRendering:
        return cls(metallic=data["metallic"], roughness=data["roughness"])


@dataclass
class Sage10kWall(Sage10kWithID):
    """
    Description of a wall for a room.
    """

    start_point: Sage10kPosition
    """
    The start point of the wall.
    Only x and y matter.
    """

    end_point: Sage10kPosition
    """
    The end point of the wall.
    Only x and y matter.
    """

    material: str
    """
    The wall materials filename found in the `materials` folder.
    """

    height: float
    """
    The height of the wall
    """

    thickness: float
    """
    The thickness of the wall
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "start_point": to_json(self.start_point),
            "end_point": to_json(self.end_point),
            "material": self.material,
            "height": self.height,
            "thickness": self.thickness,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kWall:
        return cls(
            id=data["id"],
            start_point=Sage10kPosition._from_json(data["start_point"], **kwargs),
            end_point=Sage10kPosition._from_json(data["end_point"], **kwargs),
            material=data["material"],
            height=data["height"],
            thickness=data["thickness"],
        )

    @property
    def wall_length_and_yaw(self) -> Tuple[float, float]:
        """
        :return: The length of the wall and the yaw that can be used for creating it with
        `Wall.create_with_new_body_in_world`.
        """
        # the wall length is given by x
        if self.start_point.x != self.end_point.x:
            wall_length = self.end_point.x - self.start_point.x
            yaw = math.pi / 2
        # the wall length is given by y
        elif self.start_point.y != self.end_point.y:
            wall_length = self.end_point.y - self.start_point.y
            yaw = 0
        else:
            assert_never(self)
        return wall_length, yaw

    def create_in_world(self, world: World, directory: Path, parent: Body) -> Wall:
        wall_name = PrefixedName(name=self.id)

        wall_length, yaw = self.wall_length_and_yaw

        wall_scale = Scale(x=self.thickness, y=wall_length, z=self.height)

        center_x = (self.end_point.x + self.start_point.x) / 2
        center_y = (self.end_point.y + self.start_point.y) / 2

        parent_T_wall = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=center_x,
            y=center_y,
            z=0.0,
            yaw=yaw,
            reference_frame=parent,
        )

        with world.modify_world():
            annotation = Wall.create_with_new_body_in_world(
                name=wall_name,
                scale=wall_scale,
                world=world,
                world_root_T_self=parent_T_wall,
            )

        body = annotation.root

        wall_mesh = body.collision.combined_mesh

        wall_mesh = Mesh.project_texture_coordinates(
            mesh=wall_mesh,
            projection_axis=np.array([1, 0, 0]),
            scale=np.array([self.thickness, wall_length, self.height]),
        )

        wall_length, _ = self.wall_length_and_yaw

        geometry_with_texture = ShapeCollection(
            [
                Mesh.from_trimesh(
                    origin=HomogeneousTransformationMatrix(reference_frame=body),
                    mesh=wall_mesh,
                    texture_file_path=str(
                        directory / "materials" / f"{self.material}.png"
                    ),
                )
            ],
            reference_frame=body,
        )
        body.collision = geometry_with_texture
        body.visual = geometry_with_texture

        return annotation


@dataclass
class Sage10kObject(Sage10kWithID):
    """
    Like a Body, but from Sage10k.
    """

    room_id: str
    """
    The room id where this object occurs in.
    """

    type: str
    """
    The type of the object as one word.
    """

    description: str
    """
    A textual description of the object.
    """

    source: str
    """
    Always generation
    """

    source_id: str
    """
    The prefix of the filenames in the objects folder that related to this object.
    """

    place_id: str
    """
    Either the id of the room, wall or floor.
    """

    place_guidance: str
    """
    A textual description of the place where the object is located.
    """

    mass: float
    """
    The weight of the object in kilograms
    """

    position: Sage10kPosition
    """
    The global position of the object
    """

    rotation: Sage10kRotation
    """
    The orientation of the object
    """

    dimensions: Sage10kSize
    """
    The scale of the object.
    This seems to be already incorporated in the meshes themselves, so dont use it.
    """

    pbr_parameters: Sage10kPhysicallyBasedRendering
    """
    Physical rendering parameters. Currently unused
    """

    def create_in_world(
        self,
        world: World,
        directory: Path,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> Body:
        ply_file = directory / "objects" / f"{self.source_id}.ply"
        texture_file = directory / "objects" / f"{self.source_id}_texture.png"
        body = Body()
        body.name = PrefixedName(name=str(body.id), prefix=self.id)

        # Define the pose for the object in the world
        root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
            self.position.x,
            self.position.y,
            self.position.z,
            *self.rotation.as_roll_pitch_yaw_in_radians(),
            reference_frame=parent,
            child_frame=body,
        )

        # Load the mesh and texture
        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
        )

        # Create a Body with the loaded mesh as both visual and collision geometry
        visual = ShapeCollection([mesh], reference_frame=body)
        collision = ShapeCollection([mesh], reference_frame=body)
        body.visual = visual
        body.collision = collision

        if self.place_id in ["floor", "wall"]:
            connection_type = FixedConnection
        else:
            connection_type = Connection6DoF

        with world.modify_world():
            root_C_body = connection_type.create_with_dofs(
                world=world,
                parent=parent,
                child=body,
                parent_T_connection_expression=root_T_body,
            )
            # Add the body to the world
            world.add_body(body)
            world.add_connection(root_C_body)

        # create semantic annotation
        annotation = NaturalLanguageDescriptionWithTypeDescription(
            root=body, description=self.description, type_description=self.type
        )

        with world.modify_world():
            world.add_semantic_annotation(annotation)

        return body

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room_id": self.room_id,
            "type": self.type,
            "description": self.description,
            "source": self.source,
            "source_id": self.source_id,
            "place_id": self.place_id,
            "place_guidance": self.place_guidance,
            "mass": self.mass,
            "position": to_json(self.position),
            "rotation": to_json(self.rotation),
            "dimensions": to_json(self.dimensions),
            "pbr_parameters": to_json(self.pbr_parameters),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kObject:
        place_guidance = data["place_guidance"]
        if isinstance(place_guidance, dict):
            import json

            place_guidance = json.dumps(place_guidance)

        return cls(
            id=data["id"],
            room_id=data["room_id"],
            type=data["type"],
            description=data["description"],
            source=data["source"],
            source_id=data["source_id"],
            place_id=data["place_id"],
            place_guidance=place_guidance,
            mass=data["mass"],
            position=Sage10kPosition._from_json(data["position"], **kwargs),
            rotation=Sage10kRotation._from_json(data["rotation"], **kwargs),
            dimensions=Sage10kSize._from_json(data["dimensions"], **kwargs),
            pbr_parameters=Sage10kPhysicallyBasedRendering._from_json(
                data["pbr_parameters"], **kwargs
            ),
        )


@dataclass
class Sage10kDoor(Sage10kWithID):
    """
    A door of a wall in Sage10k.
    """

    wall_id: str
    """
    Id of the wall where the door should be created on.
    """

    position_on_wall: float
    """
    Position on wall w. r. t. its starting point as percentage of the wall length.
    """

    width: float
    """
    Width of the door in meters.
    """

    height: float
    """
    Height of the door in meters.
    """

    door_type: str
    """
    Type of the door.
    """

    opens_inward: bool
    """
    Rather it opens to the inside of the room or the outside.
    """

    opening: bool
    """
    No idea
    """

    door_material: str
    """
    The door materials filename found in the `materials` folder.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "wall_id": self.wall_id,
            "position_on_wall": self.position_on_wall,
            "width": self.width,
            "height": self.height,
            "door_type": self.door_type,
            "opens_inward": self.opens_inward,
            "opening": self.opening,
            "door_material": self.door_material,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kDoor:
        return cls(
            id=data["id"],
            wall_id=data["wall_id"],
            position_on_wall=data["position_on_wall"],
            width=data["width"],
            height=data["height"],
            door_type=data["door_type"],
            opens_inward=data["opens_inward"],
            opening=data["opening"],
            door_material=data["door_material"],
        )

    def create_in_world(
        self,
        world: World,
        directory: Path,
        parent: KinematicStructureEntity,
        sage_10k_wall: Sage10kWall,
        wall_annotation: Wall,
        **kwargs,
    ) -> Door:
        """
        The parent must always be the wall body.

        :param sage_10k_wall: The sage 10k wall that is referenced by `self.wall_id`.
        :param wall_annotation: The wall annotation created in `world` before this call.
        """
        name = PrefixedName(name=self.id, prefix=sage_10k_wall.id)

        scale = Scale(x=sage_10k_wall.thickness, y=self.width, z=self.height)

        wall_length, _ = sage_10k_wall.wall_length_and_yaw

        parent_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=-wall_length / 2 + (self.position_on_wall * wall_length),
            z=self.height / 2,
            reference_frame=parent,
        )
        world_root_T_self = world.transform(parent_T_body, world.root)

        with world.modify_world():
            annotation = Door.create_with_new_body_in_world(
                name=name,
                scale=scale,
                world=world,
                world_root_T_self=world_root_T_self,
            )

        body = annotation.root
        door_mesh = body.collision.combined_mesh

        door_mesh = Mesh.project_texture_coordinates(
            mesh=door_mesh,
            projection_axis=np.array([1, 0, 0]),
            scale=np.array([sage_10k_wall.thickness, self.width, self.height]),
        )

        geometry_with_texture = ShapeCollection(
            [
                Mesh.from_trimesh(
                    origin=HomogeneousTransformationMatrix(reference_frame=body),
                    mesh=door_mesh,
                    texture_file_path=str(
                        directory / "materials" / f"{self.door_material}_texture.png"
                    ),
                )
            ],
            reference_frame=body,
        )
        body.collision = geometry_with_texture
        body.visual = geometry_with_texture

        with world.modify_world():
            wall_annotation.add_aperture(annotation.entry_way)

        self._create_handle_in_world(world, annotation)
        self._create_hinge_in_world(world, annotation)
        return annotation

    def _create_handle_in_world(self, world: World, door: Door) -> Handle:
        """
        Create the handle of the door.

        :param world: The world where the handle is created.
        :param door: The door to create the handle for.
        :return: The handle of the door.
        """
        door_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=0.1 if self.opens_inward else -0.1,
            reference_frame=door.root,
        )
        world_root_T_handle = world.transform(door_T_handle, world.root)
        handle_name = PrefixedName(name=f"{self.id}_handle", prefix=self.id)

        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=handle_name,
                world=world,
                world_root_T_self=world_root_T_handle,
                scale=Scale(x=0.1, y=0.1, z=0.05),
            )
            door.add_handle(handle)
        return handle

    def _create_hinge_in_world(self, world: World, door: Door) -> Hinge:
        """
        Create the hinge (the joint that makes the door openable) of the door.
        :param world: The world where the hinge is created.
        :param door: The door to create the hinge for.
        :return: The hinge
        """
        world_root_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())

        if self.opens_inward:
            lower = DerivativeMap(position=0.0)
            upper = DerivativeMap(position=np.pi / 2)
        else:
            upper = DerivativeMap(position=0.0)
            lower = DerivativeMap(position=-np.pi / 2)

        with world.modify_world():
            hinge = Hinge.create_with_new_body_in_world(
                name=PrefixedName(name="hinge", prefix=door.root.name.name),
                world=world,
                active_axis=Vector3.Z(),
                world_root_T_self=world_root_T_hinge,
                connection_limits=DegreeOfFreedomLimits(lower=lower, upper=upper),
            )
            door.add_hinge(hinge)

        return hinge


@dataclass
class Sage10kRoom(Sage10kWithID):
    """
    A room of the Sage10k dataset.
    """

    room_type: str
    """
    The type of the room.
    """

    dimensions: Sage10kSize
    """
    The scale of the room.
    """

    position: Sage10kPosition
    """
    The position of the rooms lower left corner? in the scene.
    """

    floor_material: str
    """
    The floor materials filename found in the `materials` folder.
    """

    objects: List[Sage10kObject] = field(default_factory=list)
    """
    Objects found in this room.
    """

    walls: List[Sage10kWall] = field(default_factory=list)
    """
    Walls of this room.
    """

    doors: List[Sage10kDoor] = field(default_factory=list)
    """
    The doors of the room
    """

    def _create_floor(
        self, world: World, directory: Path, parent: KinematicStructureEntity
    ) -> Floor:
        """
        Create the floor of this room.

        :param world: The world to create the floor in.
        :param directory: The directory of this scene
        :param parent: The parent kinematic structure entity.
        :return: The annotation of the created floor.
        """
        # create the floor
        floor_name = PrefixedName(name="floor", prefix=self.id)
        floor_mesh = Box(
            scale=Scale(x=self.dimensions.x, y=self.dimensions.y, z=0.01)
        ).mesh

        floor_mesh = Mesh.project_texture_coordinates(
            mesh=floor_mesh,
            projection_axis=np.array([0, 0, 1]),
            scale=np.array([self.dimensions.x, self.dimensions.y, 0.01]),
        )

        # convert position from lower left to center point
        x_center = self.position.x + (self.dimensions.x / 2)
        y_center = self.position.y + (self.dimensions.y / 2)

        parent_T_floor = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x_center,
            y=y_center,
            z=self.position.z,
            reference_frame=parent,
        )

        with world.modify_world():
            floor_annotation = Floor.create_with_new_body_in_world(
                scale=Scale(x=self.dimensions.x, y=self.dimensions.y, z=0.01),
                world=world,
                name=floor_name,
                world_root_T_self=parent_T_floor,
            )

        floor_body = floor_annotation.root

        floor_geometry_with_texture = ShapeCollection(
            [
                Mesh.from_trimesh(
                    origin=HomogeneousTransformationMatrix(reference_frame=floor_body),
                    mesh=floor_mesh,
                    texture_file_path=str(
                        directory / "materials" / f"{self.floor_material}.png"
                    ),
                )
            ],
            reference_frame=floor_body,
        )
        floor_body.collision = floor_geometry_with_texture
        floor_body.visual = floor_geometry_with_texture

        return floor_annotation

    def create_in_world(
        self,
        world: World,
        directory: Path,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> Body:
        floor_annotation = self._create_floor(world, directory, parent)

        walls_of_room = []
        doors_of_room = []

        for wall in self.walls:
            wall_annotation = wall.create_in_world(world, directory, parent)
            walls_of_room.append(wall_annotation)
            doors_of_this_wall = [
                door for door in self.doors if door.wall_id == wall.id
            ]  # join doors on this wall

            # create doors
            doors_of_room += [
                door.create_in_world(
                    world, directory, wall_annotation.root, wall, wall_annotation
                )
                for door in doors_of_this_wall
            ]

            # After all doors are added and the mesh is modified, re-project UVs and set texture
            wall_length, _ = wall.wall_length_and_yaw
            body = wall_annotation.root
            wall_mesh = body.collision.combined_mesh

            wall_mesh = Mesh.project_texture_coordinates(
                mesh=wall_mesh,
                projection_axis=np.array([1, 0, 0]),
                scale=np.array([wall.thickness, wall_length, wall.height]),
            )

            geometry_with_texture = ShapeCollection(
                [
                    Mesh.from_trimesh(
                        origin=HomogeneousTransformationMatrix(reference_frame=body),
                        mesh=wall_mesh,
                        texture_file_path=str(
                            directory / "materials" / f"{wall.material}.png"
                        ),
                    )
                ],
                reference_frame=body,
            )
            body.collision = geometry_with_texture
            body.visual = geometry_with_texture

        room_annotation = RoomWithWallsAndDoors(
            floor=floor_annotation,
            walls=walls_of_room,
            doors=doors_of_room,
            room_type=self.room_type,
        )

        with world.modify_world():
            world.add_annotation(room_annotation)

        # create the objects
        for sage_object in self.objects:
            sage_object.create_in_world(world, directory, parent=parent)

        return world.root

    def to_json(self) -> Dict[str, Any]:
        return {
            JSON_TYPE_NAME: get_full_class_name(self.__class__),
            "id": self.id,
            "room_type": self.room_type,
            "dimensions": to_json(self.dimensions),
            "position": to_json(self.position),
            "floor_material": self.floor_material,
            "objects": to_json(self.objects),
            "walls": to_json(self.walls),
            "doors": to_json(self.doors),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kRoom:
        return cls(
            id=data["id"],
            room_type=data["room_type"],
            dimensions=Sage10kSize._from_json(data["dimensions"], **kwargs),
            position=Sage10kPosition._from_json(data["position"], **kwargs),
            floor_material=data["floor_material"],
            objects=[Sage10kObject._from_json(d, **kwargs) for d in data["objects"]],
            walls=[Sage10kWall._from_json(w, **kwargs) for w in data["walls"]],
            doors=[Sage10kDoor._from_json(d, **kwargs) for d in data["doors"]],
        )


@dataclass
class Sage10kScene(Sage10kWithID):
    """
    An entire scene from Sage10k.
    """

    building_style: str
    """
    A textual description of the building style.
    """

    description: str
    """
    A textual description of the scene.
    """

    created_from_text: str
    """
    I think this is the entire prompt that was used to generate the scene.
    Usually contains just the descriptiom + 'Complete layout with doors/windows:'
    """

    total_area: float
    """
    The total area of the scene in square meters.
    """

    rooms: List[Sage10kRoom] = field(default_factory=list)
    """
    The rooms of the scene.
    """

    directory: Optional[Path] = None
    """
    The directory of the scenes json file.
    The layout files are named like `layout*.json`.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "building_style": self.building_style,
            "description": self.description,
            "created_from_text": self.created_from_text,
            "total_area": self.total_area,
            "rooms": to_json(self.rooms),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kScene:
        return cls(
            id=data["id"],
            building_style=data["building_style"],
            description=data["description"],
            created_from_text=data["created_from_text"],
            total_area=data["total_area"],
            rooms=[Sage10kRoom._from_json(r, **kwargs) for r in data["rooms"]],
        )

    def create_world(self) -> World:
        """
        :return: The semantically annotated world.
        """
        world = World()

        root = Body(name=PrefixedName(name="map"))

        with world.modify_world():
            world.add_body(root)

        for room in self.rooms:
            room.create_in_world(
                world=world,
                directory=self.directory,
                parent=root,
            )

        return world
