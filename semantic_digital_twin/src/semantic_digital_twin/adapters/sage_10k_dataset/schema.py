from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Self

import tqdm
import trimesh
import trimesh.visual
from PIL import Image
from typing_extensions import Optional, Tuple

from krrood.utils import get_full_class_name


from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale, Box
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class HasXYZ(SubclassJSONSerializer):
    x: float
    y: float
    z: float

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Deserialize from JSON.
        """
        return cls(x=data["x"], y=data["y"], z=data["z"])


@dataclass
class Sage10kRotation(HasXYZ):
    """
    Rotations in the Sage 10k world.
    The format is roll(x), pitch (y), and yaw (z).
    They are given in degrees.
    """

    def as_rpy_in_radians(self) -> Tuple[float, float, float]:
        conversion_factor = math.pi / 180
        return (
            self.x * conversion_factor,
            self.y * conversion_factor,
            self.z * conversion_factor,
        )


@dataclass
class Sage10kPosition(HasXYZ): ...


@dataclass
class Sage10kSize(SubclassJSONSerializer):
    """
    The scale of an object in the Sage 10k world.
    """

    height: float
    """
    Scale in z?
    """

    length: float
    """
    Scale in x?
    """

    width: float
    """
    Scale in y?
    """

    @property
    def x(self) -> float:
        return self.length

    @property
    def y(self) -> float:
        return self.width

    @property
    def z(self) -> float:
        return self.height

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "height": self.height,
            "length": self.length,
            "width": self.width,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Deserialize from JSON.
        """
        return cls(height=data["height"], length=data["length"], width=data["width"])


@dataclass
class Sage10kPhysicallyBasedRendering(SubclassJSONSerializer):
    metallic: float
    roughness: float

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            **super().to_json(),
            "metallic": self.metallic,
            "roughness": self.roughness,
        }

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], **kwargs
    ) -> Sage10kPhysicallyBasedRendering:
        """
        Deserialize from JSON.
        """
        return cls(metallic=data["metallic"], roughness=data["roughness"])


@dataclass
class Sage10kWall(SubclassJSONSerializer):
    id: str
    start_point: Sage10kPosition
    end_point: Sage10kPosition
    material: str
    height: float
    thickness: float

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
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
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            start_point=Sage10kPosition._from_json(data["start_point"], **kwargs),
            end_point=Sage10kPosition._from_json(data["end_point"], **kwargs),
            material=data["material"],
            height=data["height"],
            thickness=data["thickness"],
        )


@dataclass
class Sage10kObject(SubclassJSONSerializer):
    id: str
    room_id: str
    type: str
    description: str
    source: str

    source_id: str
    """
    The prefix of the filenames in the objects folder that related to this object.
    """

    place_id: str
    place_guidance: str

    mass: float
    """
    The weight of the object in kilograms
    """

    position: Sage10kPosition
    """
    The position of the object (relative to the room?)
    """

    rotation: Sage10kRotation
    """
    The orientation of the object
    """

    dimensions: Sage10kSize
    """
    The scale of the object
    """

    pbr_parameters: Sage10kPhysicallyBasedRendering
    """
    Physical rendering parameters. Currently unused
    """

    def create_in_world(self, world: World, directory_path: Path, parent: Body) -> Body:
        ply_file = directory_path / "objects" / f"{self.source_id}.ply"
        texture_file = directory_path / "objects" / f"{self.source_id}_texture.png"
        body = Body(name=PrefixedName(self.id))

        # Define the pose for the object in the world
        # The sage_10k_dataset uses position (x, y, z) and rotation (x, y, z as RPY)
        root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
            self.position.x,
            self.position.y,
            self.position.z,
            *self.rotation.as_rpy_in_radians(),
            reference_frame=parent,
            child_frame=body,
        )

        # Load the mesh and texture using the stable Mesh.from_file method
        # It automatically handles trimesh loading and applies the texture
        mesh = Mesh.from_file(
            file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix(reference_frame=body),
            scale=Scale(
                self.dimensions.width, self.dimensions.height, self.dimensions.length
            ),
        )

        # Create a Body with the loaded mesh as both visual and collision geometry
        visual = ShapeCollection([mesh], reference_frame=body)
        collision = ShapeCollection([mesh], reference_frame=body)
        body.visual = visual
        body.collision = collision

        with world.modify_world():
            root_C_body = Connection6DoF.create_with_dofs(
                world=world,
                parent=parent,
                child=body,
                parent_T_connection_expression=root_T_body,
            )
            # Add the body to the world
            world.add_body(body)
            world.add_connection(root_C_body)
        return body

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
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
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            room_id=data["room_id"],
            type=data["type"],
            description=data["description"],
            source=data["source"],
            source_id=data["source_id"],
            place_id=data["place_id"],
            place_guidance=data["place_guidance"],
            mass=data["mass"],
            position=Sage10kPosition._from_json(data["position"], **kwargs),
            rotation=Sage10kRotation._from_json(data["rotation"], **kwargs),
            dimensions=Sage10kSize._from_json(data["dimensions"], **kwargs),
            pbr_parameters=Sage10kPhysicallyBasedRendering._from_json(
                data["pbr_parameters"], **kwargs
            ),
        )


@dataclass
class Sage10kRoom(SubclassJSONSerializer):
    id: str

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
    The position of the rooms center? in the scene.
    """

    floor_material: str
    """
    The floor materials filename found in the materials folder.
    """

    objects: List[Sage10kObject] = field(default_factory=list)
    walls: List[Sage10kWall] = field(default_factory=list)

    def _create_floor(self, world: World, directory_path: Path, parent: Body):
        # create the floor
        floor_body = Body(name=PrefixedName(name="floor", prefix=self.id))
        floor_mesh = Box(
            scale=Scale(x=self.dimensions.x, y=self.dimensions.y, z=0.01)
        ).mesh
        floor_geometry = ShapeCollection(
            [
                Mesh.from_trimesh(
                    origin=HomogeneousTransformationMatrix(reference_frame=floor_body),
                    mesh=floor_mesh,
                    texture_file_path=str(
                        directory_path / f"{self.floor_material}.png"
                    ),
                )
            ],
            reference_frame=floor_body,
        )
        floor_body.collision = floor_geometry
        floor_body.visual = floor_geometry

        floor_annotation = Floor()

        with world.modify_world():
            parent_C_floor = FixedConnection(
                parent=parent,
                child=floor_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=self.position.x, y=self.position.y, z=self.position.z
                ),
            )
            world.add_body(floor_body)
            world.add_connection(parent_C_floor)
        return floor_body

    def create_in_world(self, world: World, directory_path: Path, parent: Body) -> Body:
        floor = self._create_floor(world, directory_path, parent)

        # create the objects
        for sage_object in tqdm.tqdm(self.objects, desc=f"Parsing objects"):
            sage_object.create_in_world(world, directory_path, parent=world.root)

        return world.root

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
        return {
            JSON_TYPE_NAME: get_full_class_name(self.__class__),
            "id": self.id,
            "room_type": self.room_type,
            "dimensions": to_json(self.dimensions),
            "position": to_json(self.position),
            "floor_material": self.floor_material,
            "objects": to_json(self.objects),
            "walls": to_json(self.walls),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Sage10kRoom:
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            room_type=data["room_type"],
            dimensions=Sage10kSize._from_json(data["dimensions"], **kwargs),
            position=Sage10kPosition._from_json(data["position"], **kwargs),
            floor_material=data["floor_material"],
            objects=[Sage10kObject._from_json(d, **kwargs) for d in data["objects"]],
            walls=[Sage10kWall._from_json(w, **kwargs) for w in data["walls"]],
        )


@dataclass
class Sage10kDoor(SubclassJSONSerializer):
    id: str
    wall_id: str
    position_on_wall: float
    width: float
    height: float
    door_type: str
    opens_inward: bool
    opening: bool
    door_material: str

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
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
        """
        Deserialize from JSON.
        """
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


@dataclass
class Sage10kScene(SubclassJSONSerializer):
    id: str
    """
    The id of the scene.
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
    """

    total_area: float
    """
    The total area of the scene in square meters.
    """

    rooms: List[Sage10kRoom] = field(default_factory=list)
    """
    The rooms of the scene.
    """

    directory_path: Optional[Path] = None
    """
    The directory path of the scenes json file.
    Usually named like `layout*.json`.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize to JSON.
        """
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
        """
        Deserialize from JSON.
        """
        return cls(
            id=data["id"],
            building_style=data["building_style"],
            description=data["description"],
            created_from_text=data["created_from_text"],
            total_area=data["total_area"],
            rooms=[Sage10kRoom._from_json(r, **kwargs) for r in data["rooms"]],
        )

    def create_world(self) -> World:
        if self.directory_path is None:
            raise ValueError("Directory path is not set.")

        world = World()

        root = Body(name=PrefixedName(name="root"))

        with world.modify_world():
            world.add_body(root)

        for room in self.rooms:
            room_body = room.create_in_world(
                world=world, directory_path=self.directory_path, parent=root
            )
        return world
