"""
World descriptor for RoboKudo.

This module provides classes for describing a world with predefined objects and regions.
It builds semantic_digital_twin entities into a shared World instance.
"""

from dataclasses import dataclass, field
from pathlib import Path

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Region,
    SemanticAnnotation,
)
from typing_extensions import Any, Dict, List, Optional


@dataclass
class ObjectSpec:
    """
    Data structure to define typical object properties for predefined objects.
    Will be consumed by BaseWorldDescriptor.build_objects to create a world
    and the corresponding connections.
    """

    name: str
    pose: HomogeneousTransformationMatrix
    mesh_path: Optional[Path] = None
    box_scale: Optional[Scale] = None
    center_mesh: bool = True
    color: Color = field(default_factory=lambda: Color(0.1, 0.2, 0.8, 1.0))


@dataclass
class RegionSpec:
    """
    Data structure to define typical properties for predefined regions.
    Will be consumed by BaseWorldDescriptor.build_regions to create
    regions and their connections.
    """

    name: str
    pose: HomogeneousTransformationMatrix
    box_scale: Scale
    color: Color = field(default_factory=Color)


@dataclass(eq=False)
class PredefinedObject(SemanticAnnotation):
    body: Body = field(default=None)


class BaseWorldDescriptor:
    """Base class for describing a world with predefined objects and regions."""

    def __init__(
        self,
        world: Optional[World] = None,
        root_name: str = "descriptor_world_root",
        root_prefix: Optional[str] = "world",
    ) -> None:
        """
        Initialize a world descriptor.

        :param world: Optional shared World instance. If not provided, a new World is created.
        :param root_name: Name of the root body if a new root must be created.
        :param root_prefix: Prefix for the root body name (None for no prefix).
        """
        self.world = world if world is not None else World()

        if self.world.is_empty():
            root = Body(name=PrefixedName(name=root_name, prefix=root_prefix))
            with self.world.modify_world():
                self.world.add_body(body=root)

    @staticmethod
    def _center_mesh_origin(mesh: Mesh) -> None:
        """Shift mesh origin to the center of its local bounding box."""
        bb = mesh.local_frame_bounding_box
        center_x = (bb.min_x + bb.max_x) / 2
        center_y = (bb.min_y + bb.max_y) / 2
        center_z = (bb.min_z + bb.max_z) / 2
        mesh.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            -center_x, -center_y, -center_z, 0, 0, 0
        )

    def build_objects(self, root: Body, specs: List[Any]) -> Dict[str, Connection6DoF]:
        """Create bodies, connections, annotations, and poses from object specs."""
        connections: Dict[str, Connection6DoF] = {}

        with self.world.modify_world():
            for spec in specs:
                shapes: List[Any] = []
                collision: ShapeCollection

                if spec.mesh_path is not None:
                    mesh = Mesh(
                        origin=HomogeneousTransformationMatrix(),
                        filename=str(spec.mesh_path),
                    )
                    if spec.center_mesh:
                        self._center_mesh_origin(mesh)
                    box = mesh.local_frame_bounding_box.as_shape()
                    shapes = [box, mesh]
                    collision = ShapeCollection([box])
                else:
                    if spec.box_scale is None:
                        raise ValueError(
                            f"ObjectSpec {spec.name} requires mesh_path or box_scale."
                        )
                    box = Box(scale=spec.box_scale, color=spec.color)
                    shapes = [box]
                    collision = ShapeCollection([box])

                body = Body(
                    name=PrefixedName(name=spec.name),
                    visual=ShapeCollection(shapes),
                    collision=collision,
                )
                connection = Connection6DoF.create_with_dofs(
                    parent=root, child=body, world=self.world
                )
                self.world.add_connection(connection)
                self.world.add_semantic_annotation(PredefinedObject(body=body))
                connections[spec.name] = connection

        # Set origins in a separate modification block so FK is compiled first
        with self.world.modify_world():
            for spec in specs:
                connections[spec.name].origin = spec.pose

        return connections

    def build_regions(
        self, root: Body, specs: List[RegionSpec]
    ) -> Dict[str, Connection6DoF]:
        """Create regions and connections from region specs."""
        connections: Dict[str, Connection6DoF] = {}

        with self.world.modify_world():
            for spec in specs:
                box = Box(scale=spec.box_scale, color=spec.color)
                region = Region(name=PrefixedName(name=spec.name))
                area = ShapeCollection()
                area.reference_frame = region
                area.append(box)
                region.area = area

                connection = Connection6DoF.create_with_dofs(
                    parent=root, child=region, world=self.world
                )
                self.world.add_connection(connection)
                connections[spec.name] = connection

        # Set origins in a separate modification block so FK is compiled first
        with self.world.modify_world():
            for spec in specs:
                connections[spec.name].origin = spec.pose

        return connections

    def get_predefined_object_bodies(self) -> List[Body]:
        """Get list of pre-defined objects."""
        predefined_object_annotations = self.world.get_semantic_annotations_by_type(
            PredefinedObject
        )
        return list(
            [
                predefined_object.body
                for predefined_object in predefined_object_annotations
            ]
        )
