from importlib.util import find_spec
from pathlib import Path

from robokudo.world_descriptor import (
    BaseWorldDescriptor,
    ObjectSpec,
    PredefinedObject,
    RegionSpec,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Color, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class WorldDescriptor(BaseWorldDescriptor):
    """A compact tabletop world for SemDT RayTracer-based camera simulation."""

    def __init__(self) -> None:
        super().__init__()
        root = self.world.root

        table_top_z = 0.78
        table_thickness = 0.06
        pycram_spec = find_spec("pycram")
        if pycram_spec is None or pycram_spec.origin is None:
            raise ImportError("Could not locate the pycram package for mesh loading.")
        object_mesh_dir = (
            Path(pycram_spec.origin).resolve().parents[2] / "resources" / "objects"
        )
        cup_mesh_path = object_mesh_dir / "jeroen_cup.stl"
        milk_mesh_path = object_mesh_dir / "milk.stl"

        object_specs = [
            ObjectSpec(
                name="table",
                box_scale=Scale(1.20, 0.80, table_thickness),
                color=Color(0.65, 0.58, 0.48, 1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.10,
                    y=1.25,
                    z=table_top_z - (table_thickness / 2.0),
                    reference_frame=root,
                ),
            ),
            ObjectSpec(
                name="box_red",
                box_scale=Scale(0.10, 0.08, 0.10),
                color=Color(0.83, 0.20, 0.20, 1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.20,
                    y=1.20,
                    z=table_top_z + (0.10 / 2.0),
                    reference_frame=root,
                ),
            ),
            ObjectSpec(
                name="box_blue",
                box_scale=Scale(0.08, 0.08, 0.14),
                color=Color(0.22, 0.37, 0.82, 1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.00,
                    y=1.32,
                    z=table_top_z + (0.14 / 2.0),
                    yaw=0.40,
                    reference_frame=root,
                ),
            ),
        ]

        region_specs = [
            RegionSpec(
                name="table_surface_region",
                box_scale=Scale(1.00, 0.60, 0.03),
                color=Color(0.10, 0.70, 0.30, 0.20),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.10,
                    y=1.25,
                    z=table_top_z + 0.015,
                    reference_frame=root,
                ),
            )
        ]

        self.build_objects(root, object_specs)

        cup_color = Color(0.18, 0.78, 0.28, 1.0)
        milk_color = Color(0.95, 0.85, 0.10, 1.0)
        with self.world.modify_world():
            cup_visual_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(cup_mesh_path),
                color=cup_color,
            )
            cup_collision_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(cup_mesh_path),
                color=cup_color,
            )
            cup_body = Body(
                name=PrefixedName(name="jeroen_cup"),
                visual=ShapeCollection([cup_visual_mesh]),
                collision=ShapeCollection([cup_collision_mesh]),
            )
            cup_connection = Connection6DoF.create_with_dofs(
                parent=root, child=cup_body, world=self.world
            )
            self.world.add_connection(cup_connection)
            self.world.add_semantic_annotation(PredefinedObject(body=cup_body))

            milk_visual_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(milk_mesh_path),
                color=milk_color,
            )
            milk_collision_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=str(milk_mesh_path),
                color=milk_color,
            )
            milk_body = Body(
                name=PrefixedName(name="milk"),
                visual=ShapeCollection([milk_visual_mesh]),
                collision=ShapeCollection([milk_collision_mesh]),
            )
            milk_connection = Connection6DoF.create_with_dofs(
                parent=root, child=milk_body, world=self.world
            )
            self.world.add_connection(milk_connection)
            self.world.add_semantic_annotation(PredefinedObject(body=milk_body))

        with self.world.modify_world():
            cup_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-1.37,
                y=0.95,
                z=table_top_z,
                yaw=-0.25,
                reference_frame=root,
            )
            milk_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.85,
                y=0.98,
                z=table_top_z - milk_collision_mesh.local_frame_bounding_box.min_z,
                yaw=0.18,
                reference_frame=root,
            )

        self.build_regions(root, region_specs)
