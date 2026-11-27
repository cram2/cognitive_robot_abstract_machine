"""
World descriptor ``world_semdt_raytracer_tabletop``: a table with boxes, a cup and a
milk carton, rendered by the semantic digital twin ray tracer.
"""

# %% Imports
from __future__ import annotations
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    CameraRange,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.robots.robot_parts import Camera

from robokudo.world_descriptor import (
    BaseWorldDescriptor,
    CameraPlacement,
    ObjectSpec,
    PredefinedObject,
    RegionSpec,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Color, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% World descriptor


class WorldDescriptor(BaseWorldDescriptor):
    """
    A compact tabletop world for SemDT RayTracer-based camera simulation.
    """

    def __init__(self) -> None:
        """
        Place the table, its surface region, a red and a blue box, a cup and a milk
        carton on it and the camera optical frame in the world.
        """
        super().__init__()
        root = self.world.root

        table_top_z = 0.78
        table_thickness = 0.06
        cup_mesh_uri = "package://iai_objects/objects/jeroen_cup.stl"
        milk_mesh_uri = "package://iai_objects/objects/milk.stl"

        object_specs = [
            ObjectSpec(
                name="table",
                box_scale=Scale(x=1.20, y=0.80, z=table_thickness),
                color=Color.from_rgba(red=0.65, green=0.58, blue=0.48, opacity=1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.10,
                    y=1.25,
                    z=table_top_z - (table_thickness / 2.0),
                    reference_frame=root,
                ),
            ),
            ObjectSpec(
                name="box_red",
                box_scale=Scale(x=0.10, y=0.08, z=0.10),
                color=Color.from_rgba(red=0.83, green=0.20, blue=0.20, opacity=1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.20,
                    y=1.20,
                    z=table_top_z + (0.10 / 2.0),
                    reference_frame=root,
                ),
            ),
            ObjectSpec(
                name="box_blue",
                box_scale=Scale(x=0.08, y=0.08, z=0.14),
                color=Color.from_rgba(red=0.22, green=0.37, blue=0.82, opacity=1.0),
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
                box_scale=Scale(x=1.00, y=0.60, z=0.03),
                color=Color.from_rgba(red=0.10, green=0.70, blue=0.30, opacity=0.20),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.10,
                    y=1.25,
                    z=table_top_z + 0.015,
                    reference_frame=root,
                ),
            )
        ]

        self.build_objects(root=root, specs=object_specs)

        cup_color = Color.from_rgba(red=0.18, green=0.78, blue=0.28, opacity=1.0)
        milk_color = Color.from_rgba(red=0.95, green=0.85, blue=0.10, opacity=1.0)
        with self.world.modify_world():
            cup_visual_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=cup_mesh_uri,
                color=cup_color,
            )
            cup_collision_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=cup_mesh_uri,
                color=cup_color,
            )
            cup_body = Body(
                name=PrefixedName(name="jeroen_cup"),
                visual=ShapeCollection(shapes=[cup_visual_mesh]),
                collision=ShapeCollection(shapes=[cup_collision_mesh]),
            )
            cup_connection = Connection6DoF.create_with_dofs(
                parent=root, child=cup_body, world=self.world
            )
            self.world.add_connection(connection=cup_connection)
            self.world.add_semantic_annotation(
                semantic_annotation=PredefinedObject(body=cup_body)
            )

            milk_visual_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=milk_mesh_uri,
                color=milk_color,
            )
            milk_collision_mesh = Mesh(
                origin=HomogeneousTransformationMatrix(),
                filename=milk_mesh_uri,
                color=milk_color,
            )
            milk_body = Body(
                name=PrefixedName(name="milk"),
                visual=ShapeCollection(shapes=[milk_visual_mesh]),
                collision=ShapeCollection(shapes=[milk_collision_mesh]),
            )
            milk_connection = Connection6DoF.create_with_dofs(
                parent=root, child=milk_body, world=self.world
            )
            self.world.add_connection(connection=milk_connection)
            self.world.add_semantic_annotation(
                semantic_annotation=PredefinedObject(body=milk_body)
            )

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

        self.build_regions(root=root, specs=region_specs)
        camera_name = PrefixedName(name="semdt_camera_optical_frame")
        self.build_camera(
            parent=root,
            placement=CameraPlacement(
                camera=Camera(
                    name=camera_name,
                    root=Body(name=camera_name),
                    camera_model=PinholeCameraModel.from_field_of_view(
                        resolution=CameraResolution(), field_of_view=FieldOfView()
                    ),
                    camera_range=CameraRange(
                        minimum_distance=0.05, maximum_distance=8.0
                    ),
                    modalities=(
                        CameraModality.COLOR,
                        CameraModality.DEPTH,
                        CameraModality.SEGMENTATION,
                    ),
                ),
                parent_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.20,
                    y=0.40,
                    z=1.05,
                    roll=-2.2689280275926285,
                    pitch=0.0,
                    yaw=-0.2707963267948965,
                    reference_frame=root,
                ),
            ),
        )
