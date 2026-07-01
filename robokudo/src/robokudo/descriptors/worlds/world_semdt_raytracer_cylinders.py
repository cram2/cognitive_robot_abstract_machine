from robokudo.world_descriptor import BaseWorldDescriptor, ObjectSpec, RegionSpec
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale, Color


class WorldDescriptor(BaseWorldDescriptor):
    """A compact tabletop world for SemDT RayTracer with cylindrical target objects."""

    def __init__(self) -> None:
        super().__init__()
        root = self.world.root

        table_top_z = 0.78
        table_thickness = 0.06
        table_surface_region_thickness = 0.03

        red_cyl_width = 0.10
        red_cyl_height = 0.10
        blue_cyl_width = 0.08
        blue_cyl_height = 0.14

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
                name="cylinder_red",
                cylinder_width=red_cyl_width,
                cylinder_height=red_cyl_height,
                color=Color(0.83, 0.20, 0.20, 1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.20,
                    y=1.20,
                    z=table_top_z + (red_cyl_height / 2.0),
                    reference_frame=root,
                ),
            ),
            ObjectSpec(
                name="cylinder_blue",
                cylinder_width=blue_cyl_width,
                cylinder_height=blue_cyl_height,
                color=Color(0.22, 0.37, 0.82, 1.0),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.00,
                    y=1.32,
                    z=table_top_z + (blue_cyl_height / 2.0),
                    yaw=0.40,
                    reference_frame=root,
                ),
            ),
        ]

        region_specs = [
            RegionSpec(
                name="table_surface_region",
                box_scale=Scale(1.00, 0.60, table_surface_region_thickness),
                parent_name="table",
                color=Color(0.10, 0.70, 0.30, 0.20),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.0,
                    y=0.0,
                    z=(table_thickness / 2.0) + (table_surface_region_thickness / 2.0),
                ),
            )
        ]

        self.build_objects(root, object_specs)
        self.build_regions(root, region_specs)
