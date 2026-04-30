from pathlib import Path

from robokudo.world_descriptor import BaseWorldDescriptor, ObjectSpec, RegionSpec
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale


class WorldDescriptor(BaseWorldDescriptor):
    def __init__(self) -> None:
        super().__init__()
        root = self.world.root

        milk_path = (
            Path(__file__).resolve().parents[5]
            / "semantic_digital_twin"
            / "resources"
            / "stl"
            / "milk.stl"
        )

        object_specs = [
            ObjectSpec(
                name="cereal",
                box_scale=Scale(0.20, 0.20, 0.20),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.1, y=1.0, z=1.0, reference_frame=root
                ),
            ),
            ObjectSpec(
                name="milk",
                mesh_path=milk_path,
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.1, y=1.2, z=1.0, reference_frame=root
                ),
            ),
        ]

        region_specs = [
            RegionSpec(
                name="kitchen_island",
                box_scale=Scale(1.0, 2.5, 0.85),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.10, y=1.7, z=1.20, reference_frame=root
                ),
            )
        ]

        self.build_objects(root, object_specs)
        self.build_regions(root, region_specs)
