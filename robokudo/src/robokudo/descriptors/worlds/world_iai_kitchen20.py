"""
World descriptor of the IAI kitchen with a cereal box and a milk carton near the kitchen island.
"""

from __future__ import annotations

from pathlib import Path

from robokudo.world_descriptor import BaseWorldDescriptor, ObjectSpec, RegionSpec
from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale


class WorldDescriptor(BaseWorldDescriptor):
    """
    The IAI kitchen with a cereal box and a milk carton and the kitchen island as region.
    """

    def __init__(self) -> None:
        """
        Adds the objects and the region to the world.
        """
        super().__init__()
        root = self.world.root

        milk_path = Path(
            PackageUriResolver().resolve("package://iai_objects/objects/milk.stl")
        )

        object_specs = [
            ObjectSpec(
                name="cereal",
                box_scale=Scale(0.10, 0.20, 0.30),
                pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-1.3,
                    y=0.75,
                    z=1.0,
                    reference_frame=root,
                    yaw=0.3,
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
