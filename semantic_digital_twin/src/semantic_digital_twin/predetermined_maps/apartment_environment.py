"""
The apartment described by the ``iai_apartment`` package, built from its visual meshes.
"""

from dataclasses import dataclass

import numpy as np

from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Handle,
    Hinge,
    Shelf,
    ShelfLayer,
    SideTable,
    Sofa,
    Wall,
    Wardrobe,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale

WARDROBE_DOOR_VELOCITY_LIMIT = np.pi / 2
"""
Angular velocity limit of a wardrobe door in rad/s.

Taken from the ``wardrobe_door_*_joint`` limits of the apartment's own URDF, which
describes the same wardrobe this map spawns from meshes.
"""


@dataclass
class ApartmentEnvironment:
    """
    The furniture of the apartment: a shelf, a wall, a bedside table, a sofa, the
    apartment's wall meshes and a two-leaf wardrobe.
    """

    def get_world(self) -> World:
        """
        Create a world holding nothing but the apartment.
        """
        world = World.create_with_root_body("root")
        self.populate(world)
        return world

    def populate(self, world: World) -> None:
        """
        Spawn the apartment's furniture into an existing world.
        """
        self._build_shelf(world)
        self._build_wall_next_to_shelf(world)
        self._build_bedside_table(world)
        self._build_sofa(world)
        self._build_wall_meshes(world)
        self._build_wardrobe(world)

    @staticmethod
    def mesh_path(mesh_file_name: str) -> str:
        """
        Resolve one of the apartment's visual meshes to a local file path.
        """
        return CompositePathResolver().resolve(
            f"package://iai_apartment/meshes/visual/{mesh_file_name}"
        )

    # %% shelf

    def _build_shelf(self, world: World) -> None:
        """
        Spawn the shelf with its four layers.
        """
        with world.modify_world():
            shelf = Shelf.get_annotation_specification(
                "shelf",
                Shelf.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(0.305, 0.85, 1.9), wall_thickness=0.035
                ),
            ).spawn(
                world,
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.455 + (0.85 / 2),
                    -0.17,
                    1.9 / 2,
                    yaw=-np.pi / 2,
                    reference_frame=world.root,
                ),
            )
            for layer_name, layer_height in [
                ("shelf_layer1", 0.283),
                ("shelf_layer2", 0.63),
                ("shelf_layer3", 1.265),
                ("shelf_layer4", 1.613),
            ]:
                shelf.add(
                    ShelfLayer.create_with_new_body_in_world(
                        world=world,
                        name=layer_name,
                        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                            0.455 + (0.85 / 2),
                            -0.17,
                            layer_height,
                            yaw=-np.pi / 2,
                            reference_frame=world.root,
                        ),
                        scale=Scale(0.305, 0.85, 0.018),
                    )
                )

    # %% wall next to the shelf

    def _build_wall_next_to_shelf(self, world: World) -> None:
        """
        Spawn the box-shaped wall segment the shelf stands next to.
        """
        with world.modify_world():
            Wall.create_with_new_body_in_world(
                world=world,
                name="wall",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0, (2.81 / 2), 0, reference_frame=world.root
                ),
                scale=Scale(0.03, 2.81, 0.265),
            )

    # %% bedside table

    def _build_bedside_table(self, world: World) -> None:
        """
        Spawn the bedside table.
        """
        SideTable.get_annotation_specification(
            "bedside_table.dae",
            BodySpecification.mesh(
                "bedside_table.dae",
                self.mesh_path("bedside_table.dae"),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.92, y=2.68, yaw=17.8 * (-np.pi / 32)
                ),
            ),
        ).spawn(world)

    # %% sofa

    def _build_sofa(self, world: World) -> None:
        """
        Spawn the sofa bed.
        """
        # The sofa rests on the floor, so its placement needs the height of its own
        # geometry, which only the specification can measure.
        sofa_body = BodySpecification.mesh(
            "sofa_bed.obj", self.mesh_path("sofa_bed.obj")
        )
        Sofa.get_annotation_specification("sofa_bed.obj", sofa_body).spawn(
            world,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.0, y=3.15, z=sofa_body.scale.z / 2, yaw=17.75 * (-np.pi / 32)
            ),
        )

    # %% walls

    def _build_wall_meshes(self, world: World) -> None:
        """
        Spawn the apartment's walls as one mesh.
        """
        Wall.get_annotation_specification(
            "walls.dae",
            BodySpecification.mesh(
                "walls.dae",
                self.mesh_path("walls.dae"),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-7.34, y=1.43, z=-0.2, yaw=0
                ),
            ),
        ).spawn(world)

    # %% wardrobe

    def _build_wardrobe(self, world: World) -> None:
        """
        Spawn the wardrobe with its two hinged doors and their handles.
        """
        # In this case, each door mesh has its origin on the edge it hangs from, so a hinge at the
        # door's own frame already sits on the rotation axis. The wardrobe opens towards
        # -x and the leaves reach it from opposite sides, so they swing about the shared
        # vertical axis in opposite directions.
        doors = [
            Door.get_annotation_specification(
                door_mesh,
                BodySpecification.mesh(
                    door_mesh,
                    self.mesh_path(door_mesh),
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        -0.3246, door_y
                    ),
                ),
                part_specifications={
                    "handle": Handle.get_annotation_specification(
                        f"wardrobe_door_handle_{side}",
                        BodySpecification.mesh(
                            f"wardrobe_door_handle_{side}",
                            self.mesh_path("wardrobe_door_handle.dae"),
                            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                                -0.032089, handle_y, 0.973703
                            ),
                        ),
                    ),
                    "mechanical_joint": Hinge.get_annotation_specification(
                        f"wardrobe_hinge_{side}",
                        Hinge.get_default_root_kinematic_structure_entity_specification(),
                        parent_connection_specification=Hinge.parent_connection_specification(
                            axis=Vector3.Z(),
                            dof_limits=DegreeOfFreedomLimits(
                                lower=DerivativeMap[float](
                                    position=min(0.0, opening_angle),
                                    velocity=-WARDROBE_DOOR_VELOCITY_LIMIT,
                                ),
                                upper=DerivativeMap[float](
                                    position=max(0.0, opening_angle),
                                    velocity=WARDROBE_DOOR_VELOCITY_LIMIT,
                                ),
                            ),
                        ),
                    ),
                },
            )
            for side, door_mesh, handle_y, door_y, opening_angle in [
                ("left", "wardrobe_door_left.dae", -0.460513, 0.5, -np.pi / 2),
                ("right", "wardrobe_door_right.dae", 0.460513, -0.5, np.pi / 2),
            ]
        ]

        Wardrobe.get_annotation_specification(
            "wardrobe.dae",
            BodySpecification.mesh(
                "wardrobe.dae",
                self.mesh_path("wardrobe.dae"),
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2, y=-0.17, yaw=-np.pi / 2
                ),
            ),
            part_specifications={"doors": doors},
        ).spawn(world)
