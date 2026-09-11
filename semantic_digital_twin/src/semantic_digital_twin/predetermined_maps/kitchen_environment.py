import numpy as np
import threading
import rclpy

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Table,
    Sofa,
    TrashCan,
    Fridge,
    CounterTop,
    Wall,
    Cabinet,
    Cupboard,
    ShelfLayer,
    Hinge,
    Door,
    Handle,
    DiningTable,
    Leg,
    Drawer,
    Desk,
    Lid,
    Sink,
    Dishwasher,
    Cooktop,
    Oven,
    WallPanel,
    Slider,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
    DegreeOfFreedom,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
    PrismaticConnection,
)
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room, Floor
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class KitchenEnvironment:
    """
    Manages the Kitchen Environment world with walls, furniture, and room layouts.
    """

    def get_world(self) -> World:
        """
        Constructs and returns a new World instance, setting up its environment,
        including walls, furniture, and rooms.

        :return: A new world instance with the initialized environment.
        """
        world = World.create_with_root_body("root")

        self._build_environment_walls(world)
        self._build_environment_furniture(world)
        self._build_environment_rooms(world)

        return world

    def _build_environment_walls(self, world: World):
        """
        Builds and configures the environment walls for a given world.
        """
        root = world.root

        north_west_wall = Cylinder(width=1.53, height=3.00)
        shape_geometry = ShapeCollection([north_west_wall])
        north_west_wall_body = Body(
            name=PrefixedName("north_west_wall_body"),
            collision=shape_geometry,
            visual=shape_geometry,
        )

        root_C_north_west_wall = FixedConnection(
            parent=root,
            child=north_west_wall_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=4.924, y=6.295, z=1.50
            ),
        )

        with world.modify_world():
            south_wall1 = Wall.create_with_new_body_in_world(
                world=world,
                name="south_wall1",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=-2.01),
                scale=Scale(x=0.05, y=1.00, z=3.00),
            )

            south_wall2 = Wall.create_with_new_body_in_world(
                world=world,
                name="south_wall2",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-1.45, yaw=np.pi / 2
                ),
                scale=Scale(x=0.05, y=0.29, z=3.00),
            )

            south_wall3 = Wall.create_with_new_body_in_world(
                world=world,
                name="south_wall3",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29, y=-0.9925
                ),
                scale=Scale(x=0.05, y=1.085, z=1.00),
            )

            south_wall4 = Wall.create_with_new_body_in_world(
                world=world,
                name="south_wall4",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-0.45, yaw=np.pi / 2
                ),
                scale=Scale(x=0.05, y=0.29, z=1.00),
            )

            south_wall5 = Wall.create_with_new_body_in_world(
                world=world,
                name="south_wall5",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=0.45, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 0.29, 1.00),
            )

            south_wall6 = Wall.create_with_new_body_in_world(
                world=world,
                name="south_wall6",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=1.80
                ),
                scale=Scale(0.05, 2.75, 1.00),
            )

            south_wall7 = Wall.create_with_new_body_in_world(
                world=world,
                name="south_wall7",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=5.16
                ),
                scale=Scale(0.05, 2.27, 1.00),
            )

            east_wall = Wall.create_with_new_body_in_world(
                world=world,
                name="east_wall",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.462, y=-2.535, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 4.924, 3.00),
            )

            middle_wall = Wall.create_with_new_body_in_world(
                world=world,
                name="middle_wall",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.20975, y=5.00
                ),
                scale=Scale(0.05, 2.67, 1.00),
            )

            west_wall = Wall.create_with_new_body_in_world(
                world=world,
                name="west_wall",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.9345, y=6.32, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 4.449, 3.00),
            )

            north_wall = Wall.create_with_new_body_in_world(
                world=world,
                name="north_wall",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.949, y=1.51
                ),
                scale=Scale(0.05, 8.04, 3.00),
            )

            world.add_connection(root_C_north_west_wall)
            return world

    def _build_environment_furniture(self, world: World):
        """
        Adds furniture items and room layouts to the scene graph.
        """

        # Angular velocity limit of a hinged door in rad/s.
        # Taken from the revolute joint limits of the apartment description in ``iai_apartment``,
        # which uses this value for every one of its hinged doors.
        hinged_door_velocity_limit = np.pi / 2

        # Linear velocity limit of a sliding drawer in m/s.
        # Taken from the prismatic joint limits of the apartment description in ``iai_apartment``.
        sliding_drawer_velocity_limit = 0.5

        standard_handle_depth = 0.068
        standard_handle_height = 0.015

        with world.modify_world():
            # --- TRASH CAN ---
            trash_can = TrashCan.get_annotation_specification(
                "trash_can",
                TrashCan.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(x=0.30, y=0.30, z=0.40), wall_thickness=0.02
                ),
            ).spawn(
                world,
                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.416, y=5.5, z=0.2
                ),
            )
            for shape in trash_can.root.visual.shapes:
                shape.color = Color.GRAY()

            # --- REFRIGERATOR ---
            fridge_length, fridge_width = 0.60, 0.60
            fridge_front_width = 0.595
            fridge_drawer_floor_gap = 0.145
            fridge_drawer_height = 0.354
            fridge_drawer_door_gap = 0.003
            fridge_door_height = 0.965
            fridge_door_top_gap = 0.008
            fridge_top_plate_height = 0.025
            fridge_top_gap_facing_setback = 0.023
            fridge_base_facing_setback = 0.07
            fridge_base_facing_thickness = 0.02
            fridge_height = (
                fridge_drawer_floor_gap
                + fridge_drawer_height
                + fridge_drawer_door_gap
                + fridge_door_height
                + fridge_door_top_gap
                + fridge_top_plate_height
            )
            south_wall_inner_surface_x = 0.025
            fridge_wall_gap = 0.478
            fridge_counter_boundary_x = (
                south_wall_inner_surface_x + fridge_wall_gap + fridge_width
            )
            fridge_center_x = fridge_counter_boundary_x - fridge_width / 2
            fridge_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=fridge_center_x,
                y=-2.181,
                z=fridge_height / 2,
                yaw=-np.pi / 2,
            )

            refrigerator = Fridge.get_annotation_specification(
                "refrigerator",
                Fridge.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(x=fridge_length, y=fridge_width, z=fridge_height),
                    wall_thickness=0.02,
                ),
            ).spawn(world, parent_T_self=fridge_pose)
            for shape in refrigerator.root.visual.shapes:
                shape.color = Color.GRAY()

            door_thickness = 0.02
            fridge_top_plate = WallPanel.create_with_new_body_in_world(
                world=world,
                name="fridge_top_plate",
                world_root_T_self=fridge_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-fridge_length / 2,
                    z=fridge_height / 2 - fridge_top_plate_height / 2,
                ),
                scale=Scale(
                    x=door_thickness,
                    y=fridge_width,
                    z=fridge_top_plate_height,
                ),
            )
            for shape in fridge_top_plate.root.visual.shapes:
                shape.color = Color.GRAY()
            refrigerator.add_object(fridge_top_plate)

            fridge_top_gap_facing = WallPanel.create_with_new_body_in_world(
                world=world,
                name="fridge_top_gap_facing",
                world_root_T_self=fridge_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-fridge_length / 2 + fridge_top_gap_facing_setback,
                    z=(
                        fridge_height / 2
                        - fridge_top_plate_height
                        - fridge_door_top_gap / 2
                    ),
                ),
                scale=Scale(
                    x=door_thickness,
                    y=fridge_width,
                    z=fridge_door_top_gap,
                ),
            )
            for shape in fridge_top_gap_facing.root.visual.shapes:
                shape.color = Color.GRAY()
            refrigerator.add_object(fridge_top_gap_facing)

            hinge_local_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-fridge_length / 2,
                y=-fridge_front_width / 2,
                z=(
                    -fridge_height / 2
                    + fridge_drawer_floor_gap
                    + fridge_drawer_height
                    + fridge_drawer_door_gap
                    + fridge_door_height / 2
                ),
            )
            hinge_world_pose = fridge_pose @ hinge_local_pose
            fridge_door_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name="fridge_door_hinge",
                world_root_T_self=hinge_world_pose,
                parent_connection_specification=Hinge.parent_connection_specification(
                    axis=Vector3.Z(),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](
                            position=0.0, velocity=-hinged_door_velocity_limit
                        ),
                        upper=DerivativeMap[float](
                            position=np.pi / 2, velocity=hinged_door_velocity_limit
                        ),
                    ),
                ),
            )

            fridge_door = Door.create_with_new_body_in_world(
                world=world,
                name="fridge_door",
                world_root_T_self=hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=fridge_front_width / 2
                ),
                scale=Scale(
                    x=door_thickness,
                    y=fridge_front_width,
                    z=fridge_door_height,
                ),
            )
            for shape in fridge_door.root.visual.shapes:
                shape.color = Color.WHITE()
            fridge_door.add(fridge_door_hinge)
            refrigerator.add(fridge_door)

            drawer_depth = 0.5
            drawer_world_pose = (
                fridge_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=(
                        -fridge_length / 2
                        - door_thickness / 2
                        + drawer_depth / 2
                    ),
                    z=(
                        -fridge_height / 2
                        + fridge_drawer_floor_gap
                        + fridge_drawer_height / 2
                    ),
                )
            )
            fridge_drawer = Drawer.create_with_new_body_in_world(
                world=world,
                name="fridge_drawer",
                world_root_T_self=drawer_world_pose,
                scale=Scale(
                    x=drawer_depth,
                    y=fridge_front_width,
                    z=fridge_drawer_height,
                ),
            )

            fridge_slider = Slider.create_with_new_body_in_world(
                world=world,
                name="fridge_drawer_slider",
                world_root_T_self=drawer_world_pose,
                parent_connection_specification=Slider.parent_connection_specification(
                    axis=Vector3.NEGATIVE_X(),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](
                            position=0.0, velocity=-sliding_drawer_velocity_limit
                        ),
                        upper=DerivativeMap[float](
                            position=0.5, velocity=sliding_drawer_velocity_limit
                        ),
                    ),
                ),
            )

            fridge_drawer.add(fridge_slider)

            for shape in fridge_drawer.root.visual.shapes:
                shape.color = Color.WHITE()
            refrigerator.add(fridge_drawer)

            fridge_base_facing = WallPanel.create_with_new_body_in_world(
                world=world,
                name="fridge_base_facing",
                world_root_T_self=drawer_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=(
                        -drawer_depth / 2
                        + fridge_base_facing_setback
                        + fridge_base_facing_thickness / 2
                    ),
                    z=(
                        -fridge_drawer_height / 2
                        - fridge_drawer_floor_gap / 2
                    ),
                ),
                scale=Scale(
                    x=fridge_base_facing_thickness,
                    y=fridge_width,
                    z=fridge_drawer_floor_gap,
                ),
            )
            for shape in fridge_base_facing.root.visual.shapes:
                shape.color = Color.GRAY()
            refrigerator.add_object(fridge_base_facing)

            fridge_door_handle_length = 0.855
            door_handle_world_pose = (
                hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-door_thickness / 2,
                    y=fridge_front_width - 0.03,
                    roll=np.pi / 2,
                )
            )
            fridge_door_handle = Handle.get_annotation_specification(
                "fridge_door_handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=standard_handle_depth,
                        y=fridge_door_handle_length,
                        z=standard_handle_height,
                    ),
                    thickness=standard_handle_height,
                ),
            ).spawn(world, parent_T_self=door_handle_world_pose)
            for shape in fridge_door_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            fridge_door.add(fridge_door_handle)

            drawer_handle_world_pose = (
                drawer_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-drawer_depth / 2,
                    z=fridge_drawer_height / 2 - 0.03,
                )
            )
            fridge_drawer_handle = Handle.get_annotation_specification(
                "fridge_drawer_handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=standard_handle_depth,
                        y=0.5,
                        z=standard_handle_height,
                    ),
                    thickness=standard_handle_height,
                ),
            ).spawn(world, parent_T_self=drawer_handle_world_pose)
            for shape in fridge_drawer_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            fridge_drawer.add(fridge_drawer_handle)

            # --- KITCHEN COUNTER ---
            counter_top_length, counter_top_depth = 2.044, 0.658
            counter_top_surface_height = 0.85
            counter_top_thickness = 0.026
            counter_base_facing_height = 0.099
            counter_base_facing_setback = 0.07
            counter_cabinet_height = (
                counter_top_surface_height
                - counter_top_thickness
                - counter_base_facing_height
            )
            counter_top_center_x = fridge_counter_boundary_x + counter_top_length / 2
            root_T_counter_cabinets = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=counter_top_center_x,
                y=-2.181,
                z=counter_base_facing_height + counter_cabinet_height / 2,
                yaw=-np.pi / 2,
            )
            root_T_counter_top = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=counter_top_center_x,
                y=-2.181,
                z=counter_top_surface_height - counter_top_thickness / 2,
                yaw=-np.pi / 2,
            )

            counter_top = CounterTop.create_with_new_body_in_world(
                world=world,
                name="counter_top",
                world_root_T_self=root_T_counter_top,
                scale=Scale(
                    x=counter_top_depth,
                    y=counter_top_length,
                    z=counter_top_thickness,
                ),
            )
            for shape in counter_top.root.visual.shapes:
                shape.color = Color.BEIGE()

            counter_base_facing = WallPanel.create_with_new_body_in_world(
                world=world,
                name="counter_base_facing",
                world_root_T_self=root_T_counter_cabinets
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-counter_top_depth / 2 + counter_base_facing_setback,
                    z=-counter_cabinet_height / 2 - counter_base_facing_height / 2,
                ),
                scale=Scale(
                    x=0.02,
                    y=counter_top_length,
                    z=counter_base_facing_height,
                ),
            )
            for shape in counter_base_facing.root.visual.shapes:
                shape.color = Color.GRAY()
            counter_top.add_object(counter_base_facing)

            sink_width, sink_depth, sink_fridge_gap = 0.86, 0.50, 0.115
            counter_top_sink_y = (
                fridge_center_x
                + fridge_width / 2
                + sink_fridge_gap
                + sink_width / 2
                - counter_top_center_x
            )
            sink = Sink.create_with_new_body_in_world(
                world=world,
                name="sink",
                world_root_T_self=root_T_counter_top
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=counter_top_sink_y, z=counter_top_thickness / 2 + 0.005
                ),
                scale=Scale(x=sink_depth, y=sink_width, z=0.005),
            )
            for shape in sink.root.visual.shapes:
                shape.color = Color.BLACK()
            counter_top.add(sink)

            module_1_width, module_2_width = 0.60, 0.60
            module_3_width = counter_top_length - module_1_width - module_2_width
            module_3_handle_width = 0.705
            module_1_front_width = 0.595
            module_1_face_plate_height = 0.143
            module_1_door_thickness = 0.02
            module_1_door_gap = 0.005
            module_1_door_height = (
                counter_cabinet_height
                - module_1_face_plate_height
                - module_1_door_gap
            )
            module_1_door_center_height = (
                module_1_door_height - counter_cabinet_height
            ) / 2
            module_1_handle_height = standard_handle_height
            module_1_handle_top_inset = 0.04
            module_2_door_gap = 0.005
            module_2_door_height = counter_cabinet_height - module_2_door_gap

            # Module 1: Cabinet
            module_1_pose = (
                root_T_counter_cabinets
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-counter_top_length / 2 + module_1_width / 2
                )
            )
            module_1_cabinet = Cabinet.get_annotation_specification(
                "module_1_cabinet",
                Cabinet.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=counter_top_depth,
                        y=module_1_width,
                        z=counter_cabinet_height,
                    ),
                    wall_thickness=0.02,
                ),
            ).spawn(world, parent_T_self=module_1_pose)
            for shape in module_1_cabinet.root.visual.shapes:
                shape.color = Color.GRAY()

            module_1_face_plate = WallPanel.create_with_new_body_in_world(
                world=world,
                name="module_1_face_plate",
                world_root_T_self=module_1_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-counter_top_depth / 2 + module_1_door_thickness / 2,
                    z=(counter_cabinet_height - module_1_face_plate_height) / 2,
                ),
                scale=Scale(
                    x=module_1_door_thickness,
                    y=module_1_front_width,
                    z=module_1_face_plate_height,
                ),
            )
            for shape in module_1_face_plate.root.visual.shapes:
                shape.color = Color.WHITE()

            module_1_hinge_world_pose = (
                module_1_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-counter_top_depth / 2 + module_1_door_thickness / 2,
                    y=-module_1_front_width / 2,
                    z=module_1_door_center_height,
                )
            )
            module_1_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name="module_1_hinge",
                world_root_T_self=module_1_hinge_world_pose,
                parent_connection_specification=Hinge.parent_connection_specification(
                    axis=Vector3.Z(),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](
                            position=0.0, velocity=-hinged_door_velocity_limit
                        ),
                        upper=DerivativeMap[float](
                            position=np.pi / 2, velocity=hinged_door_velocity_limit
                        ),
                    ),
                ),
            )
            module_1_door = Door.create_with_new_body_in_world(
                world=world,
                name="module_1_door",
                world_root_T_self=module_1_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=module_1_front_width / 2
                ),
                scale=Scale(
                    x=module_1_door_thickness,
                    y=module_1_front_width,
                    z=module_1_door_height,
                ),
            )
            for shape in module_1_door.root.visual.shapes:
                shape.color = Color.WHITE()
            module_1_door.add(module_1_hinge)
            module_1_cabinet.add(module_1_door)

            module_1_handle = Handle.get_annotation_specification(
                "module_1_handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=standard_handle_depth,
                        y=module_1_front_width - 0.06,
                        z=module_1_handle_height,
                    ),
                    thickness=standard_handle_height,
                ),
            ).spawn(
                world,
                parent_T_self=module_1_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-module_1_door_thickness / 2,
                    y=module_1_front_width / 2,
                    z=(
                        module_1_door_height / 2
                        - module_1_handle_top_inset
                        - module_1_handle_height / 2
                    ),
                ),
            )
            for shape in module_1_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            module_1_door.add(module_1_handle)

            # Module 2: Dishwasher
            module_2_door_thickness = 0.02
            module_2_pose = (
                root_T_counter_cabinets
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-counter_top_length / 2 + module_1_width + module_2_width / 2
                )
            )
            dishwasher = Dishwasher.get_annotation_specification(
                "dishwasher",
                Dishwasher.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=counter_top_depth,
                        y=module_2_width,
                        z=counter_cabinet_height,
                    ),
                    wall_thickness=0.02,
                ),
            ).spawn(world, parent_T_self=module_2_pose)
            for shape in dishwasher.root.visual.shapes:
                shape.color = Color.GRAY()

            module_2_hinge_world_pose = (
                module_2_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-counter_top_depth / 2 + module_2_door_thickness / 2,
                    z=-counter_cabinet_height / 2,
                )
            )
            module_2_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name="dishwasher_hinge",
                world_root_T_self=module_2_hinge_world_pose,
                parent_connection_specification=Hinge.parent_connection_specification(
                    axis=Vector3.NEGATIVE_Y(),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](
                            position=0.0, velocity=-hinged_door_velocity_limit
                        ),
                        upper=DerivativeMap[float](
                            position=np.pi / 2, velocity=hinged_door_velocity_limit
                        ),
                    ),
                ),
            )
            module_2_door = Door.create_with_new_body_in_world(
                world=world,
                name="dishwasher_door",
                world_root_T_self=module_2_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=module_2_door_height / 2
                ),
                scale=Scale(
                    x=module_2_door_thickness,
                    y=module_2_width,
                    z=module_2_door_height,
                ),
            )
            for shape in module_2_door.root.visual.shapes:
                shape.color = Color.WHITE()
            module_2_door.add(module_2_hinge)
            dishwasher.add(module_2_door)

            module_2_handle = Handle.get_annotation_specification(
                "dishwasher_handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=standard_handle_depth,
                        y=module_2_width - 0.06,
                        z=standard_handle_height,
                    ),
                    thickness=standard_handle_height,
                ),
            ).spawn(
                world,
                parent_T_self=module_2_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-module_2_door_thickness / 2,
                    z=module_2_door_height - 0.03,
                ),
            )
            for shape in module_2_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            module_2_door.add(module_2_handle)

            # Module 3: Cabinet with Drawers
            module_3_pose = (
                root_T_counter_cabinets
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=counter_top_length / 2 - module_3_width / 2
                )
            )
            module_3_cabinet = Cabinet.get_annotation_specification(
                "module_3_cabinet",
                Cabinet.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=counter_top_depth,
                        y=module_3_width,
                        z=counter_cabinet_height,
                    ),
                    wall_thickness=0.02,
                ),
            ).spawn(world, parent_T_self=module_3_pose)
            for shape in module_3_cabinet.root.visual.shapes:
                shape.color = Color.GRAY()
            counter_top.add_object(module_3_cabinet)

            drawer_heights = [
                counter_cabinet_height * 0.4,
                counter_cabinet_height * 0.4,
                counter_cabinet_height * 0.2,
            ]
            drawer_face_heights = [0.287, 0.287, 0.143]
            module_3_drawer_depth = 0.30
            drawer_bottom_height = -counter_cabinet_height / 2
            for drawer_index, (height, face_height) in enumerate(
                zip(drawer_heights, drawer_face_heights)
            ):
                drawer_center_height = drawer_bottom_height + height / 2
                drawer_pose = (
                    module_3_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-counter_top_depth / 2 + module_3_drawer_depth / 2,
                        z=drawer_center_height,
                    )
                )
                drawer = Drawer.create_with_new_body_in_world(
                    world=world,
                    name=f"counter_drawer_{drawer_index}",
                    world_root_T_self=drawer_pose,
                    scale=Scale(
                        x=module_3_drawer_depth,
                        y=module_3_width - 0.04,
                        z=face_height,
                    ),
                )

                slider = Slider.create_with_new_body_in_world(
                    world=world,
                    name=f"counter_drawer_{drawer_index}_slider",
                    world_root_T_self=drawer_pose,
                    parent_connection_specification=Slider.parent_connection_specification(
                        axis=Vector3.NEGATIVE_X(),
                        dof_limits=DegreeOfFreedomLimits(
                            lower=DerivativeMap[float](
                                position=0.0, velocity=-sliding_drawer_velocity_limit
                            ),
                            upper=DerivativeMap[float](
                                position=0.25, velocity=sliding_drawer_velocity_limit
                            ),
                        ),
                    ),
                )
                drawer.add(slider)

                for shape in drawer.root.visual.shapes:
                    shape.color = Color.WHITE()
                module_3_cabinet.add(drawer)

                handle_pose = (
                    drawer_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-module_3_drawer_depth / 2,
                        z=face_height / 2 - 0.03,
                    )
                )
                handle = Handle.get_annotation_specification(
                    f"counter_drawer_{drawer_index}_handle",
                    Handle.get_default_root_kinematic_structure_entity_specification(
                        scale=Scale(
                            x=standard_handle_depth,
                            y=module_3_handle_width,
                            z=standard_handle_height,
                        ),
                        thickness=standard_handle_height,
                    ),
                ).spawn(world, parent_T_self=handle_pose)
                for shape in handle.root.visual.shapes:
                    shape.color = Color.GRAY()
                drawer.add(handle)
                drawer_bottom_height += height

            # --- OVEN TOWER ---
            oven_width, oven_depth = 1.20, 0.658
            oven_base_facing_height = 0.145
            oven_side_drawer_height = 1.324
            oven_side_drawer_top_gap = 0.008
            oven_top_plate_height = 0.025
            oven_top_gap_facing_setback = 0.023
            oven_height = (
                oven_base_facing_height
                + oven_side_drawer_height
                + oven_side_drawer_top_gap
                + oven_top_plate_height
            )
            counter_tower_gap = 0.001
            tower_center_x = (
                fridge_counter_boundary_x
                + counter_top_length
                + counter_tower_gap
                + oven_width / 2
            )
            tower_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=tower_center_x, y=-2.181, z=oven_height / 2, yaw=-np.pi / 2
            )
            tower = Cupboard.get_annotation_specification(
                "oven_tower",
                Cupboard.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(x=oven_depth, y=oven_width, z=oven_height),
                    wall_thickness=0.02,
                ),
            ).spawn(world, parent_T_self=tower_pose)
            for shape in tower.root.visual.shapes:
                shape.color = Color.GRAY()

            center_width, side_width = 0.60, 0.30
            side_drawer_width = 0.294
            center_front_width = 0.595
            center_door_thickness = 0.02
            center_handle_width = 0.505
            center_drawer_depth = 0.30
            center_cabinet_door_height = 0.577
            center_cabinet_drawer_gap = 0.002
            center_drawer_height = 0.143
            center_oven_height = (
                oven_side_drawer_height
                - center_cabinet_door_height
                - center_cabinet_drawer_gap
                - center_drawer_height
            )
            oven_panel_thickness = 0.02
            oven_base_facing_setback = 0.07

            oven_base_facing = WallPanel.create_with_new_body_in_world(
                world=world,
                name="oven_base_facing",
                world_root_T_self=tower_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=(
                        -oven_depth / 2
                        + oven_base_facing_setback
                        + oven_panel_thickness / 2
                    ),
                    z=-oven_height / 2 + oven_base_facing_height / 2,
                ),
                scale=Scale(
                    x=oven_panel_thickness,
                    y=oven_width,
                    z=oven_base_facing_height,
                ),
            )
            for shape in oven_base_facing.root.visual.shapes:
                shape.color = Color.GRAY()
            tower.add_object(oven_base_facing)

            oven_top_plate = WallPanel.create_with_new_body_in_world(
                world=world,
                name="oven_top_plate",
                world_root_T_self=tower_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-oven_depth / 2,
                    z=oven_height / 2 - oven_top_plate_height / 2,
                ),
                scale=Scale(
                    x=oven_panel_thickness,
                    y=oven_width,
                    z=oven_top_plate_height,
                ),
            )
            for shape in oven_top_plate.root.visual.shapes:
                shape.color = Color.GRAY()
            tower.add_object(oven_top_plate)

            oven_top_gap_facing = WallPanel.create_with_new_body_in_world(
                world=world,
                name="oven_top_gap_facing",
                world_root_T_self=tower_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-oven_depth / 2 + oven_top_gap_facing_setback,
                    z=(
                        oven_height / 2
                        - oven_top_plate_height
                        - oven_side_drawer_top_gap / 2
                    ),
                ),
                scale=Scale(
                    x=oven_panel_thickness,
                    y=oven_width,
                    z=oven_side_drawer_top_gap,
                ),
            )
            for shape in oven_top_gap_facing.root.visual.shapes:
                shape.color = Color.GRAY()
            tower.add_object(oven_top_gap_facing)

            # Side Drawers
            for side in [-1, 1]:
                side_name = "left" if side == -1 else "right"
                side_drawer_pose = (
                    tower_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=side * (center_width / 2 + side_width / 2),
                        z=(
                            -oven_height / 2
                            + oven_base_facing_height
                            + oven_side_drawer_height / 2
                        ),
                    )
                )
                drawer = Drawer.create_with_new_body_in_world(
                    world=world,
                    name=f"oven_side_drawer_{side_name}",
                    world_root_T_self=side_drawer_pose,
                    scale=Scale(
                        x=oven_depth,
                        y=side_drawer_width,
                        z=oven_side_drawer_height,
                    ),
                )

                slider = Slider.create_with_new_body_in_world(
                    world=world,
                    name=f"oven_side_drawer_{side_name}_slider",
                    world_root_T_self=side_drawer_pose,
                    parent_connection_specification=Slider.parent_connection_specification(
                        axis=Vector3.NEGATIVE_X(),
                        dof_limits=DegreeOfFreedomLimits(
                            lower=DerivativeMap[float](
                                velocity=-sliding_drawer_velocity_limit
                            ),
                            upper=DerivativeMap[float](
                                velocity=sliding_drawer_velocity_limit
                            ),
                        ),
                    ),
                )
                drawer.add(slider)

                for shape in drawer.root.visual.shapes:
                    shape.color = Color.WHITE()
                tower.add(drawer)

                handle_pose = (
                    side_drawer_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-oven_depth / 2, roll=np.pi / 2
                    )
                )
                handle = Handle.get_annotation_specification(
                    f"oven_side_handle_{side_name}",
                    Handle.get_default_root_kinematic_structure_entity_specification(
                        scale=Scale(
                            x=standard_handle_depth,
                            y=oven_side_drawer_height - 0.08,
                            z=standard_handle_height,
                        ),
                        thickness=standard_handle_height,
                    ),
                ).spawn(world, parent_T_self=handle_pose)
                for shape in handle.root.visual.shapes:
                    shape.color = Color.GRAY()
                drawer.add(handle)

            # Center: Bottom Cabinet
            cabinet_pose = tower_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(
                z=(
                    -oven_height / 2
                    + oven_base_facing_height
                    + center_cabinet_door_height / 2
                )
            )
            oven_cabinet_hinge_world_pose = (
                cabinet_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-oven_depth / 2, y=center_front_width / 2
                )
            )
            oven_cabinet_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name="oven_cabinet_hinge",
                world_root_T_self=oven_cabinet_hinge_world_pose,
                parent_connection_specification=Hinge.parent_connection_specification(
                    axis=Vector3.Z(),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](
                            position=0.0, velocity=-hinged_door_velocity_limit
                        ),
                        upper=DerivativeMap[float](
                            position=np.pi / 2, velocity=hinged_door_velocity_limit
                        ),
                    ),
                ),
            )
            oven_cabinet_door = Door.create_with_new_body_in_world(
                world=world,
                name="oven_cabinet_door",
                world_root_T_self=oven_cabinet_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-center_front_width / 2
                ),
                scale=Scale(
                    x=center_door_thickness,
                    y=center_front_width,
                    z=center_cabinet_door_height,
                ),
            )
            for shape in oven_cabinet_door.root.visual.shapes:
                shape.color = Color.WHITE()
            oven_cabinet_door.add(oven_cabinet_hinge)
            tower.add(oven_cabinet_door)

            oven_cabinet_handle = Handle.get_annotation_specification(
                "oven_cabinet_handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=standard_handle_depth,
                        y=center_handle_width,
                        z=standard_handle_height,
                    ),
                    thickness=standard_handle_height,
                ),
            ).spawn(
                world,
                parent_T_self=oven_cabinet_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-center_door_thickness / 2,
                    y=-center_front_width / 2,
                    z=center_cabinet_door_height / 2 - 0.05,
                ),
            )
            for shape in oven_cabinet_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            oven_cabinet_door.add(oven_cabinet_handle)

            # Center: Middle Drawer
            drawer_pose = tower_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=(
                    -oven_depth / 2
                    - center_door_thickness / 2
                    + center_drawer_depth / 2
                ),
                z=(
                    -oven_height / 2
                    + oven_base_facing_height
                    + center_cabinet_door_height
                    + center_cabinet_drawer_gap
                    + center_drawer_height / 2
                ),
            )
            drawer = Drawer.create_with_new_body_in_world(
                world=world,
                name="oven_center_drawer",
                world_root_T_self=drawer_pose,
                scale=Scale(
                    x=center_drawer_depth,
                    y=center_front_width,
                    z=center_drawer_height,
                ),
            )

            slider = Slider.create_with_new_body_in_world(
                world=world,
                name="oven_center_drawer_slider",
                world_root_T_self=drawer_pose,
                parent_connection_specification=Slider.parent_connection_specification(
                    axis=Vector3.NEGATIVE_X(),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](
                            position=0.0, velocity=-sliding_drawer_velocity_limit
                        ),
                        upper=DerivativeMap[float](
                            position=0.25, velocity=sliding_drawer_velocity_limit
                        ),
                    ),
                ),
            )
            drawer.add(slider)

            for shape in drawer.root.visual.shapes:
                shape.color = Color.WHITE()
            tower.add(drawer)

            oven_center_drawer_handle = Handle.get_annotation_specification(
                "oven_center_drawer_handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=standard_handle_depth,
                        y=center_handle_width,
                        z=standard_handle_height,
                    ),
                    thickness=standard_handle_height,
                ),
            ).spawn(
                world,
                parent_T_self=drawer_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-center_drawer_depth / 2,
                    z=center_drawer_height / 2 - 0.05,
                ),
            )
            for shape in oven_center_drawer_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            drawer.add(oven_center_drawer_handle)

            # Center: Oven (Top)
            oven_pose = tower_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(
                z=(
                    -oven_height / 2
                    + oven_base_facing_height
                    + center_cabinet_door_height
                    + center_cabinet_drawer_gap
                    + center_drawer_height
                    + center_oven_height / 2
                )
            )
            oven = Oven.create_with_new_body_in_world(
                world=world,
                name="oven",
                world_root_T_self=oven_pose,
                scale=Scale(x=oven_depth, y=center_width, z=center_oven_height),
            )
            for shape in oven.root.visual.shapes:
                shape.color = Color.GRAY()
            tower.add_object(oven)

            oven_hinge_world_pose = (
                oven_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-oven_depth / 2, z=-center_oven_height / 2
                )
            )
            oven_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name="oven_hinge",
                world_root_T_self=oven_hinge_world_pose,
                parent_connection_specification=Hinge.parent_connection_specification(
                    axis=Vector3.NEGATIVE_Y(),
                    dof_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](
                            position=0.0, velocity=-hinged_door_velocity_limit
                        ),
                        upper=DerivativeMap[float](
                            position=np.pi / 2, velocity=hinged_door_velocity_limit
                        ),
                    ),
                ),
            )
            oven_door = Door.create_with_new_body_in_world(
                world=world,
                name="oven_door",
                world_root_T_self=oven_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=center_oven_height / 2
                ),
                scale=Scale(x=0.02, y=center_width, z=center_oven_height),
            )
            for shape in oven_door.root.visual.shapes:
                shape.color = Color.WHITE()

            oven_door.add(oven_hinge)
            oven.add(oven_door)

            oven_handle_height = 0.025
            oven_handle_top_inset = 0.13  # Measured from the top edge of the oven.
            oven_handle = Handle.get_annotation_specification(
                "oven_handle",
                Handle.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(
                        x=0.04,
                        y=center_handle_width,
                        z=oven_handle_height,
                    ),
                    thickness=0.02,
                ),
            ).spawn(
                world,
                parent_T_self=oven_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.02,
                    z=(
                        center_oven_height
                        - oven_handle_top_inset
                        - oven_handle_height / 2
                    ),
                ),
            )
            for shape in oven_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            oven_door.add(oven_handle)

            # --- SIDEBOARD / KITCHEN ISLAND ---
            sideboard_length, sideboard_width, sideboard_height = 2.250, 0.597, 0.825
            sideboard_top_length, sideboard_top_width = 2.447, 0.796
            sideboard_top_thickness = 0.026
            sideboard_top_left_overhang = 0.098
            sideboard_top_back_overhang = 0.104
            sideboard_top_right_overhang = (
                sideboard_top_length
                - sideboard_length
                - sideboard_top_left_overhang
            )
            sideboard_top_front_overhang = (
                sideboard_top_width
                - sideboard_width
                - sideboard_top_back_overhang
            )
            sideboard_top_x_offset = (
                sideboard_top_back_overhang - sideboard_top_front_overhang
            ) / 2
            sideboard_top_y_offset = (
                sideboard_top_left_overhang - sideboard_top_right_overhang
            ) / 2
            sideboard_cooktop_width = 0.572
            sideboard_cooktop_depth = 0.502
            sideboard_cooktop_right_gap = 0.139
            sideboard_cooktop_front_gap = 0.145
            sideboard_cooktop_x_offset = (
                sideboard_top_x_offset
                - sideboard_top_width / 2
                + sideboard_cooktop_front_gap
                + sideboard_cooktop_depth / 2
            )
            sideboard_cooktop_y_offset = (
                sideboard_top_y_offset
                - sideboard_top_length / 2
                + sideboard_cooktop_right_gap
                + sideboard_cooktop_width / 2
            )
            sideboard_base_facing_height = 0.099
            sideboard_base_facing_setback = 0.07
            sideboard_front_panel_thickness = 0.02
            sideboard_corpus_wall_thickness = 0.02
            sideboard_drawer_height = 0.288
            sideboard_drawer_depth = 0.4
            sideboard_drawer_face_plate_gap = 0.001
            sideboard_upper_face_plate_height = 0.142
            sideboard_upper_face_plate_top_gap = (
                sideboard_height
                - sideboard_base_facing_height
                - 2 * sideboard_drawer_height
                - sideboard_drawer_face_plate_gap
                - sideboard_upper_face_plate_height
            )
            sideboard_outer_margin = 0.025
            sideboard_outer_column_width = 0.595
            sideboard_middle_column_width = 0.994
            sideboard_column_gap = (
                sideboard_length
                - 2 * sideboard_outer_margin
                - 2 * sideboard_outer_column_width
                - sideboard_middle_column_width
            ) / 2
            sideboard_column_widths = [
                sideboard_outer_column_width,
                sideboard_middle_column_width,
                sideboard_outer_column_width,
            ]
            sideboard_column_y_offsets = [
                -sideboard_middle_column_width / 2
                - sideboard_column_gap
                - sideboard_outer_column_width / 2,
                0,
                sideboard_middle_column_width / 2
                + sideboard_column_gap
                + sideboard_outer_column_width / 2,
            ]
            sideboard_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.545,
                y=0.203 + sideboard_width / 2,
                z=sideboard_height / 2,
                yaw=np.pi / 2,
            )

            sideboard = Table.create_with_new_body_in_world(
                world=world,
                name="sideboard",
                world_root_T_self=sideboard_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=sideboard_top_x_offset,
                    y=sideboard_top_y_offset,
                    z=sideboard_height / 2 + sideboard_top_thickness / 2,
                ),
                scale=Scale(
                    sideboard_top_width,
                    sideboard_top_length,
                    sideboard_top_thickness,
                ),
            )
            for shape in sideboard.root.visual.shapes:
                shape.color = Color.WHITE()

            sideboard_cabinet = Cabinet.get_annotation_specification(
                "sideboard_cabinet",
                Cabinet.get_default_root_kinematic_structure_entity_specification(
                    scale=Scale(sideboard_width, sideboard_length, sideboard_height),
                    wall_thickness=sideboard_corpus_wall_thickness,
                ),
            ).spawn(world, parent_T_self=sideboard_pose)
            for shape in sideboard_cabinet.root.visual.shapes:
                shape.color = Color.WHITE()

            sideboard_base_facing = WallPanel.create_with_new_body_in_world(
                world=world,
                name="sideboard_base_facing",
                world_root_T_self=sideboard_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=(
                        -sideboard_width / 2
                        + sideboard_base_facing_setback
                        + sideboard_front_panel_thickness / 2
                    ),
                    z=(
                        -sideboard_height / 2
                        + sideboard_base_facing_height / 2
                    ),
                ),
                scale=Scale(
                    x=sideboard_front_panel_thickness,
                    y=sideboard_length - sideboard_corpus_wall_thickness,
                    z=sideboard_base_facing_height,
                ),
            )
            for shape in sideboard_base_facing.root.visual.shapes:
                shape.color = Color.GRAY()
            sideboard_cabinet.add_object(sideboard_base_facing)

            sideboard_cooktop = Cooktop.create_with_new_body_in_world(
                world=world,
                name="sideboard_cooktop",
                world_root_T_self=sideboard_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=sideboard_cooktop_x_offset,
                    y=sideboard_cooktop_y_offset,
                    z=sideboard_height / 2 + sideboard_top_thickness + 0.0025,
                ),
                scale=Scale(
                    x=sideboard_cooktop_depth,
                    y=sideboard_cooktop_width,
                    z=0.005,
                ),
            )
            for shape in sideboard_cooktop.root.visual.shapes:
                shape.color = Color.BLACK()
            sideboard.add_object(sideboard_cooktop)

            sideboard_drawer_bottom_height = (
                -sideboard_height / 2 + sideboard_base_facing_height
            )
            z_offsets = [
                sideboard_drawer_bottom_height + sideboard_drawer_height / 2,
                sideboard_drawer_bottom_height + 3 * sideboard_drawer_height / 2,
            ]

            for column_index, (width, y_offset) in enumerate(
                zip(sideboard_column_widths, sideboard_column_y_offsets)
            ):
                upper_face_plate = WallPanel.create_with_new_body_in_world(
                    world=world,
                    name=f"sideboard_upper_face_plate_{column_index}",
                    world_root_T_self=sideboard_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=(
                            -sideboard_width / 2
                            + sideboard_front_panel_thickness / 2
                        ),
                        y=y_offset,
                        z=(
                            sideboard_height / 2
                            - sideboard_upper_face_plate_top_gap
                            - sideboard_upper_face_plate_height / 2
                        ),
                    ),
                    scale=Scale(
                        x=sideboard_front_panel_thickness,
                        y=width,
                        z=sideboard_upper_face_plate_height,
                    ),
                )
                for shape in upper_face_plate.root.visual.shapes:
                    shape.color = Color.WHITE()
                sideboard_cabinet.add_object(upper_face_plate)

                for row_index, z_offset in enumerate(z_offsets):
                    drawer_id = f"sideboard_drawer_{column_index}_{row_index}"
                    drawer_pose = (
                        sideboard_pose
                        @ HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=-sideboard_width / 2 + sideboard_drawer_depth / 2,
                            y=y_offset,
                            z=z_offset,
                        )
                    )

                    drawer = Drawer.create_with_new_body_in_world(
                        world=world,
                        name=drawer_id,
                        world_root_T_self=drawer_pose,
                        scale=Scale(
                            sideboard_drawer_depth,
                            width,
                            sideboard_drawer_height,
                        ),
                    )

                    slider = Slider.create_with_new_body_in_world(
                        world=world,
                        name=f"{drawer_id}_slider",
                        world_root_T_self=drawer_pose,
                        parent_connection_specification=Slider.parent_connection_specification(
                            axis=Vector3.NEGATIVE_X(),
                            dof_limits=DegreeOfFreedomLimits(
                                lower=DerivativeMap[float](
                                    position=0.0,
                                    velocity=-sliding_drawer_velocity_limit,
                                ),
                                upper=DerivativeMap[float](
                                    position=0.25,
                                    velocity=sliding_drawer_velocity_limit,
                                ),
                            ),
                        ),
                    )
                    drawer.add(slider)

                    for shape in drawer.root.visual.shapes:
                        shape.color = Color.WHITE()
                    sideboard_cabinet.add(drawer)

                    handle_pose = (
                        drawer_pose
                        @ HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=-sideboard_drawer_depth / 2,
                            z=sideboard_drawer_height / 2 - 0.05,
                        )
                    )
                    handle = Handle.get_annotation_specification(
                        f"{drawer_id}_handle",
                        Handle.get_default_root_kinematic_structure_entity_specification(
                            scale=Scale(
                                standard_handle_depth,
                                width - 0.1,
                                standard_handle_height,
                            ),
                            thickness=standard_handle_height,
                        ),
                    ).spawn(world, parent_T_self=handle_pose)
                    for shape in handle.root.visual.shapes:
                        shape.color = Color.GRAY()
                    drawer.add(handle)

            # --- SOFA ---
            sofa = Sofa.create_with_new_body_in_world(
                world=world,
                name="sofa",
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=3.60, y=1.601, z=0.34, yaw=4.7124
                ),
                scale=Scale(x=0.94, y=1.68, z=0.68),
            )
            for color in sofa.bodies[0].visual.shapes:
                color.color = Color.BEIGE()

            # --- COFFEE TABLE ---
            self._build_coffee_table(world)

            # --- CUPBOARD ---
            cupboard_scale = Scale(0.43, 0.80, 2.02)
            cupboard_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=4.55, y=4.72, z=1.01
            )
            cupboard = Cupboard.get_annotation_specification(
                "cupboard",
                Cupboard.get_default_root_kinematic_structure_entity_specification(
                    scale=cupboard_scale, wall_thickness=0.02
                ),
            ).spawn(world, parent_T_self=cupboard_pose)

            shelf_scale = Scale(0.40, 0.76, 0.02)
            for i, z in enumerate([-0.5, 0.5]):
                shelf = ShelfLayer.create_with_new_body_in_world(
                    world=world,
                    name=f"cupboard_shelf_{i}",
                    world_root_T_self=cupboard_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(z=z),
                    scale=shelf_scale,
                )
                for shape in shelf.root.visual.shapes:
                    shape.color = Color.WHITE()
                cupboard.add_object(shelf)

            cupboard_door_height, cupboard_door_width = 1.055, 0.40
            cupboard_door_z_relative = -(cupboard_scale.z / 2) + (
                cupboard_door_height / 2
            )
            cupboard_door_x_relative = -(cupboard_scale.x / 2) - 0.01
            cupboard_door_scale = Scale(0.02, cupboard_door_width, cupboard_door_height)

            for i, (side, limits, y_off) in enumerate(
                [("left", [0.0, np.pi / 2], -0.40), ("right", [-np.pi / 2, 0.0], 0.40)]
            ):
                handle_pose = (
                    cupboard_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=cupboard_door_x_relative, y=y_off, z=cupboard_door_z_relative
                    )
                )
                hinge = Hinge.create_with_new_body_in_world(
                    world=world,
                    name=f"cupboard_hinge_{side}",
                    world_root_T_self=handle_pose,
                    parent_connection_specification=Hinge.parent_connection_specification(
                        axis=Vector3.Z(),
                        dof_limits=DegreeOfFreedomLimits(
                            lower=DerivativeMap[float](
                                position=limits[0],
                                velocity=-hinged_door_velocity_limit,
                            ),
                            upper=DerivativeMap[float](
                                position=limits[1],
                                velocity=hinged_door_velocity_limit,
                            ),
                        ),
                    ),
                )
                door = Door.create_with_new_body_in_world(
                    world=world,
                    name=f"cupboard_door_{side}",
                    world_root_T_self=handle_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=0.2 if side == "left" else -0.2
                    ),
                    scale=cupboard_door_scale,
                )
                for shape in door.root.visual.shapes:
                    shape.color = Color.WHITE()
                door.add(hinge)
                cupboard.add(door)

                handle = Handle.get_annotation_specification(
                    f"cupboard_handle_{side}",
                    Handle.get_default_root_kinematic_structure_entity_specification(
                        scale=Scale(0.04, 0.04, 0.04),
                        thickness=0.02,
                    ),
                ).spawn(
                    world,
                    parent_T_self=handle_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-0.03, y=0.15 if side == "left" else -0.15
                    ),
                )
                for shape in handle.root.visual.shapes:
                    shape.color = Color.GRAY()
                door.add(handle)

            # --- DESK ---
            desk_length, desk_width, desk_height = 0.60, 1.20, 0.75
            desk_color = Color.WHITE()
            desk_plate_thickness = 0.03
            desk_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.05, y=1.28, z=desk_height
            )
            desk = Desk.create_with_new_body_in_world(
                world=world,
                name="desk",
                world_root_T_self=desk_pose,
                scale=Scale(desk_length, desk_width, desk_plate_thickness),
            )
            for shape in desk.root.visual.shapes:
                shape.color = desk_color

            leg_scale = Scale(0.04, 0.04, desk_height - desk_plate_thickness)
            x_offset, y_offset, z_position = (
                (desk_length / 2) - 0.02,
                (desk_width / 2) - 0.02,
                -(desk_plate_thickness / 2) - (leg_scale.z / 2),
            )
            for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                leg = Leg.create_with_new_body_in_world(
                    world=world,
                    name=f"desk_leg_{i}",
                    world_root_T_self=desk_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=sx * x_offset, y=sy * y_offset, z=z_position
                    ),
                    scale=leg_scale,
                )
                for shape in leg.root.visual.shapes:
                    shape.color = desk_color
                desk.add(leg)

            # --- MODULAR COOKING TABLE ---
            (
                cooking_table_length,
                cooking_table_depth,
                cooking_table_height,
                cooking_table_thickness,
            ) = (1.75, 0.64, 0.71, 0.04)
            cooking_table_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.28, y=5.99, z=cooking_table_height
            )
            cooking_table = Table.create_with_new_body_in_world(
                world=world,
                name="cooking_table",
                world_root_T_self=cooking_table_pose,
                scale=Scale(
                    cooking_table_length, cooking_table_depth, cooking_table_thickness
                ),
            )
            for shape in cooking_table.bodies[0].visual.shapes:
                shape.color = Color.BEIGE()

            cooking_table_cooktop = Cooktop.create_with_new_body_in_world(
                world=world,
                name="cooktop",
                world_root_T_self=cooking_table_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=cooking_table_thickness / 2 + 0.005
                ),
                scale=Scale(0.5, 0.5, 0.01),
            )
            for shape in cooking_table_cooktop.root.visual.shapes:
                shape.color = Color.BLACK()
            cooking_table.add_object(cooking_table_cooktop)

            cooking_table_bottom = ShelfLayer.create_with_new_body_in_world(
                world=world,
                name="cooking_table_bottom",
                world_root_T_self=cooking_table_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-cooking_table_height + cooking_table_thickness
                ),
                scale=Scale(
                    cooking_table_length, cooking_table_depth, cooking_table_thickness
                ),
            )
            for shape in cooking_table_bottom.root.visual.shapes:
                shape.color = Color.BEIGE()
            cooking_table.add_object(cooking_table_bottom)

            module_width = (cooking_table_length - 0.60) / 2
            for side in [-1, 1]:
                side_name = "left" if side == -1 else "right"
                module_pose = (
                    cooking_table_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=side * (0.265 + module_width / 2),
                        z=-cooking_table_height / 2 + cooking_table_thickness,
                        yaw=1.5708,
                    )
                )
                mod = Cupboard.create_with_new_body_in_world(
                    name=f"cooking_mod_{side_name}",
                    world=world,
                    world_root_T_self=module_pose,
                    scale=Scale(
                        module_width,
                        cooking_table_depth,
                        cooking_table_height - 2 * cooking_table_thickness,
                    ),
                )
                for shape in mod.bodies[0].visual.shapes:
                    shape.color = Color.BEIGE()
                cooking_table.add_object(mod)

                drawer_pose = (
                    module_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(z=0.2)
                )
                drawer = Drawer.create_with_new_body_in_world(
                    world=world,
                    name=f"cooking_drawer_{side_name}",
                    world_root_T_self=drawer_pose,
                    scale=Scale(module_width - 0.04, cooking_table_depth - 0.02, 0.18),
                )

                slider = Slider.create_with_new_body_in_world(
                    world=world,
                    name=f"cooking_drawer_{side_name}_slider",
                    world_root_T_self=drawer_pose,
                    parent_connection_specification=Slider.parent_connection_specification(
                        axis=Vector3.NEGATIVE_X(),
                        dof_limits=DegreeOfFreedomLimits(
                            lower=DerivativeMap[float](
                                position=0.0, velocity=-sliding_drawer_velocity_limit
                            ),
                            upper=DerivativeMap[float](
                                position=0.40, velocity=sliding_drawer_velocity_limit
                            ),
                        ),
                    ),
                )
                drawer.add(slider)

                for shape in drawer.root.visual.shapes:
                    shape.color = Color.BEIGE()
                mod.add(drawer)

                handle_pose = (
                    drawer_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-module_width / 2 + 0.02
                    )
                )
                handle = Handle.get_annotation_specification(
                    f"cooking_drawer_handle_{side_name}",
                    Handle.get_default_root_kinematic_structure_entity_specification(
                        scale=Scale(0.04, module_width / 3, 0.04),
                        thickness=0.02,
                    ),
                ).spawn(world, parent_T_self=handle_pose)
                for shape in handle.root.visual.shapes:
                    shape.color = Color.GRAY()
                drawer.add(handle)

            # --- DINING TABLE ---
            (
                dining_table_length,
                dining_table_width,
                dining_table_height,
                dining_table_plate_thickness,
            ) = (0.73, 1.18, 0.76, 0.04)
            dining_table_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=2.59975, y=5.705, z=dining_table_height
            )
            dining_table = DiningTable.create_with_new_body_in_world(
                world=world,
                name="dining_table",
                world_root_T_self=dining_table_pose,
                scale=Scale(
                    dining_table_length,
                    dining_table_width,
                    dining_table_plate_thickness,
                ),
            )
            for shape in dining_table.root.visual.shapes:
                shape.color = Color.BEIGE()

            leg_scale = Scale(
                0.06, 0.06, dining_table_height - dining_table_plate_thickness
            )
            x_offset, y_offset, z_position = (
                (dining_table_length / 2) - 0.03,
                (dining_table_width / 2) - 0.03,
                -(dining_table_plate_thickness / 2) - (leg_scale.z / 2),
            )
            for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                leg = Leg.create_with_new_body_in_world(
                    world=world,
                    name=f"dining_table_leg_{i}",
                    world_root_T_self=dining_table_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=sx * x_offset, y=sy * y_offset, z=z_position
                    ),
                    scale=leg_scale,
                )
                for shape in leg.root.visual.shapes:
                    shape.color = Color.BEIGE()
                dining_table.add(leg)
        return world

    def _build_coffee_table(self, world):
        """
        Builds a refined coffee table with a middle shelf and a closed front.

        This design uses a white, front-closed structure with a floor plate to match the
        physical appearance of the coffee table in the target environment. It is
        constructed from multiple boxes to allow for detailed semantic labeling of its
        components (shelf, floor, walls).
        """
        length, width, height = 0.37, 0.91, 0.44
        thick, color = 0.02, Color.WHITE()
        pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=4.22, y=2.621, z=height, yaw=np.pi
        )

        table = Table.create_with_new_body_in_world(
            world=world,
            name="coffee_table",
            world_root_T_self=pose,
            scale=Scale(length, width, thick),
        )
        for shape in table.bodies[0].visual.shapes:
            shape.color = color

        shelf = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name="coffee_table_shelf",
            world_root_T_self=pose
            @ HomogeneousTransformationMatrix.from_xyz_rpy(z=-height / 2),
            scale=Scale(length, width, 0.01),
        )
        for shape in shelf.root.visual.shapes:
            shape.color = color
        table.add_object(shelf)

        floor = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name="coffee_table_floor",
            world_root_T_self=pose
            @ HomogeneousTransformationMatrix.from_xyz_rpy(z=-height + thick / 2),
            scale=Scale(length, width, thick),
        )
        for shape in floor.root.visual.shapes:
            shape.color = color
        table.add_object(floor)

        for i, y_direction in enumerate([-1, 1]):
            side_wall_body = Body(
                name=PrefixedName(f"coffee_table_wall_short_{i}_body")
            )
            side_wall_box = Box(scale=Scale(length, thick, height), color=color)
            side_wall_box.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=side_wall_body
            )
            side_wall_geometry = ShapeCollection(
                [side_wall_box], reference_frame=side_wall_body
            )
            side_wall_body.collision, side_wall_body.visual = (
                side_wall_geometry,
                side_wall_geometry,
            )
            world.add_connection(
                FixedConnection(
                    parent=table.root,
                    child=side_wall_body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=y_direction * (width / 2 - thick / 2), z=-height / 2
                    ),
                )
            )

        wall_length = width / 3
        for side in [-1, 1]:
            side_name = "left" if side == -1 else "right"
            long_wall_body = Body(
                name=PrefixedName(f"coffee_table_wall_long_{side_name}_body")
            )
            long_wall_box = Box(scale=Scale(thick, wall_length, height), color=color)
            long_wall_box.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=long_wall_body
            )
            long_wall_geometry = ShapeCollection(
                [long_wall_box], reference_frame=long_wall_body
            )
            long_wall_body.collision, long_wall_body.visual = (
                long_wall_geometry,
                long_wall_geometry,
            )
            world.add_connection(
                FixedConnection(
                    parent=table.root,
                    child=long_wall_body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=side * (length / 2 - thick / 2),
                        y=width / 2 - wall_length / 2,
                        z=-height / 2,
                    ),
                )
            )

    def _build_environment_rooms(self, world: World):
        room_annotations = []

        with world.modify_world():
            kitchen_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 3.334, 0),
                Point3(5.214, 3.334, 0),
                Point3(5.214, 0, 0),
            ]

            living_room_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 2.971, 0),
                Point3(5.214, 2.971, 0),
                Point3(5.214, 0, 0),
            ]

            bed_room_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 2.67, 0.0),
                Point3(2.50, 2.67, 0.0),
                Point3(2.50, 0, 0.0),
            ]

            office_floor_polytope = [
                Point3(0, 0, 0),
                Point3(0, 2.67, 0),
                Point3(2.71, 2.67, 0),
                Point3(2.71, 0, 0),
            ]

            kitchen_floor = Floor.create_with_new_body_from_polytope_in_world(
                name="kitchen_floor",
                world=world,
                floor_polytope=kitchen_floor_polytope,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.317, y=-0.843
                ),
            )
            kitchen = Room(floor=kitchen_floor, name=PrefixedName("kitchen"))
            room_annotations.append(kitchen)

            living_room_floor = Floor.create_with_new_body_from_polytope_in_world(
                name="living_room_floor",
                world=world,
                floor_polytope=living_room_floor_polytope,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.317, y=2.3095
                ),
            )
            living_room = Room(
                floor=living_room_floor, name=PrefixedName("living_room")
            )
            room_annotations.append(living_room)

            bed_room_floor = Floor.create_with_new_body_from_polytope_in_world(
                name="bed_room_floor",
                world=world,
                floor_polytope=bed_room_floor_polytope,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.96, y=4.96
                ),
            )
            bed_room = Room(floor=bed_room_floor, name=PrefixedName("bed_room"))
            room_annotations.append(bed_room)

            office_floor = Floor.create_with_new_body_from_polytope_in_world(
                name="office_floor",
                world=world,
                floor_polytope=office_floor_polytope,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=3.56, y=4.96
                ),
            )
            office = Room(floor=office_floor, name=PrefixedName("office"))
            room_annotations.append(office)

            world.add_semantic_annotations(room_annotations)

        return world
