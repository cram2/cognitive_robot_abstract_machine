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
        world = World()
        root = Body(name=PrefixedName("root"))
        with world.modify_world():
            world.add_body(root)

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
                name=PrefixedName("south_wall1"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=-2.01),
                scale=Scale(x=0.05, y=1.00, z=3.00),
            )

            south_wall2 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall2"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-1.45, yaw=np.pi / 2
                ),
                scale=Scale(x=0.05, y=0.29, z=3.00),
            )

            south_wall3 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall3"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29, y=-0.9925
                ),
                scale=Scale(x=0.05, y=1.085, z=1.00),
            )

            south_wall4 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall4"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-0.45, yaw=np.pi / 2
                ),
                scale=Scale(x=0.05, y=0.29, z=1.00),
            )

            south_wall5 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall5"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=0.45, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 0.29, 1.00),
            )

            south_wall6 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall6"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=1.80
                ),
                scale=Scale(0.05, 2.75, 1.00),
            )

            south_wall7 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall7"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=5.16
                ),
                scale=Scale(0.05, 2.27, 1.00),
            )

            east_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("east_wall"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.462, y=-2.535, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 4.924, 3.00),
            )

            middle_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("middle_wall"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.20975, y=5.00
                ),
                scale=Scale(0.05, 2.67, 1.00),
            )

            west_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("west_wall"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.9345, y=6.32, yaw=np.pi / 2
                ),
                scale=Scale(0.05, 4.449, 3.00),
            )

            north_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("north_wall"),
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
        with world.modify_world():
            # --- TRASH CAN ---
            trash_can = TrashCan.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("trash_can"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.416, y=5.5, z=0.2
                ),
                scale=Scale(x=0.30, y=0.30, z=0.40),
                wall_thickness=0.02,
            )
            for shape in trash_can.root.visual.shapes:
                shape.color = Color.GRAY()

            # --- REFRIGERATOR ---
            fridge_length, fridge_width, fridge_height = 0.60, 0.658, 1.49
            fridge_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.537, y=-2.181, z=fridge_height / 2, yaw=-np.pi / 2
            )

            refrigerator = Fridge.create_with_new_body_in_world(
                name=PrefixedName("refrigerator"),
                world=world,
                world_root_T_self=fridge_pose,
                scale=Scale(x=fridge_length, y=fridge_width, z=fridge_height),
                wall_thickness=0.02,
            )
            for shape in refrigerator.root.visual.shapes:
                shape.color = Color.GRAY()

            door_height = (fridge_height - 0.08) * 0.75
            hinge_local_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-fridge_length / 2,
                y=-fridge_width / 2,
                z=fridge_height / 2 - door_height / 2,
            )
            hinge_world_pose = fridge_pose @ hinge_local_pose
            fridge_door_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("fridge_door_hinge"),
                world_root_T_self=hinge_world_pose,
                active_axis=Vector3.Z(),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap[float](position=0.0),
                    upper=DerivativeMap[float](position=np.pi / 2),
                ),
            )

            fridge_door = Door.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("fridge_door"),
                world_root_T_self=hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(y=fridge_width / 2),
                scale=Scale(x=0.02, y=fridge_width, z=door_height),
            )
            for shape in fridge_door.root.visual.shapes:
                shape.color = Color.WHITE()
            fridge_door.add(fridge_door_hinge)
            refrigerator.add(fridge_door)

            drawer_height = (fridge_height - 0.08) * 0.25
            drawer_world_pose = (
                fridge_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-fridge_length / 2 + 0.25,
                    z=-fridge_height / 2 + 0.08 + drawer_height / 2,
                )
            )
            fridge_drawer = Drawer.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("fridge_drawer"),
                world_root_T_self=drawer_world_pose,
                scale=Scale(x=0.5, y=fridge_width - 0.04, z=drawer_height - 0.01),
            )

            fridge_slider = Slider.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("fridge_drawer_slider"),
                world_root_T_self=drawer_world_pose,
                active_axis=Vector3.NEGATIVE_X(),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap[float](position=0.0),
                    upper=DerivativeMap[float](position=0.5),
                ),
            )

            fridge_drawer.add(fridge_slider)

            for shape in fridge_drawer.root.visual.shapes:
                shape.color = Color.WHITE()
            refrigerator.add(fridge_drawer)

            handle_depth, handle_thickness = 0.04, 0.02
            door_handle_world_pose = (
                hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.02, y=fridge_width / 2 - 0.03, roll=np.pi / 2
                )
            )
            fridge_door_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("fridge_door_handle"),
                world_root_T_self=door_handle_world_pose,
                scale=Scale(x=handle_depth, y=0.5, z=handle_thickness),
                thickness=handle_thickness,
            )
            for shape in fridge_door_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            fridge_door.add(fridge_door_handle)

            drawer_handle_world_pose = (
                drawer_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.26, z=drawer_height / 2 - 0.03
                )
            )
            fridge_drawer_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("fridge_drawer_handle"),
                world_root_T_self=drawer_handle_world_pose,
                scale=Scale(x=0.04, y=0.5, z=0.02),
                thickness=0.02,
            )
            for shape in fridge_drawer_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            fridge_drawer.add(fridge_drawer_handle)

            # --- KITCHEN COUNTER ---
            counter_top_length, counter_top_depth, counter_top_height = (
                2.044,
                0.658,
                0.6,
            )
            counter_top_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.887, y=-2.181, z=counter_top_height / 2, yaw=-np.pi / 2
            )

            counter_top = CounterTop.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("counter_top"),
                world_root_T_self=counter_top_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=counter_top_height / 2 + 0.02
                ),
                scale=Scale(x=counter_top_depth, y=counter_top_length, z=0.04),
            )
            for shape in counter_top.root.visual.shapes:
                shape.color = Color.BEIGE()

            sink = Sink.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("sink"),
                world_root_T_self=counter_top_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-0.7, z=counter_top_height / 2 + 0.045
                ),
                scale=Scale(x=0.4, y=0.6, z=0.005),
            )
            for shape in sink.root.visual.shapes:
                shape.color = Color.BLACK()
            counter_top.add(sink)

            module_1_width, module_2_width = 0.60, 0.55
            module_3_width = counter_top_length - module_1_width - module_2_width

            # Module 1: Cabinet
            module_1_pose = (
                counter_top_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-counter_top_length / 2 + module_1_width / 2
                )
            )
            module_1_cabinet = Cabinet.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("module_1_cabinet"),
                world_root_T_self=module_1_pose,
                scale=Scale(
                    x=counter_top_depth, y=module_1_width, z=counter_top_height
                ),
                wall_thickness=0.02,
            )
            for shape in module_1_cabinet.root.visual.shapes:
                shape.color = Color.GRAY()

            module_1_hinge_world_pose = (
                module_1_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-counter_top_depth / 2, y=-module_1_width / 2
                )
            )
            module_1_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("module_1_hinge"),
                world_root_T_self=module_1_hinge_world_pose,
                active_axis=Vector3.Z(),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap[float](position=0.0),
                    upper=DerivativeMap[float](position=np.pi / 2),
                ),
            )
            module_1_door = Door.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("module_1_door"),
                world_root_T_self=module_1_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(y=module_1_width / 2),
                scale=Scale(x=0.02, y=module_1_width, z=counter_top_height),
            )
            for shape in module_1_door.root.visual.shapes:
                shape.color = Color.WHITE()
            module_1_door.add(module_1_hinge)
            module_1_cabinet.add(module_1_door)

            module_1_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("module_1_handle"),
                world_root_T_self=module_1_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.02, y=module_1_width - 0.05, z=counter_top_height / 2 - 0.05
                ),
                scale=Scale(x=0.04, y=module_1_width - 0.06, z=0.02),
                thickness=0.02,
            )
            for shape in module_1_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            module_1_door.add(module_1_handle)

            # Module 2: Dishwasher
            module_2_pose = (
                counter_top_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-counter_top_length / 2 + module_1_width + module_2_width / 2
                )
            )
            dishwasher = Dishwasher.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("dishwasher"),
                world_root_T_self=module_2_pose,
                scale=Scale(
                    x=counter_top_depth, y=module_2_width, z=counter_top_height
                ),
                wall_thickness=0.02,
            )
            for shape in dishwasher.root.visual.shapes:
                shape.color = Color.GRAY()

            module_2_hinge_world_pose = (
                module_2_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-counter_top_depth / 2, z=-counter_top_height / 2
                )
            )
            module_2_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("dishwasher_hinge"),
                world_root_T_self=module_2_hinge_world_pose,
                active_axis=Vector3.NEGATIVE_Y(),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap[float](position=0.0),
                    upper=DerivativeMap[float](position=np.pi / 2),
                ),
            )
            module_2_door = Door.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("dishwasher_door"),
                world_root_T_self=module_2_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=counter_top_height / 2
                ),
                scale=Scale(x=0.02, y=module_2_width, z=counter_top_height),
            )
            for shape in module_2_door.root.visual.shapes:
                shape.color = Color.WHITE()
            module_2_door.add(module_2_hinge)
            dishwasher.add(module_2_door)

            module_2_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("dishwasher_handle"),
                world_root_T_self=module_2_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.02, z=counter_top_height - 0.03, y=module_2_width / 2
                ),
                scale=Scale(x=0.04, y=module_2_width - 0.06, z=0.02),
                thickness=0.02,
            )
            for shape in module_2_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            module_2_door.add(module_2_handle)

            # Module 3: Cabinet with Drawers
            module_3_pose = (
                counter_top_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=counter_top_length / 2 - module_3_width / 2
                )
            )
            module_3_cabinet = Cabinet.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("module_3_cabinet"),
                world_root_T_self=module_3_pose,
                scale=Scale(
                    x=counter_top_depth, y=module_3_width, z=counter_top_height
                ),
                wall_thickness=0.02,
            )
            for shape in module_3_cabinet.root.visual.shapes:
                shape.color = Color.GRAY()
            counter_top.add_object(module_3_cabinet)

            drawer_heights = [
                counter_top_height * 0.4,
                counter_top_height * 0.4,
                counter_top_height * 0.2,
            ]
            drawer_z_positions = [-0.18, 0.06, 0.24]
            for i, (height, z_pos) in enumerate(
                zip(drawer_heights, drawer_z_positions)
            ):
                drawer_pose = (
                    module_3_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-counter_top_depth / 2 + 0.15, z=z_pos
                    )
                )
                drawer = Drawer.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"counter_drawer_{i}"),
                    world_root_T_self=drawer_pose,
                    scale=Scale(x=0.3, y=module_3_width - 0.04, z=height - 0.01),
                )

                slider = Slider.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"counter_drawer_{i}_slider"),
                    world_root_T_self=drawer_pose,
                    active_axis=Vector3.NEGATIVE_X(),
                    connection_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](position=0.0),
                        upper=DerivativeMap[float](position=0.25),
                    ),
                )
                drawer.add(slider)

                for shape in drawer.root.visual.shapes:
                    shape.color = Color.WHITE()
                module_3_cabinet.add(drawer)

                handle_pose = (
                    drawer_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-0.16, z=height / 2 - 0.03
                    )
                )
                handle = Handle.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"counter_drawer_{i}_handle"),
                    world_root_T_self=handle_pose,
                    scale=Scale(x=0.04, y=module_3_width - 0.06, z=0.02),
                    thickness=0.02,
                )
                for shape in handle.root.visual.shapes:
                    shape.color = Color.GRAY()
                drawer.add(handle)

            # --- OVEN TOWER ---
            oven_width, oven_depth, oven_height = 1.20, 0.658, 1.49
            tower_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.51, y=-2.181, z=oven_height / 2, yaw=-np.pi / 2
            )
            tower = Cupboard.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_tower"),
                world_root_T_self=tower_pose,
                scale=Scale(x=oven_depth, y=oven_width, z=oven_height),
                wall_thickness=0.02,
            )
            for shape in tower.root.visual.shapes:
                shape.color = Color.GRAY()

            center_width, side_width = 0.60, 0.30
            cabinet_height, drawer_height = 0.60, 0.15
            oven_height_center = oven_height - cabinet_height - drawer_height

            # Side Drawers
            for side in [-1, 1]:
                side_name = "left" if side == -1 else "right"
                drawer_pose = tower_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=side * (center_width / 2 + side_width / 2)
                )
                drawer = Drawer.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"oven_side_drawer_{side_name}"),
                    world_root_T_self=drawer_pose,
                    scale=Scale(x=oven_depth, y=side_width, z=oven_height),
                )

                slider = Slider.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"oven_side_drawer_{side_name}_slider"),
                    world_root_T_self=drawer_pose,
                    active_axis=Vector3.NEGATIVE_X(),
                )
                drawer.add(slider)

                for shape in drawer.root.visual.shapes:
                    shape.color = Color.WHITE()
                tower.add(drawer)

                handle_pose = (
                    drawer_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-oven_depth / 2, roll=np.pi / 2
                    )
                )
                handle = Handle.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"oven_side_handle_{side_name}"),
                    world_root_T_self=handle_pose,
                    scale=Scale(x=0.04, y=oven_height - 0.08, z=0.02),
                    thickness=0.02,
                )
                for shape in handle.root.visual.shapes:
                    shape.color = Color.GRAY()
                drawer.add(handle)

            # Center: Bottom Cabinet
            cab_pose = tower_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(
                z=-oven_height / 2 + cabinet_height / 2
            )
            oven_cabinet_hinge_world_pose = (
                cab_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-oven_depth / 2, y=center_width / 2
                )
            )
            oven_cabinet_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_cabinet_hinge"),
                world_root_T_self=oven_cabinet_hinge_world_pose,
                active_axis=Vector3.Z(),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap[float](position=0.0),
                    upper=DerivativeMap[float](position=np.pi / 2),
                ),
            )
            oven_cabinet_door = Door.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_cabinet_door"),
                world_root_T_self=oven_cabinet_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(y=-center_width / 2),
                scale=Scale(x=0.02, y=center_width, z=cabinet_height),
            )
            for shape in oven_cabinet_door.root.visual.shapes:
                shape.color = Color.WHITE()
            oven_cabinet_door.add(oven_cabinet_hinge)
            tower.add(oven_cabinet_door)

            oven_cabinet_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_cabinet_handle"),
                world_root_T_self=oven_cabinet_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.02, y=-center_width + 0.05, z=cabinet_height / 2 - 0.05
                ),
                scale=Scale(x=0.04, y=center_width - 0.06, z=0.02),
                thickness=0.02,
            )
            for shape in oven_cabinet_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            oven_cabinet_door.add(oven_cabinet_handle)

            # Center: Middle Drawer
            drawer_pose = tower_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-oven_depth / 2 + 0.15,
                z=-oven_height / 2 + cabinet_height + drawer_height / 2,
            )
            drawer = Drawer.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_center_drawer"),
                world_root_T_self=drawer_pose,
                scale=Scale(x=0.3, y=center_width - 0.04, z=drawer_height - 0.01),
            )

            slider = Slider.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_center_drawer_slider"),
                world_root_T_self=drawer_pose,
                active_axis=Vector3.NEGATIVE_X(),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap[float](position=0.0),
                    upper=DerivativeMap[float](position=0.25),
                ),
            )
            drawer.add(slider)

            for shape in drawer.root.visual.shapes:
                shape.color = Color.WHITE()
            tower.add(drawer)

            # Center: Oven (Top)
            oven_pose = tower_pose @ HomogeneousTransformationMatrix.from_xyz_rpy(
                z=oven_height / 2 - oven_height_center / 2
            )
            oven = Oven.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven"),
                world_root_T_self=oven_pose,
                scale=Scale(x=oven_depth, y=center_width, z=oven_height_center),
            )
            for shape in oven.root.visual.shapes:
                shape.color = Color.GRAY()
            tower.add_object(oven)

            oven_hinge_world_pose = (
                oven_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-oven_depth / 2, z=-oven_height_center / 2
                )
            )
            oven_hinge = Hinge.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_hinge"),
                world_root_T_self=oven_hinge_world_pose,
                active_axis=Vector3.NEGATIVE_Y(),
                connection_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap[float](position=0.0),
                    upper=DerivativeMap[float](position=np.pi / 2),
                ),
            )
            oven_door = Door.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_door"),
                world_root_T_self=oven_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=oven_height_center / 2
                ),
                scale=Scale(x=0.02, y=center_width, z=oven_height_center),
            )
            for shape in oven_door.root.visual.shapes:
                shape.color = Color.WHITE()

            oven_door.add(oven_hinge)
            oven.add(oven_door)

            oven_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven_handle"),
                world_root_T_self=oven_hinge_world_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.02, y=center_width / 2, z=oven_height_center - 0.05
                ),
                scale=Scale(x=0.04, y=center_width - 0.06, z=0.02),
                thickness=0.02,
            )
            for shape in oven_handle.root.visual.shapes:
                shape.color = Color.GRAY()
            oven_door.add(oven_handle)

            # --- SIDEBOARD / KITCHEN ISLAND ---
            sideboard_length, sideboard_width, sideboard_height = 2.45, 0.796, 0.845
            sideboard_thickness = 0.04
            sideboard_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.545, y=0.2, z=sideboard_height / 2, yaw=np.pi / 2
            )

            sideboard = Table.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("sideboard"),
                world_root_T_self=sideboard_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=sideboard_height / 2 - sideboard_thickness / 2
                ),
                scale=Scale(sideboard_width, sideboard_length, sideboard_thickness),
            )
            for shape in sideboard.root.visual.shapes:
                shape.color = Color.WHITE()

            sideboard_cabinet = Cabinet.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("sideboard_cabinet"),
                world_root_T_self=sideboard_pose,
                scale=Scale(sideboard_width, sideboard_length, sideboard_height),
                wall_thickness=0.02,
            )
            for shape in sideboard_cabinet.root.visual.shapes:
                shape.color = Color.WHITE()

            sideboard_cooktop = Cooktop.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("sideboard_cooktop"),
                world_root_T_self=sideboard_pose
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-0.7, z=sideboard_height / 2 + 0.005
                ),
                scale=Scale(x=0.5, y=0.6, z=0.005),
            )
            for shape in sideboard_cooktop.root.visual.shapes:
                shape.color = Color.BLACK()
            sideboard.add_object(sideboard_cooktop)

            # Define widths for the three sections of the sideboard
            width_outer, width_middle = sideboard_length * 0.3, sideboard_length * 0.4
            widths = [width_outer, width_middle, width_outer]

            # Correctly calculate y-offsets for each section
            y_offsets = [
                -sideboard_length / 2 + width_outer / 2,
                0,
                sideboard_length / 2 - width_outer / 2,
            ]

            sideboard_drawer_height = (sideboard_height - 0.15) / 2
            z_offsets = [
                -sideboard_height / 2 + 0.05 + sideboard_drawer_height / 2,
                -sideboard_height / 2 + 0.05 + 3 * sideboard_drawer_height / 2,
            ]

            for column_index, (width, y_offset) in enumerate(zip(widths, y_offsets)):
                for row_index, z_offset in enumerate(z_offsets):
                    drawer_id = f"sideboard_drawer_{column_index}_{row_index}"
                    drawer_pose = (
                        sideboard_pose
                        @ HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=-sideboard_width / 2 + 0.2, y=y_offset, z=z_offset
                        )
                    )

                    drawer = Drawer.create_with_new_body_in_world(
                        world=world,
                        name=PrefixedName(drawer_id),
                        world_root_T_self=drawer_pose,
                        scale=Scale(0.4, width - 0.01, sideboard_drawer_height - 0.01),
                    )

                    slider = Slider.create_with_new_body_in_world(
                        world=world,
                        name=PrefixedName(f"{drawer_id}_slider"),
                        world_root_T_self=drawer_pose,
                        active_axis=Vector3.NEGATIVE_X(),
                        connection_limits=DegreeOfFreedomLimits(
                            lower=DerivativeMap[float](position=0.0),
                            upper=DerivativeMap[float](position=0.25),
                        ),
                    )
                    drawer.add(slider)

                    for shape in drawer.root.visual.shapes:
                        shape.color = Color.WHITE()
                    sideboard_cabinet.add(drawer)

                    handle_pose = (
                        drawer_pose
                        @ HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=-0.2, z=sideboard_drawer_height / 2 - 0.05
                        )
                    )
                    handle = Handle.create_with_new_body_in_world(
                        world=world,
                        name=PrefixedName(f"{drawer_id}_handle"),
                        world_root_T_self=handle_pose,
                        scale=Scale(0.04, width - 0.1, 0.02),
                        thickness=0.02,
                    )
                    for shape in handle.root.visual.shapes:
                        shape.color = Color.GRAY()
                    drawer.add(handle)

            # --- SOFA ---
            sofa = Sofa.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("sofa"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=3.60, y=1.20, z=0.34, yaw=4.7124
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
            cupboard = Cupboard.create_with_new_body_in_world(
                name=PrefixedName("cupboard"),
                world=world,
                world_root_T_self=cupboard_pose,
                scale=cupboard_scale,
                wall_thickness=0.02,
            )

            shelf_scale = Scale(0.40, 0.76, 0.02)
            for i, z in enumerate([-0.5, 0.5]):
                shelf = ShelfLayer.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"cupboard_shelf_{i}"),
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
                    name=PrefixedName(f"cupboard_hinge_{side}"),
                    active_axis=Vector3.Z(),
                    world_root_T_self=handle_pose,
                    connection_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](position=limits[0]),
                        upper=DerivativeMap[float](position=limits[1]),
                    ),
                )
                door = Door.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"cupboard_door_{side}"),
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

                handle = Handle.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"cupboard_handle_{side}"),
                    world_root_T_self=handle_pose
                    @ HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-0.03, y=0.15 if side == "left" else -0.15
                    ),
                    scale=Scale(0.04, 0.04, 0.04),
                    thickness=0.02,
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
                name=PrefixedName("desk"),
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
                    name=PrefixedName(f"desk_leg_{i}"),
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
                name=PrefixedName("cooking_table"),
                world_root_T_self=cooking_table_pose,
                scale=Scale(
                    cooking_table_length, cooking_table_depth, cooking_table_thickness
                ),
            )
            for shape in cooking_table.bodies[0].visual.shapes:
                shape.color = Color.BEIGE()

            cooking_table_cooktop = Cooktop.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("cooktop"),
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
                name=PrefixedName("cooking_table_bottom"),
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
                    name=PrefixedName(f"cooking_mod_{side_name}"),
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
                    name=PrefixedName(f"cooking_drawer_{side_name}"),
                    world_root_T_self=drawer_pose,
                    scale=Scale(module_width - 0.04, cooking_table_depth - 0.02, 0.18),
                )

                slider = Slider.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"cooking_drawer_{side_name}_slider"),
                    world_root_T_self=drawer_pose,
                    active_axis=Vector3.NEGATIVE_X(),
                    connection_limits=DegreeOfFreedomLimits(
                        lower=DerivativeMap[float](position=0.0),
                        upper=DerivativeMap[float](position=0.40),
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
                handle = Handle.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(f"cooking_drawer_handle_{side_name}"),
                    world_root_T_self=handle_pose,
                    scale=Scale(0.04, module_width / 3, 0.04),
                    thickness=0.02,
                )
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
                name=PrefixedName("dining_table"),
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
                    name=PrefixedName(f"dining_table_leg_{i}"),
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
            x=4.22, y=2.22, z=height, yaw=np.pi
        )

        table = Table.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("coffee_table"),
            world_root_T_self=pose,
            scale=Scale(length, width, thick),
        )
        for shape in table.bodies[0].visual.shapes:
            shape.color = color

        shelf = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("coffee_table_shelf"),
            world_root_T_self=pose
            @ HomogeneousTransformationMatrix.from_xyz_rpy(z=-height / 2),
            scale=Scale(length, width, 0.01),
        )
        for shape in shelf.root.visual.shapes:
            shape.color = color
        table.add_object(shelf)

        floor = ShelfLayer.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("coffee_table_floor"),
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
                name=PrefixedName("kitchen_floor"),
                world=world,
                floor_polytope=kitchen_floor_polytope,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.317, y=-0.843
                ),
            )
            kitchen = Room(floor=kitchen_floor, name=PrefixedName("kitchen"))
            room_annotations.append(kitchen)

            living_room_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("living_room_floor"),
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
                name=PrefixedName("bed_room_floor"),
                world=world,
                floor_polytope=bed_room_floor_polytope,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.96, y=4.96
                ),
            )
            bed_room = Room(floor=bed_room_floor, name=PrefixedName("bed_room"))
            room_annotations.append(bed_room)

            office_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("office_floor"),
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