from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Table,
    Sofa,
    TrashCan,
    Fridge, CounterTop, Wall, Cabinet,
    Cupboard,
    Door,
    Desk,
    Handle,
    ShelfLayer,
    Hinge, Oven,
)
from semantic_digital_twin.world import World
import threading
import rclpy
import numpy as np
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room, Floor
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import FixedConnection, RevoluteConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits, DegreeOfFreedom,
)

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
        Builds and configures the environment walls for a given world. This involves creating
        various walls with predefined dimensions, transformation matrices, and connections.

        :param world: An instance representing the environment world where walls are to be
        configured and added.

        :return: The modified world instance with configured walls and connections.
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
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-2.01
                ),
                scale=Scale(x=0.05, y=1.00, z=3.00),
            )

            south_wall2 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall2"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-1.45, yaw=np.pi/2
                ),
                scale=Scale(x=0.05, y=0.29, z=3.00),
            )

            south_wall3 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall3"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29, y=-0.9925
                ),
                scale=Scale(x=0.05, y=1.085, z=1.00),
            )

            south_wall4 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall4"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=-0.45, yaw=np.pi/2
                ),
                scale=Scale(x=0.05, y=0.29, z=1.00),
            )

            south_wall5 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall5"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.145, y=0.45, yaw=np.pi/2
                ),
                scale=Scale(0.05, 0.29, 1.00),
            )

            south_wall6 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall6"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=1.80
                ),
                scale=Scale(0.05, 2.75, 1.00),
            )

            south_wall7 = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("south_wall7"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.29025, y=5.16
                ),
                scale=Scale(0.05, 2.27, 1.00),
            )

            east_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("east_wall"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.462, y=-2.535, yaw=np.pi/2
                ),
                scale=Scale(0.05, 4.924, 3.00),
            )

            middle_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("middle_wall"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.20975, y=5.00
                ),
                scale=Scale(0.05, 2.67, 1.00),
            )

            west_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("west_wall"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.9345, y=6.32, yaw=np.pi/2
                ),
                scale=Scale(0.05, 4.449, 3.00),
            )

            north_wall = Wall.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("north_wall"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.949, y=1.51
                ),
                scale=Scale(0.05, 8.04, 3.00),
            )

            world.add_connection(root_C_north_west_wall)
            return world

    def _build_environment_furniture(self, world: World):
        """
        Adds furniture items and room layouts (kitchen, living room, bedroom, office) to the scene graph.
        Connects furniture bodies and room structures hierarchically under the main root.
        Returns the updated World object with furniture integrated.
        """
        root = world.root

        with world.modify_world():
            cupboard_scale = Scale(0.43, 0.80, 2.02)

            cupboard = Cupboard.create_with_new_body_in_world(
                name=PrefixedName("cupboard"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.55, y=4.72, z=1.01
                ),
                scale=cupboard_scale,
                wall_thickness=0.02,
            )

            # create shelflayers manually
            shelf_1 = ShelfLayer.create_with_new_body_in_world(
                name=PrefixedName("shelf_1"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.55, y=4.72, z=0.51
                ),
                scale=Scale(0.40, 0.76, 0.02),
            )

            shelf_2 = ShelfLayer.create_with_new_body_in_world(
                name=PrefixedName("shelf_2"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.55, y=4.72, z=1.51
                ),
                scale=Scale(0.40, 0.76, 0.02),
            )


            # Creating doors manually and attaching them directly to the cupboard
            door_height = 1.055
            # Position Z: Bottom of cupboard is at -cupboard_scale.z / 2.
            # Door center should be at Bottom + door_height / 2
            door_z_rel = -(cupboard_scale.z / 2) + (door_height / 2)

            door_x_rel = -(cupboard_scale.x / 2) - 0.01
            door_scale = Scale(0.02, 0.40, door_height)

            # Define limits for doors
            # Left door opens outwards (0 to +90 degrees)
            left_lower = DerivativeMap[float](position=0.0)
            left_upper = DerivativeMap[float](position=np.pi / 2)
            left_door_limits = DegreeOfFreedomLimits(lower=left_lower, upper=left_upper)

            # Right door opens outwards (-90 to 0 degrees)
            right_lower = DerivativeMap[float](position=-np.pi / 2)
            right_upper = DerivativeMap[float](position=0.0)
            right_door_limits = DegreeOfFreedomLimits(lower=right_lower, upper=right_upper)

            # Left Door (Open via Hinge)
            # Create Hinge for the left door
            hinge_left_body = Body(name=PrefixedName("cupboard_hinge_left_body"))
            hinge_left = Hinge(
                root=hinge_left_body,
                name=PrefixedName("cupboard_hinge_left"),
            )

            cupboard_C_hinge_left = RevoluteConnection.create_with_dofs(
                world=world,
                parent=cupboard.root,
                child=hinge_left_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=door_x_rel, y=-0.40, z=door_z_rel
                ),
                axis=Vector3.Z(),
                dof_limits=left_door_limits,
            )
            world.add_connection(cupboard_C_hinge_left)
            world.add_semantic_annotation(hinge_left)

            left_door = Door.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("cupboard_left_door"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.325, y=4.52, z=0.5275#, yaw=np.pi/2
                ),
                scale=door_scale,
            )



            # Right Door (Closed via Hinge)
            hinge_right_body = Body(name=PrefixedName("cupboard_hinge_right_body"))
            hinge_right = Hinge(
                root=hinge_right_body,
                name=PrefixedName("cupboard_hinge_right"),
            )

            cupboard_C_hinge_right = RevoluteConnection.create_with_dofs(
                world=world,
                parent=cupboard.root,
                child=hinge_right_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=door_x_rel, y=0.40, z=door_z_rel
                ),
                axis=Vector3.Z(),
                dof_limits=right_door_limits,
            )
            world.add_connection(cupboard_C_hinge_right)
            world.add_semantic_annotation(hinge_right)

            right_door = Door.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("cupboard_right_door"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.325, y=4.92, z=0.5275  # , yaw=np.pi/2
                ),
                scale=door_scale,
            )


            left_door_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("left_door_handle"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.315, y=4.68, z=0.5275, yaw=np.pi
                ),
                scale=Scale(0.04, 0.02, 0.02),
            )

            right_door_handle = Handle.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("right_door_handle"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.315, y=4.76, z=0.5275, yaw=np.pi
                ),
                scale=Scale(0.04, 0.02, 0.02),
            )


            left_door.add_handle(left_door_handle)
            right_door.add_handle(right_door_handle)

            oven = Oven.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("oven"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=3.481, y=-2.181, z=0.745
                ),
                scale=Scale(1.20, 0.658, 1.49),
            )

            trash_can = TrashCan.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("trash_can"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.416, y=5.5, z=0.20
                ),
                scale=Scale(x=0.30, y=0.30, z=0.40),
            )

            refrigerator = Fridge.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("refrigerator"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                     x=0.537, y=-2.181, z=0.745, yaw=np.pi*3/2),
                scale=Scale(x=0.60, y=0.658, z=1.49),
            )

            counterTop = CounterTop.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("counterTop"),
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(x=1.859, y=-2.181, z=0.2725),
                scale=Scale(x=2.044, y=0.658, z=0.545),
            )
            for color in counterTop.bodies[0].visual.shapes:
                color.color = Color.BEIGE()


            table = Table.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("table"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.545, y=0.426, z=0.4225),
                scale=Scale(x=2.45, y=0.796, z=0.845),
            )

            sofa = Sofa.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("sofa"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.60, y=1.20, z=0.34),
                scale=Scale(x=1.68, y=0.94, z=0.68),
            )
            for color in sofa.bodies[0].visual.shapes:
                color.color = Color.BEIGE()

            lowerTable = Table.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("lowerTable"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=4.22, y=2.22, z=0.22),
                scale=Scale(x=0.37, y=0.91, z=0.44),
            )

            desk = Table.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("desk"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.05, y=1.28, z=0.375),
                scale=Scale(x=0.60, y=1.20, z=0.75),
            )

            cooking_table = Table.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("cooking_table"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.325, y=5.99, z=0.355),
                scale=Scale(1.75, 0.64, 0.71),
            )
            for color in cooking_table.bodies[0].visual.shapes:
                color.color = Color.BEIGE()

            dinning_table = Table.create_with_new_body_in_world(
                world=world,
                name=PrefixedName("dining_table"),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=2.59975, y=5.705,
                                                                                                     z=0.365),
                scale=Scale(0.73, 1.18, 0.73),
            )
            for color in dinning_table.bodies[0].visual.shapes:
                color.color = Color.BEIGE()

            # world.add_connection(root_C_ovenArea)
        return world

    def _build_environment_rooms(self, world: World):


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
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(x=2.317, y=-0.843),
            )
            kitchen = Room(floor=kitchen_floor, name=PrefixedName("kitchen"))

            living_room_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("living_room_floor"),
                world=world,
                floor_polytope=living_room_floor_polytope,
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(x=2.317, y=2.3095),
            )
            living_room = Room(floor=living_room_floor, name=PrefixedName("living_room"))

            bed_room_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("bed_room_floor"),
                world=world,
                floor_polytope=bed_room_floor_polytope,
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(x=0.96, y=4.96),
            )
            bed_room = Room(floor=bed_room_floor, name=PrefixedName("bed_room"))

            office_floor = Floor.create_with_new_body_from_polytope_in_world(
                name=PrefixedName("office_floor"),
                world=world,
                floor_polytope=office_floor_polytope,
                world_root_T_self= HomogeneousTransformationMatrix.from_xyz_rpy(x=3.56, y=4.96),
            )
            office = Room(floor=office_floor, name=PrefixedName("office"))

        return world
