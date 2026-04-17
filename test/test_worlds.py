from semantic_digital_twin.collision_checking.collision_detector import CollisionCheck
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Cup,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import Cylinder, Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def setup_contact_world():
    world = World()
    root = Body(name=PrefixedName("root"))

    cylinder = Cylinder(width=0.50, height=0.5)
    shape_geometry = ShapeCollection([cylinder])
    cylinder_body = Body(
        name=PrefixedName("cylinder_body"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    box1 = Box(scale=Scale(0.60, 0.5, 0.5))
    shape_geometry = ShapeCollection([box1])
    box_body1 = Body(
        name=PrefixedName("box_body1"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    box2 = Box(scale=Scale(0.60, 0.5, 0.5))
    shape_geometry = ShapeCollection([box2])
    box_body2 = Body(
        name=PrefixedName("box_body2"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    with world.modify_world():
        root_C_box1 = FixedConnection(
            parent=root,
            child=box_body1,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.5, y=-3.5, z=0.25
            ),
        )


        root_C_box2 = FixedConnection(
            parent=root,
            child=box_body2,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.5, y=-3.5, z=0.25
            ),
        )

        root_C_cylinder = Connection6DoF.create_with_dofs(
            world=world,
            parent=root,
            child=cylinder_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, y=-3, z=0.25
            ),
        )

        world.add_kinematic_structure_entity(root)
        world.add_connection(root_C_cylinder)
        world.add_connection(root_C_box2)
        world.add_connection(root_C_box1)

    return world




def setup_spatial_world():
    world = World()
    root = Body(name=PrefixedName("root"))

    box1 = Box(scale=Scale(0.20, 0.2, 0.2))
    shape_geometry = ShapeCollection([box1])
    box_body1 = Body(
        name=PrefixedName("box_body1"),
        collision=shape_geometry,
        visual=shape_geometry,
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

    with world.modify_world():
        cabinet = Cabinet.create_with_new_body_in_world(
            name=PrefixedName("cabinet"), world=world, scale=Scale(0.5, 0.5, 1.0)
        )
        root_C_box1 = Connection6DoF.create_with_dofs(
            world=world,
            parent=root,
            child=box_body1,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.4, y=0, z=0
            ),
        )
        world.add_connection(root_C_box1)
    return world


def setup_support_world():
    world = World()
    root = Body(name=PrefixedName("root"))
    table = Box(scale=Scale(1, 0.5, 0.5))
    shape_geometry = ShapeCollection([table])
    table_body = Body(
        name=PrefixedName("table_body"), collision=shape_geometry, visual=shape_geometry
    )

    hole = Box(scale=Scale(0.1, 0.1, 0.1))
    hole_geometry = ShapeCollection([hole])
    hole_body = Body(
        name=PrefixedName("hole_body"),
        collision=hole_geometry,
        visual=hole_geometry
    )


    table2 = Box(scale=Scale(1, 0.5, 0.5))
    shape_geometry = ShapeCollection([table2])
    table2_body = Body(
        name=PrefixedName("table2_body"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    cylinder = Cylinder(width=0.1, height=0.2)
    shape_geometry = ShapeCollection([cylinder])
    cylinder_body = Body(
        name=PrefixedName("cylinder_body"),
        collision=shape_geometry,
        visual=shape_geometry,
    )

    root_C_table2 = FixedConnection(
        parent=root,
        child=table2_body,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=3.545, y=0.426, z=0.25
        ),
    )

    root_C_table = FixedConnection(
        parent=root,
        child=table_body,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=3.545, y=1.426, z=0.25
        ),
    )

    root_C_hole = FixedConnection(
        parent=root,
        child=hole_body,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=3.245, y=2.426, z=0.75
        ),
    )


    with world.modify_world():
        world.add_kinematic_structure_entity(root)


    with world.modify_world():
        world.add_connection(root_C_table)
        world.add_connection(root_C_table2)
        world.add_connection(root_C_hole)

        cabinet = Cabinet.create_with_new_body_in_world(
            world=world,
            scale=Scale(0.5, 0.5, 1.0),
            name=PrefixedName("cabinet"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.545, y=2.426, z=0.5
            ),
        )

        root_C_cylinder = Connection6DoF.create_with_dofs(
            world=world,
            parent=root,
            child=cylinder_body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0, y=0, z=0.1
            ),
        )
        world.add_connection(root_C_cylinder)

    return world