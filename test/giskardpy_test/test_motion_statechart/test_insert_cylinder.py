import numpy as np

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.tracebot import InsertCylinder
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

CYLINDER_HEIGHT = 0.5


def free_cylinder_world() -> World:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        tx = Body(name=PrefixedName("tx"))
        ty = Body(name=PrefixedName("ty"))
        tz = Body(name=PrefixedName("tz"))
        rx = Body(name=PrefixedName("rx"))
        ry = Body(name=PrefixedName("ry"))
        cylinder = Body(
            name=PrefixedName("cylinder"),
            collision=ShapeCollection(
                shapes=[Cylinder(width=0.1, height=CYLINDER_HEIGHT)]
            ),
        )
        chain = [
            (PrismaticConnection, root, tx, Vector3.X()),
            (PrismaticConnection, tx, ty, Vector3.Y()),
            (PrismaticConnection, ty, tz, Vector3.Z()),
            (RevoluteConnection, tz, rx, Vector3.X()),
            (RevoluteConnection, rx, ry, Vector3.Y()),
            (RevoluteConnection, ry, cylinder, Vector3.Z()),
        ]
        for connection_type, parent, child, axis in chain:
            world.add_connection(
                connection_type.create_with_dofs(
                    world=world, parent=parent, child=child, axis=axis
                )
            )
        MinimalRobot.from_world(world)
    return world


def test_insert_cylinder():
    world = free_cylinder_world()
    cylinder = world.get_kinematic_structure_entity_by_name("cylinder")
    hole_point = Point3(0.1, 0.0, 0.0, reference_frame=world.root)

    msc = MotionStatechart()
    goal = InsertCylinder(
        tip_link=cylinder,
        tip_P_tool=Point3(0.0, 0.0, -CYLINDER_HEIGHT / 2),
        hole_point=hole_point,
        pre_grasp_height=0.05,
    )
    msc.add_node(goal)
    msc.add_node(EndMotion.when_true(goal))

    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=msc)
    executor.tick_until_end(3000)

    root_T_cylinder = world.compute_forward_kinematics_np(world.root, cylinder)
    z_axis = root_T_cylinder[:3, 2]
    bottom = root_T_cylinder[:3, 3] - z_axis * CYLINDER_HEIGHT / 2
    # reaches hole
    assert np.allclose(bottom, hole_point.to_np()[:3], atol=0.02)
    # ends upright
    assert np.allclose(z_axis, [0.0, 0.0, 1.0], atol=0.02)
