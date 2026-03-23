from copy import deepcopy

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Cylinder, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
)


def robot_factory(fucking_huge_link_length: float, vel_limit: float) -> World:
    fucking_huge_cylinder = ShapeCollection(
        shapes=[
            Cylinder(
                width=fucking_huge_link_length / 10,
                height=fucking_huge_link_length,
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-(fucking_huge_link_length / 2)
                ),
            )
        ]
    )
    fucking_huge_sphere = ShapeCollection(
        shapes=[
            Sphere(
                radius=fucking_huge_link_length / 18,
            )
        ]
    )
    dof_limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(None, -vel_limit, None, None),
        upper=DerivativeMap(None, vel_limit, None, None),
    )
    world = World()
    with world.modify_world():
        # %% joint1
        root = Body(name=PrefixedName("map"))
        link1 = Body(
            name=PrefixedName("link1"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=root,
                child=link1,
                axis=Vector3.Z(),
                dof_limits=dof_limits,
            )
        )

        # %% joint2
        link2 = Body(
            name=PrefixedName("link2"),
            collision=deepcopy(fucking_huge_cylinder),
            visual=deepcopy(fucking_huge_cylinder),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link1,
                child=link2,
                axis=Vector3.X(),
                dof_limits=dof_limits,
                connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=fucking_huge_link_length,
                ),
            )
        )

        # %% joint3
        link3 = Body(
            name=PrefixedName("link3"),
            collision=deepcopy(fucking_huge_cylinder),
            visual=deepcopy(fucking_huge_cylinder),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link2,
                child=link3,
                axis=Vector3.X(),
                dof_limits=dof_limits,
                connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=fucking_huge_link_length,
                ),
            )
        )

        # %% joint4
        link4 = Body(
            name=PrefixedName("link4"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link3,
                child=link4,
                dof_limits=dof_limits,
                axis=Vector3.Z(),
            )
        )

        # %% joint5
        link5 = Body(
            name=PrefixedName("link5"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link4,
                child=link5,
                axis=Vector3.X(),
                dof_limits=dof_limits,
            )
        )

        # %% joint6
        eef = Body(
            name=PrefixedName("eef"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link5,
                child=eef,
                axis=Vector3.Y(),
                dof_limits=dof_limits,
            )
        )
    return world


def execute(link_length: float, vel_limit: float):
    fucking_huge_robot = robot_factory(
        fucking_huge_link_length=link_length, vel_limit=vel_limit
    )
    msc = MotionStatechart()
    goal = 1
    eef = fucking_huge_robot.get_kinematic_structure_entity_by_name("eef")

    msc.add_node(
        node1 := Sequence(
            [
                SetSeedConfiguration(
                    seed_configuration=JointState.from_str_dict(
                        {
                            "map_T_link1": goal,
                            "link1_T_link2": goal,
                            "link2_T_link3": goal,
                            "link3_T_link4": goal,
                            "link4_T_link5": goal,
                            "link5_T_eef": goal,
                        },
                        world=fucking_huge_robot,
                    )
                ),
                CartesianPosition(
                    root_link=fucking_huge_robot.root,
                    tip_link=eef,
                    goal_point=Point3(y=-link_length, reference_frame=eef),
                    reference_velocity=0.2 * link_length,
                ),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(node1))

    kin_sim = Executor(
        MotionStatechartContext(
            world=fucking_huge_robot,
            qp_controller_config=QPControllerConfig(
                target_frequency=100,
                prediction_horizon=50,
            ),
        ),
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end(10_000)


def test_cart_goal01():
    execute(0.1, 0.1)


def test_cart_goal1():
    execute(1.0, 0.1)


def test_cart_goal1000(rclpy_node):
    execute(1000.0, 0.1)
