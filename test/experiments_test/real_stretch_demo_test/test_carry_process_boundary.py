"""
A carry sent to a controller running in its own process, whose context does not hold the
robot.
"""

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import RealRobotPlanExecutor
from coraplex.robot_plans.actions.composite.carrying import CarryAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from .conftest import StandaloneControllerProcess

# %% carrying across the process boundary


def test_a_carry_runs_against_a_controller_in_another_process(
    stretch_controller_process: StandaloneControllerProcess,
) -> None:
    """
    The controller reads what the arm holds when the carry starts, and adds no node the
    client does not know of.
    """
    session = stretch_controller_process.session
    world = session.fetch_world()
    WorldSynchronizer(_world=world, node=session.node)
    stretch = world.get_semantic_annotations_by_type(Stretch)[0]
    arm = stretch.all_arms[0]
    tool_frame = arm.end_effector.tool_frame
    held_body = Body(name=PrefixedName("held_body"))
    with world.modify_world():
        world.add_kinematic_structure_entity(held_body)
        world.add_connection(
            FixedConnection(
                parent=tool_frame,
                child=held_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=0.4, pitch=0.3, reference_frame=tool_frame
                ),
            )
        )
    executor = RealRobotPlanExecutor(
        context=StatechartContext.with_extensions(
            world=world, extensions=[RobotAccess(robot=stretch)]
        ),
        ros_node=session.node,
    )
    carry = CarryAction(
        navigate=NavigateAction(
            target_location=Pose.from_xyz_rpy(x=0.3, reference_frame=stretch.root)
        ),
        holding_arms=[arm],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=carry)

    executor.compile(statechart=statechart)
    executor.execute()
    executor.close()

    assert carry.life_cycle_state == LifeCycleValues.SUCCEEDED
