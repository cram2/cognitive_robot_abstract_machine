"""
A plan avoiding collisions, sent to a controller running in its own process.
"""

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import RealRobotPlanExecutor
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.statechart import Statechart
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.stretch import Stretch

from .conftest import StandaloneControllerProcess

# %% collision avoidance across the process boundary


def test_a_plan_avoiding_collisions_runs_against_a_controller_in_another_process(
    stretch_controller_process: StandaloneControllerProcess,
) -> None:
    """
    The client adds the collision avoidance to the plan, but only the controller knows
    the motion control configuration it was started with, so the client has to send that
    plan without one.
    """
    session = stretch_controller_process.session
    world = session.fetch_world()
    WorldSynchronizer(_world=world, node=session.node)
    stretch = world.get_semantic_annotations_by_type(Stretch)[0]
    executor = RealRobotPlanExecutor(
        StatechartContext.with_extensions(world, [RobotAccess(stretch)]),
        ros_node=session.node,
        collision_avoidance=True,
    )
    park_arms = ParkArmsAction(arms=stretch.all_arms)
    statechart = Statechart(context=executor.context)
    statechart.add_node(park_arms)

    executor.compile(statechart)
    executor.execute()
    executor.close()

    assert park_arms.life_cycle_state == LifeCycleValues.SUCCEEDED
