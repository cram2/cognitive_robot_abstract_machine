from cramph.executor import SimulationPacer
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.monitors import CountSeconds
from cramph.statechart import Statechart
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def test_with_executor():
    executor = StatechartExecutor(
        context=StatechartContext(world=World()),
        pacer=SimulationPacer(real_time_factor=2.0),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
            )
        ],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(counter := CountSeconds(seconds=1.0))
    statechart.add_node(EndMotion.when_true(counter))

    executor.compile(statechart)
    executor.tick_until_end(timeout=1000)
    # we tick 20 (hz) * 2 (real_time_factor) per second and sleep for 1s.
    # +2 because the endmotion needs to extra ticks
    assert executor.tick_count == 42
