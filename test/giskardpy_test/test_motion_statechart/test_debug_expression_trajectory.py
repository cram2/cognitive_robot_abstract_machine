import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.debug_expression_trajectory import (
    DebugExpressionTrajectory,
)
from giskardpy.motion_statechart.graph_node import DebugExpression, EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.plotters.debug_expression_trajectory_plotter import (
    DebugExpressionTrajectoryPlotter,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.exceptions import NonMonotonicTimeError
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World


def _build_executor(cylinder_bot_world: World) -> Executor:
    """
    Build an executor that moves the bot to a Cartesian point while recording.
    """
    root = cylinder_bot_world.root
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    motion_statechart = MotionStatechart()
    goal = CartesianPosition(
        root_link=root,
        tip_link=tip,
        goal_point=Point3(x=1, reference_frame=root),
        name="cart_pos",
    )
    motion_statechart.add_node(goal)
    motion_statechart.add_node(EndMotion.when_true(goal))
    executor = Executor(
        context=MotionStatechartContext(world=cylinder_bot_world),
        debug_expression_plotter=DebugExpressionTrajectoryPlotter(),
    )
    executor.compile(motion_statechart=motion_statechart)
    return executor


class TestDebugExpressionRecording:
    def test_records_one_value_per_cycle(self, cylinder_bot_world: World):
        executor = _build_executor(cylinder_bot_world)
        executor.tick_until_end()

        trajectory = executor.debug_expression_plotter.debug_expression_trajectory
        assert len(trajectory.recorded_debug_expressions) > 0
        assert len(trajectory.times) > 1
        for recorded in trajectory.recorded_debug_expressions:
            assert recorded.values.shape[0] == len(trajectory.times)

    def test_records_expected_component_count(self, cylinder_bot_world: World):
        executor = _build_executor(cylinder_bot_world)
        executor.tick_until_end()

        trajectory = executor.debug_expression_plotter.debug_expression_trajectory
        current = next(
            recorded
            for recorded in trajectory.recorded_debug_expressions
            if recorded.name == "cart_pos/current"
        )
        # A Point3 evaluates to four homogeneous components.
        assert current.values.shape[1] == 4
        # The recorded tip position should actually change while the bot moves.
        assert not np.allclose(current.values[0], current.values[-1])

    def test_plot_creates_non_empty_pdf(self, cylinder_bot_world: World, tmp_path):
        executor = _build_executor(cylinder_bot_world)
        executor.tick_until_end()

        output = tmp_path / "debug_expressions.pdf"
        executor.plot_debug_expressions(str(output))

        assert output.exists()
        assert output.stat().st_size > 0


class TestDebugExpressionTrajectory:
    def test_append_rejects_non_monotonic_time(self):
        trajectory = DebugExpressionTrajectory.from_debug_expressions(
            [DebugExpression(name="constant", expression=Scalar(data=2.0))]
        )
        trajectory.append(0.0)
        trajectory.append(1.0)
        with pytest.raises(NonMonotonicTimeError):
            trajectory.append(0.5)

    def test_scalar_expression_has_single_component(self):
        trajectory = DebugExpressionTrajectory.from_debug_expressions(
            [DebugExpression(name="constant", expression=Scalar(data=2.0))]
        )
        trajectory.append(0.0)
        trajectory.append(1.0)

        values = trajectory.recorded_debug_expressions[0].values
        assert values.shape == (2, 1)
        assert np.allclose(values, 2.0)
