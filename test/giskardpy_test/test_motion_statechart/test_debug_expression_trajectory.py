import numpy as np
import pytest

from giskardpy.motion_statechart.debug_expression_trajectory import (
    DebugExpressionTrajectory,
)
from giskardpy.motion_statechart.exceptions import EmptyDebugExpressionTrajectoryError
from cramph.exceptions import MissingExecutorExtensionError
from giskardpy.motion_statechart.graph_node import DebugExpression, EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.plotters.debug_expression_trajectory_plotter import (
    DebugExpressionTrajectoryPlotter,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.exceptions import NonMonotonicTimeError
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from giskardpy.motion_control import DebugExpressionRecording
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor


def _build_motion_statechart(
    cylinder_bot_world: World, executor: StatechartExecutor
) -> Statechart:
    """
    Build a motion statechart that moves the bot to a Cartesian point, in the context of
    `executor`.
    """
    root = cylinder_bot_world.root
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    motion_statechart = Statechart(context=executor.context)
    goal = CartesianPosition(
        root_link=root,
        tip_link=tip,
        goal_point=Point3(x=1, reference_frame=root),
        name="cart_pos",
    )
    motion_statechart.add_node(goal)
    motion_statechart.add_node(EndMotion.when_true(goal))
    return motion_statechart


def _build_executor(cylinder_bot_world: World) -> StatechartExecutor:
    """
    Build an executor that moves the bot to a Cartesian point while recording.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[
            MotionControl(),
            DebugExpressionRecording(plotter=DebugExpressionTrajectoryPlotter()),
        ],
    )
    executor.compile(statechart=_build_motion_statechart(cylinder_bot_world, executor))
    return executor


class TestDebugExpressionRecording:
    def test_records_one_value_per_cycle(self, cylinder_bot_world: World):
        executor = _build_executor(cylinder_bot_world)
        executor.tick_until_end()

        trajectory = executor.require_extension(
            DebugExpressionRecording
        ).plotter.debug_expression_trajectory
        assert len(trajectory.recorded_debug_expressions) > 0
        assert len(trajectory.times) > 1
        for recorded in trajectory.recorded_debug_expressions:
            assert recorded.values.shape[0] == len(trajectory.times)

    def test_records_expected_component_count(self, cylinder_bot_world: World):
        executor = _build_executor(cylinder_bot_world)
        executor.tick_until_end()

        trajectory = executor.require_extension(
            DebugExpressionRecording
        ).plotter.debug_expression_trajectory
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
        executor.require_extension(DebugExpressionRecording).plotter.plot(str(output))

        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_creates_missing_output_directory(
        self, cylinder_bot_world: World, tmp_path
    ):
        executor = _build_executor(cylinder_bot_world)
        executor.tick_until_end()

        output = tmp_path / "debug_expressions" / "debug_expressions.pdf"
        executor.require_extension(DebugExpressionRecording).plotter.plot(str(output))

        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_raises_when_plotter_not_configured(
        self, cylinder_bot_world: World, tmp_path
    ):
        executor = StatechartExecutor(
            context=StatechartContext(world=cylinder_bot_world),
            extensions=[MotionControl()],
        )
        executor.compile(
            statechart=_build_motion_statechart(cylinder_bot_world, executor)
        )

        output = tmp_path / "debug_expressions.pdf"
        with pytest.raises(MissingExecutorExtensionError):
            executor.require_extension(DebugExpressionRecording).plotter.plot(
                str(output)
            )


class TestDebugExpressionTrajectory:
    def test_plot_raises_when_no_samples_were_recorded(self, tmp_path):
        plotter = DebugExpressionTrajectoryPlotter()
        plotter.reset([DebugExpression(name="constant", expression=Scalar(data=2.0))])

        with pytest.raises(EmptyDebugExpressionTrajectoryError):
            plotter.plot(str(tmp_path / "debug_expressions.pdf"))

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
