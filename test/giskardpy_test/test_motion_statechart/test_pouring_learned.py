"""Closed-loop A/B of the analytic against the learned head-above-lip pouring model.

Trains a surrogate once per module, then drives the same single-cup :class:`PouringTask` with the
analytic and the learned fill equation. Both arms must settle at the fill goal and agree on the
final fill, proving the controller is agnostic to whether the physical head model is analytic or
learned. Requires torch + l4casadi and skips without them.
"""

import math
from dataclasses import dataclass

import pytest

pytest.importorskip("torch")
pytest.importorskip("l4casadi")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from krrood.ormatic.utils import classproperty  # noqa: E402

from giskardpy.executor import Executor, SimulationPacer  # noqa: E402
from giskardpy.motion_statechart.context import MotionStatechartContext  # noqa: E402
from giskardpy.motion_statechart.graph_node import EndMotion  # noqa: E402
from giskardpy.motion_statechart.motion_statechart import MotionStatechart  # noqa: E402
from giskardpy.motion_statechart.tasks.pouring import PouringTask  # noqa: E402
from giskardpy.qp.qp_controller_config import QPControllerConfig  # noqa: E402
from semantic_digital_twin.datastructures.prefixed_name import (
    PrefixedName,
)  # noqa: E402
from semantic_digital_twin.physics.equations.head_surrogate_training import (  # noqa: E402
    HeadSurrogateTrainer,
)
from semantic_digital_twin.physics.equations.learned_pouring_equations import (  # noqa: E402
    LearnedHeadModelReference,
    LearnedPouringEquation,
)
from semantic_digital_twin.physics.equations.pouring_equations import (  # noqa: E402
    ArticulatedPouringEquation,
    PouringEquation,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel  # noqa: E402
from semantic_digital_twin.spatial_types import Vector3  # noqa: E402
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap  # noqa: E402
from semantic_digital_twin.world import World  # noqa: E402
from semantic_digital_twin.world_description.connections import (  # noqa: E402
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (  # noqa: E402
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale  # noqa: E402
from semantic_digital_twin.world_description.world_entity import Body  # noqa: E402

CONTAINER_HEIGHT = 0.1
CONTAINER_WIDTH = 0.08
OUTFLOW_RATE_CONSTANT = 1.0

GOAL_FILL = 0.6
FILL_TOLERANCE = 0.05
SETTLE_TICKS = 20
"""Number of final ticks that must all lie within the fill tolerance to count as settled."""


@pytest.fixture(scope="module")
def trained_model_reference(tmp_path_factory) -> LearnedHeadModelReference:
    """A reference to a surrogate trained for the test cup's geometry, cached for the module."""
    checkpoint_path = tmp_path_factory.mktemp("learned_head") / "head_surrogate.pt"
    surrogate = HeadSurrogateTrainer(
        container_height=CONTAINER_HEIGHT,
        container_width=CONTAINER_WIDTH,
        sample_count=8000,
        epochs=1500,
        gradient_weight=0.3,
    ).train()
    torch.save(surrogate.state_dict(), str(checkpoint_path))
    return LearnedHeadModelReference(
        checkpoint_path=str(checkpoint_path),
        trained_container_height=CONTAINER_HEIGHT,
        trained_container_width=CONTAINER_WIDTH,
    )


@dataclass(eq=False)
class SingleCupContainer(HasFillLevel):
    """A held container with a single tilt joint, used for the single-cup pouring A/B."""

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


def _build_single_cup_world() -> tuple[World, SingleCupContainer]:
    """Minimal world with one tilt-jointed, fully filled container."""
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        cup = SingleCupContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
            scale=Scale(0.4, 0.4, 1.0),
        )
    cup.initialize_fill_level(
        world=world, initial_fill=1.0, outflow_rate_constant=OUTFLOW_RATE_CONSTANT
    )
    world.set_positions_1DOF_connection({cup.root.parent_connection: 0.1})
    return world, cup


def _run_single_cup_pour(equation: PouringEquation) -> np.ndarray:
    """Drive a single-cup :class:`PouringTask` with ``equation`` and record the fill per tick.

    The equation is installed on the cup's fill connection (so the fill ODE integrates it) and on
    the task (so the MPC predicts with it), keeping integration and prediction consistent for both
    the analytic and the learned arm.
    """
    world, cup = _build_single_cup_world()
    with world.modify_world():
        cup.add_fill_equation(equation)
    task = PouringTask(
        fill_equation=equation,
        fill_connection=cup.fill_connection,
        root_link=world.root,
        tip_link=cup.root,
        goal_value=GOAL_FILL,
        fill_level_tolerance=FILL_TOLERANCE,
        reference_velocity=0.05,
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(task)
    motion_statechart.add_node(EndMotion.when_true(task))

    fill_history: list[float] = []
    original_on_tick = task.on_tick

    def recording_on_tick(context):
        fill_history.append(float(cup.fill_level))
        return original_on_tick(context)

    task.on_tick = recording_on_tick
    executor = Executor(
        MotionStatechartContext(
            world=world,
            qp_controller_config=QPControllerConfig(
                target_frequency=80, prediction_horizon=120
            ),
        ),
        pacer=SimulationPacer(real_time_factor=1),
    )
    executor.compile(motion_statechart=motion_statechart)
    executor.tick_until_end(timeout=4000)
    return np.array(fill_history)


def _settled_at_goal(fill_history: np.ndarray) -> bool:
    """Whether the fill stayed within the goal tolerance over the final ticks."""
    return bool(
        np.all(np.abs(fill_history[-SETTLE_TICKS:] - GOAL_FILL) <= FILL_TOLERANCE)
    )


def test_learned_head_pour_matches_analytic(trained_model_reference):
    """The same controller settles at the fill goal with either head model, and both agree."""
    analytic_history = _run_single_cup_pour(
        ArticulatedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
        )
    )
    learned_history = _run_single_cup_pour(
        LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
            model_reference=trained_model_reference,
        )
    )

    assert _settled_at_goal(analytic_history), "analytic pour did not settle at goal"
    assert _settled_at_goal(learned_history), "learned pour did not settle at goal"
    assert abs(analytic_history[-1] - learned_history[-1]) < 0.03, (
        "analytic and learned pours diverge in final fill: "
        f"{analytic_history[-1]:.3f} vs {learned_history[-1]:.3f}"
    )
