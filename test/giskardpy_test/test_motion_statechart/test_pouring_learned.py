"""Closed-loop A/B of the analytic against the learned head-above-lip pouring model.

Trains a surrogate once per module, then drives the same single-cup :class:`PouringTask` with the
analytic and the learned fill equation. Both arms must settle at the fill goal and agree on the
final fill, proving the controller is agnostic to whether the physical head model is analytic or
learned. Requires torch + l4casadi and skips without them.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("l4casadi")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import krrood.symbolic_math.symbolic_math as sm  # noqa: E402

from giskardpy.executor import Executor, SimulationPacer  # noqa: E402
from giskardpy.motion_statechart.context import MotionStatechartContext  # noqa: E402
from giskardpy.motion_statechart.graph_node import EndMotion  # noqa: E402
from giskardpy.motion_statechart.motion_statechart import MotionStatechart  # noqa: E402
from giskardpy.motion_statechart.tasks.pouring import PouringTask  # noqa: E402
from giskardpy.qp.qp_controller_config import QPControllerConfig  # noqa: E402
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
    SymbolicFillContext,
)
from semantic_digital_twin.world import World  # noqa: E402

from .single_cup_world import (  # noqa: E402
    PourableContainer,
    build_single_cup_world,
)

CONTAINER_HEIGHT = 0.1
CONTAINER_WIDTH = 0.08
OUTFLOW_RATE_CONSTANT = 1.0

GOAL_FILL = 0.6
FILL_TOLERANCE = 0.05
SETTLE_TICKS = 20
"""Number of final ticks that must all lie within the fill tolerance to count as settled."""

FINAL_FILL_AGREEMENT_TOLERANCE = 0.03
"""Maximum allowed difference between the analytic and learned arm's final fill level."""

HEAD_MODEL_VELOCITY_AGREEMENT_TOLERANCE = 0.1
"""Maximum allowed fill-velocity difference between the learned surrogate and the analytic head
model at a flowing operating point; larger deviations indicate a mistrained surrogate."""

FLOWING_TILT_ANGLE = 1.3
"""Tilt angle well above the spill threshold, so the head model is actively pouring."""

FLOWING_FILL_LEVEL = 0.8
"""Fill level at which the flowing operating point is evaluated."""


@pytest.fixture(scope="module")
def trained_model_reference(
    tmp_path_factory: pytest.TempPathFactory,
) -> LearnedHeadModelReference:
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


def _build_single_cup_world() -> tuple[World, PourableContainer]:
    """Minimal world with one tilt-jointed, fully filled container."""
    return build_single_cup_world(outflow_rate_constant=OUTFLOW_RATE_CONSTANT)


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

    def recording_on_tick(context: MotionStatechartContext):
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


def _assert_settled_at_goal(fill_history: np.ndarray) -> None:
    """Assert the fill stayed at the goal (within tolerance) over the final ticks."""
    np.testing.assert_allclose(
        fill_history[-SETTLE_TICKS:], GOAL_FILL, atol=FILL_TOLERANCE
    )


@pytest.fixture(scope="module")
def analytic_equation() -> ArticulatedPouringEquation:
    """The analytic head-above-lip pouring model for the test cup's geometry."""
    return ArticulatedPouringEquation(
        container_height=CONTAINER_HEIGHT,
        container_width=CONTAINER_WIDTH,
        outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
    )


@pytest.fixture(scope="module")
def learned_equation(
    trained_model_reference: LearnedHeadModelReference,
) -> LearnedPouringEquation:
    """The learned pouring model backed by the module's trained surrogate."""
    return LearnedPouringEquation(
        container_height=CONTAINER_HEIGHT,
        container_width=CONTAINER_WIDTH,
        outflow_rate_constant=OUTFLOW_RATE_CONSTANT,
        model_reference=trained_model_reference,
    )


@pytest.fixture(scope="module")
def analytic_fill_history(
    analytic_equation: ArticulatedPouringEquation,
) -> np.ndarray:
    """Per-tick fill levels of the closed-loop pour driven by the analytic model."""
    return _run_single_cup_pour(analytic_equation)


@pytest.fixture(scope="module")
def learned_fill_history(learned_equation: LearnedPouringEquation) -> np.ndarray:
    """Per-tick fill levels of the closed-loop pour driven by the learned model."""
    return _run_single_cup_pour(learned_equation)


@pytest.mark.slow
def test_analytic_pour_settles_at_goal(analytic_fill_history: np.ndarray) -> None:
    """The controller settles the analytic arm at the fill goal."""
    _assert_settled_at_goal(analytic_fill_history)
    assert analytic_fill_history[-1] == pytest.approx(GOAL_FILL, abs=FILL_TOLERANCE)


@pytest.mark.slow
def test_learned_pour_settles_at_goal(learned_fill_history: np.ndarray) -> None:
    """The controller settles the learned arm at the fill goal."""
    _assert_settled_at_goal(learned_fill_history)
    assert learned_fill_history[-1] == pytest.approx(GOAL_FILL, abs=FILL_TOLERANCE)


@pytest.mark.slow
def test_learned_and_analytic_arms_agree_on_final_fill(
    analytic_fill_history: np.ndarray, learned_fill_history: np.ndarray
) -> None:
    """The two head models must not steer the same controller to different final fills."""
    assert learned_fill_history[-1] == pytest.approx(
        analytic_fill_history[-1], abs=FINAL_FILL_AGREEMENT_TOLERANCE
    )


@pytest.mark.slow
def test_learned_arm_exercises_the_learned_head_model(
    analytic_equation: ArticulatedPouringEquation,
    learned_equation: LearnedPouringEquation,
) -> None:
    """
    The learned arm must actually run through the surrogate: its fill velocity at a flowing
    operating point differs from the analytic model (no silent analytic fallback) while
    staying within the surrogate's training accuracy.
    """
    assert isinstance(learned_equation, LearnedPouringEquation)
    operating_point = SymbolicFillContext(
        sm.Scalar(FLOWING_TILT_ANGLE), sm.Scalar(FLOWING_FILL_LEVEL)
    )
    analytic_velocity = analytic_equation.symbolic_velocity(operating_point).evaluate()[
        0
    ]
    learned_velocity = learned_equation.symbolic_velocity(operating_point).evaluate()[0]

    assert learned_velocity != analytic_velocity
    assert learned_velocity == pytest.approx(
        analytic_velocity, abs=HEAD_MODEL_VELOCITY_AGREEMENT_TOLERANCE
    )
