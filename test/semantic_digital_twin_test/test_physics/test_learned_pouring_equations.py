"""Serialization tests for the learned pouring equations.

The learned head model must survive the JSON channels between a giskardpy client and server
(world synchronization and the motion statechart action goal). These tests prove the equation
round-trips through an actual JSON string and still evaluates the learned head afterwards.
"""

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("l4casadi")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import krrood.symbolic_math.symbolic_math as sm  # noqa: E402
from krrood.adapters.json_serializer import from_json  # noqa: E402

from semantic_digital_twin.exceptions import (  # noqa: E402
    LearnedModelGeometryMismatchError,
    MissingLearnedModelCheckpointError,
)
from semantic_digital_twin.physics.equations.head_surrogate_network import (  # noqa: E402
    HeadSurrogate,
)
from semantic_digital_twin.physics.equations.head_surrogate_training import (  # noqa: E402
    HeadSurrogateTrainer,
    analytic_head_torch,
)
from semantic_digital_twin.physics.equations.learned_pouring_equations import (  # noqa: E402
    GatedLearnedPouringEquation,
    LearnedHeadModelReference,
    LearnedPouringEquation,
    SHIPPED_HEAD_SURROGATE_CHECKPOINT,
    couple_source_with_learned_head,
)
from semantic_digital_twin.physics.equations.pouring_equations import (  # noqa: E402
    ArticulatedPouringEquation,
    GatedArticulatedPouringEquation,
    SymbolicFillContext,
)
from semantic_digital_twin.spatial_types import Vector3  # noqa: E402

from ..test_semantic_annotations import test_liquid_transfer  # noqa: E402

CONTAINER_HEIGHT = 0.1
CONTAINER_WIDTH = 0.08
HIDDEN_WIDTH = 8


@pytest.fixture
def model_reference(tmp_path) -> LearnedHeadModelReference:
    """A reference to a small randomly initialized surrogate checkpoint on disk."""
    checkpoint_path = tmp_path / "head_surrogate.pt"
    torch.manual_seed(0)
    network = HeadSurrogate(hidden_width=HIDDEN_WIDTH)
    torch.save(network.state_dict(), str(checkpoint_path))
    return LearnedHeadModelReference(
        checkpoint_path=str(checkpoint_path),
        trained_container_height=CONTAINER_HEIGHT,
        trained_container_width=CONTAINER_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )


def _evaluate_head(
    equation: ArticulatedPouringEquation, tilt: float, fill: float
) -> float:
    """Numerically evaluate the equation's head-above-lip at ``(tilt, fill)``."""
    tilt_variable = sm.FloatVariable("test_head_tilt")
    fill_variable = sm.FloatVariable("test_head_fill")
    head = equation.head_above_lip(SymbolicFillContext(tilt_variable, fill_variable))
    return float(
        head.substitute([tilt_variable, fill_variable], [tilt, fill]).evaluate()[0]
    )


def _json_string_round_trip(equation):
    """Serialize through an actual JSON string, as the client/server channels do."""
    return from_json(json.loads(json.dumps(equation.to_json())))


class TestLearnedEquationSurvivesProcessBoundary:
    def test_round_trip_preserves_learned_head_evaluation(self, model_reference):
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            outflow_rate_constant=1.5,
            model_reference=model_reference,
        )
        restored = _json_string_round_trip(equation)

        assert isinstance(restored, LearnedPouringEquation)
        assert restored.container_height == equation.container_height
        assert restored.container_width == equation.container_width
        assert restored.outflow_rate_constant == equation.outflow_rate_constant
        assert restored.model_reference == model_reference

        network = model_reference.load_torch_model()
        for tilt, fill in [(0.3, 0.9), (math.pi / 3, 0.5), (0.0, 1.0)]:
            expected = max(0.0, float(network(torch.tensor([[tilt, fill]])).item()))
            assert _evaluate_head(restored, tilt, fill) == pytest.approx(
                expected, abs=1e-5
            )

    def test_gated_round_trip_preserves_type_and_reopens_gate(self, model_reference):
        equation = GatedLearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            gate=sm.Scalar(0.0),
            model_reference=model_reference,
        )
        restored = _json_string_round_trip(equation)

        assert isinstance(restored, GatedLearnedPouringEquation)
        assert restored.model_reference == model_reference
        assert float(restored.gate.evaluate()[0]) == 1.0
        assert _evaluate_head(restored, 0.4, 0.8) == pytest.approx(
            _evaluate_head(equation, 0.4, 0.8), abs=1e-9
        )


class TestModelReferenceResolution:
    def test_relative_checkpoint_path_resolves_against_workspace_root(self):
        reference = LearnedHeadModelReference(
            checkpoint_path=SHIPPED_HEAD_SURROGATE_CHECKPOINT,
            trained_container_height=CONTAINER_HEIGHT,
            trained_container_width=CONTAINER_WIDTH,
        )
        resolved = reference.resolved_checkpoint_path()
        workspace_root = LearnedHeadModelReference.workspace_root()
        assert resolved == workspace_root / Path(SHIPPED_HEAD_SURROGATE_CHECKPOINT)
        assert (workspace_root / "semantic_digital_twin").is_dir()

    def test_shipped_reference_checkpoint_exists_and_loads(self):
        """The committed default checkpoint must resolve to a real file and load into the
        surrogate architecture, so the documented default path works on a fresh clone.
        """
        reference = LearnedHeadModelReference(
            checkpoint_path=SHIPPED_HEAD_SURROGATE_CHECKPOINT,
            trained_container_height=CONTAINER_HEIGHT,
            trained_container_width=CONTAINER_WIDTH,
            hidden_width=64,
        )
        assert reference.resolved_checkpoint_path().exists()
        assert reference.load_torch_model() is not None

    def test_absolute_checkpoint_path_is_used_verbatim(self, tmp_path):
        checkpoint_path = tmp_path / "head_surrogate.pt"
        reference = LearnedHeadModelReference(
            checkpoint_path=str(checkpoint_path),
            trained_container_height=CONTAINER_HEIGHT,
            trained_container_width=CONTAINER_WIDTH,
        )
        assert reference.resolved_checkpoint_path() == checkpoint_path

    def test_missing_checkpoint_raises_meaningful_error(self, tmp_path):
        reference = LearnedHeadModelReference(
            checkpoint_path=str(tmp_path / "does_not_exist.pt"),
            trained_container_height=CONTAINER_HEIGHT,
            trained_container_width=CONTAINER_WIDTH,
        )
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            model_reference=reference,
        )
        with pytest.raises(MissingLearnedModelCheckpointError):
            _evaluate_head(equation, 0.3, 0.9)


class TestGeometryGuard:
    def test_geometry_mismatch_raises(self, model_reference):
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT * 2,
            container_width=CONTAINER_WIDTH,
            model_reference=model_reference,
        )
        with pytest.raises(LearnedModelGeometryMismatchError):
            _evaluate_head(equation, 0.3, 0.9)


class TestPolymorphicGating:
    def test_with_gate_preserves_learned_head(self, model_reference):
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            outflow_rate_constant=2.0,
            discharge_coefficient=0.4,
            model_reference=model_reference,
        )
        gate = sm.Scalar(0.5)
        gated = equation.with_gate(gate)

        assert isinstance(gated, GatedLearnedPouringEquation)
        assert gated.model_reference == model_reference
        assert gated.container_height == equation.container_height
        assert gated.container_width == equation.container_width
        assert gated.outflow_rate_constant == equation.outflow_rate_constant
        assert gated.discharge_coefficient == equation.discharge_coefficient
        assert gated.gate is gate

    def test_with_gate_on_analytic_equation(self):
        equation = ArticulatedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            outflow_rate_constant=2.0,
        )
        gate = sm.Scalar(0.5)
        gated = equation.with_gate(gate)

        assert type(gated) is GatedArticulatedPouringEquation
        assert gated.container_height == equation.container_height
        assert gated.outflow_rate_constant == equation.outflow_rate_constant
        assert gated.gate is gate


class TestLearnedHeadStaysPhysical:
    """The learned equation never produces a negative head or a filling drain."""

    @pytest.fixture
    def undershooting_model_reference(self, tmp_path) -> LearnedHeadModelReference:
        """A surrogate checkpoint whose raw output is negative over the whole input range,
        as an MSE-trained network is in the non-pouring region."""
        checkpoint_path = tmp_path / "undershooting_head_surrogate.pt"
        torch.manual_seed(0)
        network = HeadSurrogate(hidden_width=HIDDEN_WIDTH)
        with torch.no_grad():
            network.net[-1].bias.fill_(-1.0)
        torch.save(network.state_dict(), str(checkpoint_path))
        return LearnedHeadModelReference(
            checkpoint_path=str(checkpoint_path),
            trained_container_height=CONTAINER_HEIGHT,
            trained_container_width=CONTAINER_WIDTH,
            hidden_width=HIDDEN_WIDTH,
        )

    def _evaluate_velocity(
        self, equation: LearnedPouringEquation, tilt: float, fill: float
    ) -> float:
        tilt_variable = sm.FloatVariable("test_velocity_tilt")
        fill_variable = sm.FloatVariable("test_velocity_fill")
        velocity = equation.symbolic_velocity(
            SymbolicFillContext(tilt_variable, fill_variable)
        )
        return float(
            velocity.substitute(
                [tilt_variable, fill_variable], [tilt, fill]
            ).evaluate()[0]
        )

    def test_learned_head_is_never_negative(self, undershooting_model_reference):
        """A raw MSE-trained network undershoots below zero in the non-pouring region; the
        equation must clamp it, or a negative head would flip the drain's sign."""
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            model_reference=undershooting_model_reference,
        )
        for tilt in np.linspace(-0.2, math.pi / 2, 12):
            for fill in np.linspace(0.0, 1.0, 9):
                assert _evaluate_head(equation, float(tilt), float(fill)) >= 0.0

    def test_learned_drain_never_fills_the_source(self, undershooting_model_reference):
        """The drain must be non-positive everywhere: an upright source must never gain liquid."""
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            model_reference=undershooting_model_reference,
        )
        for tilt in np.linspace(-0.2, math.pi / 2, 12):
            for fill in np.linspace(0.0, 1.0, 9):
                assert (
                    self._evaluate_velocity(equation, float(tilt), float(fill)) <= 0.0
                )


class TestTrainedSurrogateFidelity:
    """The trained surrogate reproduces the analytic head and its tilt gradient."""

    def test_trained_head_matches_analytic_values_and_gradients(self):
        """Value and tilt-gradient RMSE over the pouring region stay below the fidelity bounds
        the MPC relies on when linearizing the learned head."""
        surrogate = HeadSurrogateTrainer(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            sample_count=8000,
            epochs=1500,
            gradient_weight=0.3,
        ).train()

        tilt_grid, fill_grid = np.meshgrid(
            np.linspace(0.0, math.pi / 2, 60), np.linspace(0.0, 1.0, 60)
        )
        points = np.stack([tilt_grid.ravel(), fill_grid.ravel()], axis=1).astype(
            np.float32
        )
        analytic_inputs = torch.tensor(points, requires_grad=True)
        analytic = analytic_head_torch(
            analytic_inputs[:, 0:1],
            analytic_inputs[:, 1:2],
            CONTAINER_HEIGHT,
            CONTAINER_WIDTH,
        )
        analytic_gradient = torch.autograd.grad(analytic.sum(), analytic_inputs)[0]
        learned_inputs = torch.tensor(points, requires_grad=True)
        learned = surrogate(learned_inputs)
        learned_gradient = torch.autograd.grad(learned.sum(), learned_inputs)[0]

        pouring = analytic.detach().numpy().flatten() > 0.0
        value_error = (
            learned.detach().numpy().flatten() - analytic.detach().numpy().flatten()
        )[pouring]
        tilt_gradient_error = (
            learned_gradient.detach().numpy()[:, 0]
            - analytic_gradient.detach().numpy()[:, 0]
        )[pouring]
        value_rmse = float(np.sqrt(np.mean(value_error**2)))
        tilt_gradient_rmse = float(np.sqrt(np.mean(tilt_gradient_error**2)))

        assert (
            value_rmse < 5e-3
        ), f"surrogate head value RMSE too high: {value_rmse:.3e}"
        assert (
            tilt_gradient_rmse < 0.1
        ), f"surrogate head gradient RMSE too high: {tilt_gradient_rmse:.3e}"

        non_pouring_learned = learned.detach().numpy().flatten()[~pouring]
        non_pouring_rmse = float(np.sqrt(np.mean(non_pouring_learned**2)))
        assert non_pouring_rmse < 5e-3, (
            f"surrogate head deviates from zero in the non-pouring region: "
            f"{non_pouring_rmse:.3e}"
        )


class TestLearnedCoupling:
    """``couple_source_with_learned_head`` swaps a live coupling onto the learned head."""

    def test_coupling_installs_gated_learned_drain_and_rebuilds_inflow(self, tmp_path):
        """Coupling an already-coupled source with a learned head replaces its gated drain by a
        :class:`GatedLearnedPouringEquation` carrying the reference and rebuilds the inflow.
        """
        world, source, receiver = test_liquid_transfer.TestTransferGate()._build_world(
            source_class=test_liquid_transfer._TiltingContainer,
            source_axis=Vector3(0, 1, 0),
        )
        previous_inflow = receiver.fill_connection.inflow_equation
        checkpoint_path = tmp_path / "head_surrogate.pt"
        torch.save(
            HeadSurrogate(hidden_width=HIDDEN_WIDTH).state_dict(), str(checkpoint_path)
        )
        model_reference = LearnedHeadModelReference(
            checkpoint_path=str(checkpoint_path),
            trained_container_height=source.fill_equation.container_height,
            trained_container_width=source.fill_equation.container_width,
            hidden_width=HIDDEN_WIDTH,
        )

        couple_source_with_learned_head(receiver, source, world, model_reference)

        regated_drain = source.fill_equation
        assert isinstance(regated_drain, GatedLearnedPouringEquation)
        assert regated_drain.model_reference == model_reference
        assert receiver.fill_connection.inflow_equation is not previous_inflow
        assert receiver.inflow_coupling is not None
