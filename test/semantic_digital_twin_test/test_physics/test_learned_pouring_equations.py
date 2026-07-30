"""
Tests for the pouring equations whose head-above-lip is a learned surrogate.

The learned head model must survive the JSON channels between a giskardpy client and
server (world synchronization and the motion statechart action goal) as well as the ORM
layer, and it must reproduce the analytic head closely enough for the MPC to linearize
it.
"""

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import krrood.symbolic_math.symbolic_math as sm  # noqa: E402
from krrood.adapters.json_serializer import from_json  # noqa: E402
from krrood.ormatic.data_access_objects.helper import to_dao  # noqa: E402
from krrood.ormatic.utils import create_engine  # noqa: E402

from semantic_digital_twin.datastructures.prefixed_name import (  # noqa: E402
    PrefixedName,
)
from semantic_digital_twin.exceptions import (  # noqa: E402
    LearnedModelGeometryMismatchError,
    MissingFillEquationError,
    MissingLearnedModelCheckpointError,
    NonArticulatedDrainError,
)
from semantic_digital_twin.orm.ormatic_interface import (  # noqa: E402
    Base,
    HasFillLevelDAO,
    LearnedPouringEquationDAO,
    PouringEquationDAO,
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
    SHIPPED_HEAD_SURROGATE_CONTAINER_HEIGHT,
    SHIPPED_HEAD_SURROGATE_CONTAINER_WIDTH,
    couple_source_with_learned_head,
    shipped_head_model_reference,
)
from semantic_digital_twin.physics.equations.pouring_equations import (  # noqa: E402
    ArticulatedPouringEquation,
    GatedArticulatedPouringEquation,
    PouringEquation,
    SymbolicFillContext,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel  # noqa: E402
from semantic_digital_twin.spatial_types import Vector3  # noqa: E402
from semantic_digital_twin.world_description.world_entity import Body  # noqa: E402

from ..test_semantic_annotations import test_liquid_transfer  # noqa: E402

CONTAINER_HEIGHT = 0.1
CONTAINER_WIDTH = 0.08
HIDDEN_WIDTH = 8

VALUE_RMSE_BOUND = 5e-3
"""
Maximum head-value RMSE, in metres, a trained surrogate may deviate from the analytic
head.
"""

TILT_GRADIENT_RMSE_BOUND = 0.1
"""
Maximum RMSE of the head's tilt gradient a trained surrogate may deviate from the
analytic head; this is the bound the MPC linearization relies on.
"""

requires_l4casadi = pytest.mark.skipif(
    importlib.util.find_spec("l4casadi") is None,
    reason="l4casadi is not installed",
)


# %% fixtures and helpers


@pytest.fixture
def model_reference(tmp_path) -> LearnedHeadModelReference:
    """
    A reference to a small randomly initialized surrogate checkpoint on disk.
    """
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
    """
    Numerically evaluate the equation's head-above-lip at ``(tilt, fill)``.
    """
    tilt_variable = sm.FloatVariable("test_head_tilt")
    fill_variable = sm.FloatVariable("test_head_fill")
    head = equation.head_above_lip(SymbolicFillContext(tilt_variable, fill_variable))
    return float(
        head.substitute([tilt_variable, fill_variable], [tilt, fill]).evaluate()[0]
    )


def _json_string_round_trip(equation):
    """
    Serialize through an actual JSON string, as the client/server channels do.
    """
    return from_json(json.loads(json.dumps(equation.to_json())))


def _constant_head_model_reference(
    checkpoint_path: Path, head_value: float
) -> LearnedHeadModelReference:
    """
    A surrogate checkpoint predicting ``head_value`` for every input, saved to disk.
    """
    network = HeadSurrogate(hidden_width=HIDDEN_WIDTH)
    with torch.no_grad():
        network.net[-1].weight.zero_()
        network.net[-1].bias.fill_(head_value)
    torch.save(network.state_dict(), str(checkpoint_path))
    return LearnedHeadModelReference(
        checkpoint_path=str(checkpoint_path),
        trained_container_height=CONTAINER_HEIGHT,
        trained_container_width=CONTAINER_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )


def _fidelity_errors(
    surrogate: HeadSurrogate, container_height: float, container_width: float
) -> tuple[float, float, float]:
    """
    Grid errors of a surrogate against the analytic head for one container geometry.

    :return: Value RMSE and tilt-gradient RMSE over the pouring region, and value RMSE
        against zero over the non-pouring region.
    """
    tilt_grid, fill_grid = np.meshgrid(
        np.linspace(0.0, math.pi / 2, 60), np.linspace(0.0, 1.0, 60)
    )
    points = np.stack([tilt_grid.ravel(), fill_grid.ravel()], axis=1).astype(np.float32)
    analytic_inputs = torch.tensor(points, requires_grad=True)
    analytic = analytic_head_torch(
        analytic_inputs[:, 0:1],
        analytic_inputs[:, 1:2],
        container_height,
        container_width,
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
    non_pouring_learned = learned.detach().numpy().flatten()[~pouring]
    return (
        float(np.sqrt(np.mean(value_error**2))),
        float(np.sqrt(np.mean(tilt_gradient_error**2))),
        float(np.sqrt(np.mean(non_pouring_learned**2))),
    )


# %% json round trip


@requires_l4casadi
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


# %% orm round trip


class TestLearnedEquationPersistence:
    """
    A learned equation persists under the pouring-equation DAO root.
    """

    @pytest.fixture
    def session(self):
        engine = create_engine("sqlite:///:memory:")
        session = Session(engine)
        Base.metadata.create_all(bind=session.bind)
        yield session
        Base.metadata.drop_all(session.bind)
        session.close()

    def test_has_fill_level_with_learned_equation_round_trips(
        self, session, model_reference
    ):
        annotation = HasFillLevel(
            root=Body(name=PrefixedName("learned_cup")),
            fill_equation=LearnedPouringEquation(
                container_height=CONTAINER_HEIGHT,
                container_width=CONTAINER_WIDTH,
                outflow_rate_constant=1.5,
                model_reference=model_reference,
            ),
        )
        session.add(to_dao(annotation))
        session.commit()

        stored_equation = session.scalar(select(PouringEquationDAO))
        assert isinstance(stored_equation, LearnedPouringEquationDAO)

        restored = session.scalar(select(HasFillLevelDAO)).from_dao()
        equation = restored.fill_equation
        assert isinstance(equation, LearnedPouringEquation)
        assert equation.container_height == CONTAINER_HEIGHT
        assert equation.container_width == CONTAINER_WIDTH
        assert equation.outflow_rate_constant == 1.5
        assert equation.model_reference == model_reference


# %% model reference resolution


class TestModelReferenceResolution:
    def test_relative_checkpoint_path_resolves_against_workspace_root(self):
        reference = shipped_head_model_reference()
        resolved = reference.resolved_checkpoint_path()
        workspace_root = LearnedHeadModelReference.workspace_root()
        assert resolved == workspace_root / Path(SHIPPED_HEAD_SURROGATE_CHECKPOINT)
        assert (workspace_root / "semantic_digital_twin").is_dir()

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


# %% shipped head model


class TestShippedHeadModel:
    """
    The committed default checkpoint works for its trained geometry on a fresh clone.
    """

    def test_shipped_reference_declares_the_trained_geometry(self):
        reference = shipped_head_model_reference()
        assert reference.checkpoint_path == SHIPPED_HEAD_SURROGATE_CHECKPOINT
        assert (
            reference.trained_container_height
            == SHIPPED_HEAD_SURROGATE_CONTAINER_HEIGHT
        )
        assert (
            reference.trained_container_width == SHIPPED_HEAD_SURROGATE_CONTAINER_WIDTH
        )
        assert reference.resolved_checkpoint_path().exists()

    def test_shipped_checkpoint_reproduces_the_analytic_head(self):
        """
        The shipped surrogate stays within the fidelity bounds for its trained geometry.
        """
        surrogate = shipped_head_model_reference().load_torch_model()
        value_rmse, tilt_gradient_rmse, non_pouring_rmse = _fidelity_errors(
            surrogate,
            SHIPPED_HEAD_SURROGATE_CONTAINER_HEIGHT,
            SHIPPED_HEAD_SURROGATE_CONTAINER_WIDTH,
        )
        assert value_rmse < VALUE_RMSE_BOUND
        assert tilt_gradient_rmse < TILT_GRADIENT_RMSE_BOUND
        assert non_pouring_rmse < VALUE_RMSE_BOUND


# %% geometry guard


class TestGeometryGuard:
    def test_geometry_mismatch_raises(self, model_reference):
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT * 2,
            container_width=CONTAINER_WIDTH,
            model_reference=model_reference,
        )
        with pytest.raises(LearnedModelGeometryMismatchError):
            _evaluate_head(equation, 0.3, 0.9)


# %% generated model cache


@requires_l4casadi
class TestGeneratedModelCache:
    """
    Distinct checkpoints must never share one generated l4casadi model.
    """

    def test_same_stem_different_checkpoints_do_not_collide(self, tmp_path):
        """
        Two checkpoints named alike but holding different weights must each evaluate to
        their own head, not to whichever was compiled first.
        """
        heads = []
        for directory_name, head_value in [("first", 0.2), ("second", 0.6)]:
            checkpoint_directory = tmp_path / directory_name
            checkpoint_directory.mkdir()
            reference = _constant_head_model_reference(
                checkpoint_directory / "head_surrogate.pt", head_value
            )
            equation = LearnedPouringEquation(
                container_height=CONTAINER_HEIGHT,
                container_width=CONTAINER_WIDTH,
                model_reference=reference,
            )
            head = _evaluate_head(equation, 0.4, 0.8)
            assert head == pytest.approx(head_value, abs=1e-6)
            heads.append(head)
        assert heads[0] != heads[1]


# %% polymorphic gating


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

    def test_ungated_keeps_parameters_and_model_reference(self, model_reference):
        gated = GatedLearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            outflow_rate_constant=2.0,
            discharge_coefficient=0.4,
            gate=sm.Scalar(0.25),
            model_reference=model_reference,
        )
        ungated = gated.ungated()

        assert type(ungated) is LearnedPouringEquation
        assert ungated.container_height == gated.container_height
        assert ungated.container_width == gated.container_width
        assert ungated.outflow_rate_constant == gated.outflow_rate_constant
        assert ungated.discharge_coefficient == gated.discharge_coefficient
        assert ungated.model_reference == model_reference

    @requires_l4casadi
    def test_ungated_head_equals_gated_head(self, model_reference):
        gated = GatedLearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            gate=sm.Scalar(0.25),
            model_reference=model_reference,
        )
        assert _evaluate_head(gated.ungated(), 0.4, 0.8) == pytest.approx(
            _evaluate_head(gated, 0.4, 0.8), abs=1e-9
        )


# %% physicality of the learned head


@requires_l4casadi
class TestLearnedHeadStaysPhysical:
    """
    The learned equation never produces a negative head or a filling drain.
    """

    @pytest.fixture
    def undershooting_model_reference(self, tmp_path) -> LearnedHeadModelReference:
        """
        A surrogate checkpoint whose raw output is negative over the whole input range,
        as an MSE-trained network is in the non-pouring region.
        """
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
        """
        A raw MSE-trained network undershoots below zero in the non-pouring region; the
        equation must clamp it, or a negative head would flip the drain's sign.
        """
        equation = LearnedPouringEquation(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            model_reference=undershooting_model_reference,
        )
        for tilt in np.linspace(-0.2, math.pi / 2, 12):
            for fill in np.linspace(0.0, 1.0, 9):
                assert _evaluate_head(equation, float(tilt), float(fill)) >= 0.0

    def test_learned_drain_never_fills_the_source(self, undershooting_model_reference):
        """
        The drain must be non-positive everywhere: an upright source must never gain
        liquid.
        """
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


# %% training target agreement


class TestAnalyticHeadTorchAgreement:
    """
    The torch training target is the same function as the symbolic analytic head.
    """

    def test_torch_head_matches_symbolic_head_on_grid(self):
        equation = ArticulatedPouringEquation(
            container_height=CONTAINER_HEIGHT, container_width=CONTAINER_WIDTH
        )
        for tilt in np.linspace(-0.2, math.pi / 2 + 0.2, 7):
            for fill in np.linspace(0.0, 1.0, 7):
                torch_head = float(
                    analytic_head_torch(
                        torch.tensor([[tilt]], dtype=torch.float64),
                        torch.tensor([[fill]], dtype=torch.float64),
                        CONTAINER_HEIGHT,
                        CONTAINER_WIDTH,
                    ).item()
                )
                assert _evaluate_head(equation, float(tilt), float(fill)) == (
                    pytest.approx(torch_head, abs=1e-9)
                )


# %% trainer configuration


class TestTrainerConfiguredWidth:
    """
    The trainer builds and advertises the network width it was configured with.
    """

    def _narrow_trainer(self) -> HeadSurrogateTrainer:
        return HeadSurrogateTrainer(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            hidden_width=HIDDEN_WIDTH,
            sample_count=32,
            epochs=1,
        )

    def test_train_uses_configured_hidden_width(self):
        surrogate = self._narrow_trainer().train()
        assert surrogate.net[0].out_features == HIDDEN_WIDTH
        assert surrogate.net[2].in_features == HIDDEN_WIDTH
        assert surrogate.net[2].out_features == HIDDEN_WIDTH

    def test_emitted_model_reference_rebuilds_the_trained_network(self, tmp_path):
        trainer = self._narrow_trainer()
        surrogate = trainer.train()
        checkpoint_path = tmp_path / "narrow_head_surrogate.pt"
        torch.save(surrogate.state_dict(), str(checkpoint_path))

        reference = trainer.model_reference(str(checkpoint_path))

        assert reference == LearnedHeadModelReference(
            checkpoint_path=str(checkpoint_path),
            trained_container_height=CONTAINER_HEIGHT,
            trained_container_width=CONTAINER_WIDTH,
            hidden_width=HIDDEN_WIDTH,
        )
        loaded = reference.load_torch_model()
        inputs = torch.tensor([[0.4, 0.8]])
        assert float(loaded(inputs).item()) == pytest.approx(
            float(surrogate(inputs).item())
        )


# %% trained surrogate fidelity


class TestTrainedSurrogateFidelity:
    """
    The trained surrogate reproduces the analytic head and its tilt gradient.
    """

    def test_trained_head_matches_analytic_values_and_gradients(self):
        """
        Value and tilt-gradient RMSE over the pouring region stay below the fidelity
        bounds the MPC relies on when linearizing the learned head.
        """
        surrogate = HeadSurrogateTrainer(
            container_height=CONTAINER_HEIGHT,
            container_width=CONTAINER_WIDTH,
            sample_count=8000,
            epochs=1500,
            gradient_weight=0.3,
        ).train()

        value_rmse, tilt_gradient_rmse, non_pouring_rmse = _fidelity_errors(
            surrogate, CONTAINER_HEIGHT, CONTAINER_WIDTH
        )
        assert (
            value_rmse < VALUE_RMSE_BOUND
        ), f"surrogate head value RMSE too high: {value_rmse:.3e}"
        assert (
            tilt_gradient_rmse < TILT_GRADIENT_RMSE_BOUND
        ), f"surrogate head gradient RMSE too high: {tilt_gradient_rmse:.3e}"
        assert non_pouring_rmse < VALUE_RMSE_BOUND, (
            f"surrogate head deviates from zero in the non-pouring region: "
            f"{non_pouring_rmse:.3e}"
        )


# %% mpc linearization path


@requires_l4casadi
class TestLearnedLinearization:
    """
    The learned equation linearizes and exits like the analytic one within fidelity
    bounds.
    """

    POURING_OPERATING_POINTS = [(1.0, 0.9), (1.2, 0.6)]
    """
    ``(tilt, fill)`` points well inside the pouring region of the shipped geometry.
    """

    @pytest.fixture
    def shipped_equations(
        self,
    ) -> tuple[LearnedPouringEquation, ArticulatedPouringEquation]:
        learned = LearnedPouringEquation(
            container_height=SHIPPED_HEAD_SURROGATE_CONTAINER_HEIGHT,
            container_width=SHIPPED_HEAD_SURROGATE_CONTAINER_WIDTH,
            model_reference=shipped_head_model_reference(),
        )
        analytic = ArticulatedPouringEquation(
            container_height=SHIPPED_HEAD_SURROGATE_CONTAINER_HEIGHT,
            container_width=SHIPPED_HEAD_SURROGATE_CONTAINER_WIDTH,
        )
        return learned, analytic

    @pytest.fixture
    def symbolic_context(self) -> SymbolicFillContext:
        return SymbolicFillContext(
            sm.FloatVariable("linearization_tilt"),
            sm.FloatVariable("linearization_fill"),
        )

    @staticmethod
    def _evaluate_at(
        expression: sm.Scalar,
        context: SymbolicFillContext,
        tilt: float,
        fill: float,
    ) -> float:
        return float(
            expression.substitute(
                [context.tilt_expression, context.fill_position], [tilt, fill]
            ).evaluate()[0]
        )

    def test_ode_jacobians_match_the_analytic_ones(
        self, shipped_equations, symbolic_context
    ):
        learned, analytic = shipped_equations
        learned_jacobians = learned.symbolic_ode_jacobians(
            symbolic_context.tilt_expression, symbolic_context.fill_position
        )
        analytic_jacobians = analytic.symbolic_ode_jacobians(
            symbolic_context.tilt_expression, symbolic_context.fill_position
        )
        jacobian_bound = (
            TILT_GRADIENT_RMSE_BOUND / SHIPPED_HEAD_SURROGATE_CONTAINER_HEIGHT
        )
        for tilt, fill in self.POURING_OPERATING_POINTS:
            for learned_jacobian, analytic_jacobian in zip(
                learned_jacobians, analytic_jacobians
            ):
                assert self._evaluate_at(
                    learned_jacobian, symbolic_context, tilt, fill
                ) == pytest.approx(
                    self._evaluate_at(analytic_jacobian, symbolic_context, tilt, fill),
                    abs=jacobian_bound,
                )

    def test_exit_velocity_matches_the_analytic_one(
        self, shipped_equations, symbolic_context
    ):
        learned, analytic = shipped_equations
        learned_exit_velocity = learned.exit_velocity(symbolic_context)
        analytic_exit_velocity = analytic.exit_velocity(symbolic_context)
        for tilt, fill in self.POURING_OPERATING_POINTS:
            assert self._evaluate_at(
                learned_exit_velocity, symbolic_context, tilt, fill
            ) == pytest.approx(
                self._evaluate_at(analytic_exit_velocity, symbolic_context, tilt, fill),
                abs=0.03,
            )


# %% learned coupling


@dataclass
class _ConstantDrainEquation(PouringEquation):
    """
    Drain with a constant normalized rate and no articulated cup geometry.
    """

    def symbolic_velocity(self, context) -> sm.Scalar:
        return sm.Scalar(-self.outflow_rate_constant)


def _learned_world(**build_arguments):
    """
    A tilting-source world built with the shared liquid-transfer world builder.
    """
    return test_liquid_transfer._build_world(
        source_class=test_liquid_transfer._TiltingContainer,
        source_axis=Vector3(0, 1, 0),
        **build_arguments,
    )


def _reference_for_source(tmp_path, source) -> LearnedHeadModelReference:
    """
    A random checkpoint reference matching the source's container geometry.
    """
    checkpoint_path = tmp_path / "head_surrogate.pt"
    torch.save(
        HeadSurrogate(hidden_width=HIDDEN_WIDTH).state_dict(), str(checkpoint_path)
    )
    return LearnedHeadModelReference(
        checkpoint_path=str(checkpoint_path),
        trained_container_height=source.fill_equation.container_height,
        trained_container_width=source.fill_equation.container_width,
        hidden_width=HIDDEN_WIDTH,
    )


@requires_l4casadi
class TestLearnedCoupling:
    """
    ``couple_source_with_learned_head`` swaps a live coupling onto the learned head.
    """

    def test_coupling_installs_gated_learned_drain_and_rebuilds_inflow(self, tmp_path):
        """
        Coupling an already-coupled source with a learned head replaces its gated drain
        by a :class:`GatedLearnedPouringEquation` carrying the reference and rebuilds
        the inflow.
        """
        world, source, receiver = _learned_world()
        previous_inflow = receiver.fill_connection.inflow_equation
        model_reference = _reference_for_source(tmp_path, source)

        couple_source_with_learned_head(receiver, source, world, model_reference)

        regated_drain = source.fill_equation
        assert isinstance(regated_drain, GatedLearnedPouringEquation)
        assert regated_drain.model_reference == model_reference
        assert receiver.fill_connection.inflow_equation is not previous_inflow
        assert receiver.inflow_coupling is not None

    def test_coupling_preserves_the_receivers_coupling_parameters(self, tmp_path):
        """
        Switching to the learned head keeps the receiver's tuned exit speed and gate
        sharpnesses instead of resetting them to the defaults.
        """
        world, source, receiver = _learned_world(
            exit_speed=0.7,
            height_gate_sharpness=42.0,
            overlap_gate_sharpness=17.0,
        )
        model_reference = _reference_for_source(tmp_path, source)

        couple_source_with_learned_head(receiver, source, world, model_reference)

        assert receiver.inflow_coupling.exit_speed == 0.7
        assert receiver.inflow_coupling.height_gate_sharpness == 42.0
        assert receiver.inflow_coupling.overlap_gate_sharpness == 17.0
        assert receiver.fill_connection.inflow_equation.exit_speed == 0.7


class TestLearnedCouplingGuards:
    """
    ``couple_source_with_learned_head`` rejects sources it cannot derive a drain from.
    """

    def test_source_without_fill_equation_raises(self, tmp_path):
        world, source, receiver = _learned_world(couple=False)
        model_reference = _reference_for_source(tmp_path, source)
        with world.modify_world():
            source.add_fill_equation(None)

        with pytest.raises(MissingFillEquationError):
            couple_source_with_learned_head(receiver, source, world, model_reference)

    def test_source_with_non_articulated_drain_raises(self, tmp_path):
        world, source, receiver = _learned_world(couple=False)
        model_reference = _reference_for_source(tmp_path, source)
        with world.modify_world():
            source.add_fill_equation(_ConstantDrainEquation())

        with pytest.raises(NonArticulatedDrainError):
            couple_source_with_learned_head(receiver, source, world, model_reference)
