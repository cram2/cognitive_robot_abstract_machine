"""Pouring equations whose head-above-lip is a learned l4casadi model.

Identical to the analytic equations except ``head_above_lip`` evaluates a trained surrogate.
Because l4casadi accepts CasADi SX (giskardpy's backend), the learned head returns an SX-backed
:class:`~krrood.symbolic_math.symbolic_math.Scalar` and flows through ``symbolic_velocity``,
``exit_velocity``, the transfer gate, the projectile landing, and the terminal-state prediction
exactly like the analytic head -- no controller changes.

Neither a torch module nor the compiled l4casadi wrapper can cross a process boundary, so the
equations carry a :class:`LearnedHeadModelReference` instead: a serializable descriptor from which
every process (client or server, on any machine running the same workspace) rematerializes the
model locally on first use. This mirrors how
:class:`~semantic_digital_twin.world_description.connections.LiquidTransferCoupling` survives
world synchronization while the symbolic coupling is rebuilt per world.

..note:: :mod:`torch` and :mod:`l4casadi` are imported lazily inside the materialization step,
    so this module (and everything that deserializes worlds containing these equations) stays
    importable without them until a learned head is actually evaluated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, TYPE_CHECKING

import casadi
from typing_extensions import Self

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.exceptions import (
    LearnedModelGeometryMismatchError,
    MissingFillEquationError,
    MissingLearnedModelCheckpointError,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    FillContext,
    GatedArticulatedPouringEquation,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
    from semantic_digital_twin.world import World

SHIPPED_HEAD_SURROGATE_CHECKPOINT: str = (
    "semantic_digital_twin/resources/learned_models/head_surrogate.pt"
)
"""Workspace-relative path of the committed reference checkpoint.

Trained for the Jeroen cup geometry (container height 0.16 m, width 0.07 m, derived from
``resources/stl/jeroen_cup.stl``) with the default hidden width. Use it as a working default;
train a dedicated checkpoint for any other cup geometry."""


@dataclass
class LearnedHeadModelReference(SubclassJSONSerializer):
    """
    Serializable reference to a trained head-above-lip surrogate checkpoint.

    Carries everything needed to rematerialize the torch model in another process: the
    checkpoint location and the network architecture. The torch weights themselves never cross
    the process boundary; both sides run against the same workspace, so the checkpoint file is
    reachable from either.
    """

    checkpoint_path: str
    """POSIX path to the ``state_dict`` checkpoint. A relative path is resolved against the
    workspace root, so it stays valid across machines running the same workspace; an absolute
    path is used verbatim."""

    trained_container_height: float
    """Container height the surrogate was trained for, in metres. Guards against pairing the
    checkpoint with a differently shaped cup."""

    trained_container_width: float
    """Container width the surrogate was trained for, in metres."""

    hidden_width: int = 64
    """Hidden-layer width of the surrogate network, needed to rebuild it before loading the
    weights."""

    @staticmethod
    def workspace_root() -> Path:
        """
        The root directory of the workspace checkout, the anchor for relative checkpoint paths.

        :return: The directory containing the workspace packages.
        """
        return Path(files("semantic_digital_twin")).parents[2]

    def resolved_checkpoint_path(self) -> Path:
        """
        :return: The absolute checkpoint path; relative paths resolve against the workspace root.
        """
        path = Path(self.checkpoint_path)
        if path.is_absolute():
            return path
        return self.workspace_root() / path

    def load_torch_model(self) -> Any:
        """
        Rebuild the surrogate network and load the checkpoint weights.

        :return: The evaluated :class:`~semantic_digital_twin.physics.equations.head_surrogate_network.HeadSurrogate`.
        :raises MissingLearnedModelCheckpointError: if the checkpoint file does not exist.
        """
        import torch

        from semantic_digital_twin.physics.equations.head_surrogate_network import (
            HeadSurrogate,
        )

        path = self.resolved_checkpoint_path()
        if not path.exists():
            raise MissingLearnedModelCheckpointError(checkpoint_path=path)
        network = HeadSurrogate(hidden_width=self.hidden_width)
        network.load_state_dict(torch.load(str(path), weights_only=True))
        return network.eval()

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "checkpoint_path": self.checkpoint_path,
            "trained_container_height": self.trained_container_height,
            "trained_container_width": self.trained_container_width,
            "hidden_width": self.hidden_width,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            checkpoint_path=data["checkpoint_path"],
            trained_container_height=data["trained_container_height"],
            trained_container_width=data["trained_container_width"],
            hidden_width=data["hidden_width"],
        )


@dataclass
class HasLearnedHead:
    """
    Mixin replacing an articulated pouring equation's analytic head with a learned surrogate.

    Combine with :class:`ArticulatedPouringEquation` (or a subclass); the mixin overrides
    ``head_above_lip`` and extends the JSON round trip with the model reference. The l4casadi
    wrapper is a process-local cache, rebuilt lazily on first evaluation and never serialized.
    """

    GEOMETRY_ABSOLUTE_TOLERANCE: ClassVar[float] = 1e-6
    """Maximum allowed deviation between the trained-for and the equation's container geometry,
    in metres."""

    model_reference: LearnedHeadModelReference = field(kw_only=True)
    """Serializable reference to the trained head surrogate."""

    _l4casadi_head: Optional[Any] = field(
        default=None, init=False, repr=False, compare=False
    )
    """Process-local l4casadi wrapper around the loaded torch model."""

    _l4casadi_cache: ClassVar[Dict[str, Any]] = {}
    """Process-local cache of l4casadi wrappers keyed by generated function name, so equations
    built from the same checkpoint and geometry share one generated model instead of recompiling
    per equation instance."""

    def head_above_lip(self, context: FillContext) -> Scalar:
        """
        Height of the liquid surface above the pouring lip, evaluated by the learned surrogate.

        The surrogate output is clamped to be non-negative like the analytic head: an MSE-trained
        network routinely undershoots below zero in the non-pouring region, and a negative head
        would flip the drain's sign so an upright source *gains* liquid.

        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Symbolic head above the lip.
        """
        if self._l4casadi_head is None:
            self._l4casadi_head = self._materialize_head_model()
        tilt_sx = context.tilt_expression.casadi_sx
        fill_sx = context.fill_position.casadi_sx
        head_sx = self._l4casadi_head(casadi.horzcat(tilt_sx, fill_sx))
        return sm.max(sm.Scalar(0.0), sm.Scalar(head_sx[0, 0]))

    def _materialize_head_model(self) -> Any:
        """
        Load the torch model from the reference and wrap it for CasADi.

        :return: The callable l4casadi wrapper.
        :raises LearnedModelGeometryMismatchError: if the checkpoint was trained for a different
            container geometry than this equation describes.
        """
        import l4casadi

        self._validate_geometry_matches_reference()
        name = self._generated_function_name()
        cached = self._l4casadi_cache.get(name)
        if cached is not None:
            return cached
        torch_model = self.model_reference.load_torch_model()
        wrapper = l4casadi.L4CasADi(torch_model, name=name, batched=True)
        self._l4casadi_cache[name] = wrapper
        return wrapper

    def _validate_geometry_matches_reference(self) -> None:
        reference = self.model_reference
        height_deviation = abs(
            reference.trained_container_height - self.container_height
        )
        width_deviation = abs(reference.trained_container_width - self.container_width)
        if (
            height_deviation > self.GEOMETRY_ABSOLUTE_TOLERANCE
            or width_deviation > self.GEOMETRY_ABSOLUTE_TOLERANCE
        ):
            raise LearnedModelGeometryMismatchError(
                trained_container_height=reference.trained_container_height,
                trained_container_width=reference.trained_container_width,
                equation_container_height=self.container_height,
                equation_container_width=self.container_width,
            )

    def _generated_function_name(self) -> str:
        """
        A deterministic C identifier for the l4casadi generated code.

        Derived from the checkpoint and geometry (never from object identity), so equal
        references map to the same generated code across instances and processes.
        """
        checkpoint_stem = re.sub(
            r"\W", "_", self.model_reference.resolved_checkpoint_path().stem
        )
        geometry_tag = re.sub(
            r"\W",
            "_",
            f"{self.model_reference.hidden_width}"
            f"_{self.container_height}_{self.container_width}",
        )
        return f"learned_head_{checkpoint_stem}_{geometry_tag}"

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["model_reference"] = self.model_reference.to_json()
        return result

    @classmethod
    def _constructor_arguments_from_json(
        cls, data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        arguments = super()._constructor_arguments_from_json(data, **kwargs)
        arguments["model_reference"] = LearnedHeadModelReference.from_json(
            data["model_reference"], **kwargs
        )
        return arguments

    def with_gate(self, gate: Scalar) -> GatedLearnedPouringEquation:
        """
        The gated counterpart of this equation, keeping the learned head.

        :param gate: The shared transfer gate in ``[0, 1]``.
        :return: A gated learned equation with this equation's parameters.
        """
        return GatedLearnedPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
            gate=gate,
            model_reference=self.model_reference,
        )


@dataclass
class LearnedPouringEquation(HasLearnedHead, ArticulatedPouringEquation):
    """Articulated (ungated) pouring equation with a learned head-above-lip."""


@dataclass
class GatedLearnedPouringEquation(HasLearnedHead, GatedArticulatedPouringEquation):
    """
    Gated pouring equation with a learned head-above-lip: the source's drain after coupling.

    The gate itself is symbolic and never serialized. World synchronization stores the drain
    ungated (see :meth:`LiquidConnection.to_json`); an equation round-tripped directly (e.g. in
    an action goal) starts with the gate open. Either way the transfer coupling re-gates it when
    rebuilt against the local world.
    """

    def ungated(self) -> LearnedPouringEquation:
        """
        :return: The learned drain with this equation's parameters, without the symbolic gate.
        """
        return LearnedPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
            model_reference=self.model_reference,
        )


def couple_source_with_learned_head(
    receiver: HasFillLevel,
    source: HasFillLevel,
    world: World,
    model_reference: LearnedHeadModelReference,
) -> None:
    """
    Couple a source to a receiver so the whole transfer uses the learned head.

    Swaps the source's drain to an ungated learned equation and re-establishes the transfer
    coupling, so the gate, the receiver inflow, and the projectile are all built from the learned
    head. The gated learned drain is installed by the coupling itself via
    :meth:`HasLearnedHead.with_gate`, keeping drain and inflow volume-consistent. The source may
    already be coupled (analytically or learned); the previous coupling is replaced.

    :param receiver: The container to be filled.
    :param source: The container to pour from; must already have a fill equation.
    :param world: The world both containers live in.
    :param model_reference: Reference to the trained head surrogate for the source's geometry.

    :raises MissingFillEquationError: if the source was never initialized with a fill level.
    """
    equation = source.fill_equation
    if equation is None:
        raise MissingFillEquationError(source=source)
    receiver.recouple_outflow_from(
        source=source,
        world=world,
        fill_equation=LearnedPouringEquation(
            container_height=equation.container_height,
            container_width=equation.container_width,
            outflow_rate_constant=equation.outflow_rate_constant,
            discharge_coefficient=equation.discharge_coefficient,
            model_reference=model_reference,
        ),
    )
