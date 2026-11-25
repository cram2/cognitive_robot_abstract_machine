"""
Fill-level dynamics of containers that pour liquid out and receive it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Self

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    SubclassJSONSerializer,
)
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import Scalar
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3

from semantic_digital_twin.physics.equations.differential_equation import (
    DifferentialEquation,
)
from semantic_digital_twin.physics.exceptions import (
    NonPositiveContainerCapacityError,
    NonPositiveContainerGeometryError,
)

# %% fill context


class FillContext(ABC):
    """
    The symbolic tilt and fill level of a container that a fill equation is evaluated
    in.
    """

    @property
    @abstractmethod
    def tilt_expression(self) -> Scalar:
        """
        Symbolic tilt angle of the container from the vertical, in radians.
        """

    @property
    @abstractmethod
    def fill_position(self) -> Scalar:
        """
        Symbolic normalized fill level of the container in ``[0, 1]``.
        """

    @staticmethod
    def tilt_from_vertical(
        *, root_T_container: HomogeneousTransformationMatrix
    ) -> Scalar:
        """
        :param root_T_container: Symbolic pose of the container in a frame whose z axis
            points up.
        :return: Symbolic angle between the container's up axis and the vertical, in
            radians.
        """
        root_V_container_up = root_T_container.rotation_matrix @ Vector3.Z()
        return sm.safe_acos(root_V_container_up.z)


@dataclass(kw_only=True)
class SymbolicFillContext(FillContext):
    """
    A fill context made of given tilt and fill expressions, for callers that hold no
    fill connection.
    """

    tilt: Scalar
    """
    Symbolic tilt angle of the container from the vertical, in radians.
    """

    fill: Scalar
    """
    Symbolic normalized fill level of the container in ``[0, 1]``.
    """

    @property
    def tilt_expression(self) -> Scalar:
        """
        :return: The tilt this context was made of.
        """
        return self.tilt

    @property
    def fill_position(self) -> Scalar:
        """
        :return: The fill level this context was made of.
        """
        return self.fill


# %% container geometry


@dataclass(kw_only=True)
class RectangularContainerGeometry:
    """
    Extents of the 2-D rectangular container the fill equations model; the container
    tilts about its vertical centre axis.

    .. note:: This mixin carries no persistence of its own; the equations inheriting it
        map its fields within their own single-rooted DAO hierarchy.
    """

    container_height: float
    """
    Inner height of the rectangular container, in metres.
    """

    container_width: float
    """
    Inner width of the rectangular container, in metres.
    """

    capacity: float | None = None
    """
    Volume the container holds at full fill, in cubic metres; converts a normalized fill
    rate to a volume rate.

    Without one, the box of the width squared times the height.
    """

    def __post_init__(self) -> None:
        """
        :raises NonPositiveContainerGeometryError: If the height or width is not
            positive.
        :raises NonPositiveContainerCapacityError: If a given capacity is not positive.
        """
        if self.container_height <= 0.0 or self.container_width <= 0.0:
            raise NonPositiveContainerGeometryError(
                container_height=self.container_height,
                container_width=self.container_width,
            )
        if self.capacity is None:
            self.capacity = self.container_width**2 * self.container_height
        elif self.capacity <= 0.0:
            raise NonPositiveContainerCapacityError(capacity=self.capacity)


# %% fill equations


@dataclass(kw_only=True)
class FillEquation(DifferentialEquation):
    """
    First-order differential equation of a container's normalized fill level.
    """

    @classproperty
    def standard_gravity(cls) -> float:
        """
        Gravitational acceleration liquid falls with, in metres per second squared.
        """
        return 9.81

    @abstractmethod
    def symbolic_velocity(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level to evaluate the equation in.
        :return: Symbolic rate of the normalized fill level.
        """


@dataclass(kw_only=True)
class PouringEquation(SubclassJSONSerializer, FillEquation):
    """
    Fill equation of a container draining as it tilts.

    It serializes without any transfer gate: a gate is bound to the world it was built
    in, so the receiving world gates the drain again.
    """

    outflow_rate_constant: float = 1.0
    """
    Proportionality constant scaling the discharge gap to the normalized drain rate.
    """

    @abstractmethod
    def gate_drain(self, *, gate: Scalar) -> PouringEquation:
        """
        :param gate: The transfer gate in ``[0, 1]``.
        :return: This drain, flowing only while ``gate`` is open.
        """

    def drop_gate(self) -> PouringEquation:
        """
        :return: This drain without any transfer gate; an ungated drain returns itself.
        """
        return self

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        :return: The JSON of this drain without its transfer gate.
        """
        return DataclassJSONSerializer.to_json(instance=self.drop_gate(), **kwargs)

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        :return: The drain the JSON describes.
        """
        return DataclassJSONSerializer.from_json(data=data, clazz=cls, **kwargs)


@dataclass(kw_only=True)
class ArticulatedPouringEquation(RectangularContainerGeometry, PouringEquation):
    """
    Drain of a rectangular container of height ``A`` over a lip at horizontal distance
    ``r`` from its tilt axis, at normalized fill ``h`` and tilt ``α``::

        L(h)    = √((A · (1 − h))² + r²)
        φ(h)    = atan2(A · (1 − h), r) + repose angle
        d(α, h) = max(0, L(h) · sin(α − φ(h)))
        ḣ       = −k · d(α, h) / A
    """

    discharge_coefficient: float = 0.3
    """
    Coefficient scaling the Torricelli exit speed to a rim pour, whose thin film over
    the lip leaves far slower than an orifice's ``0.6``-``1.0``.
    """

    repose_angle: float = 0.0
    """
    Angle the contents' surface holds before it slumps, in radians; zero for a liquid,
    positive for solids that stand in a heap.
    """

    minimum_head: float = 0.01
    """
    Head the exit speed is computed from at least, in metres, which keeps its gradient
    bounded as the pour starts.
    """

    @property
    def lip_offset(self) -> float:
        """
        Horizontal distance of the pouring lip, the container's wall, from the tilt
        axis, in metres.
        """
        return self.container_width / 2

    def gate_drain(self, *, gate: Scalar) -> GatedArticulatedPouringEquation:
        """
        :param gate: The transfer gate in ``[0, 1]``.
        :return: This drain, flowing only while ``gate`` is open.
        """
        return GatedArticulatedPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            capacity=self.capacity,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
            repose_angle=self.repose_angle,
            minimum_head=self.minimum_head,
            gate=gate,
        )

    def head_above_lip(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level.
        :return: Symbolic height of the contents' surface above the lip, in metres; zero
            while the container does not pour.
        """
        dry_height = self.container_height * (1 - context.fill_position)
        lip_distance = sm.sqrt(dry_height**2 + self.lip_offset**2)
        onset = self.onset_tilt(fill=context.fill_position)
        return sm.max(
            sm.Scalar(0.0),
            lip_distance * sm.sin(context.tilt_expression - onset),
        )

    def onset_tilt(self, *, fill: sm.ScalarData) -> Scalar:
        """
        :param fill: Normalized fill level in ``[0, 1]``.
        :return: Symbolic tilt at which a container this full starts to pour, in
            radians; later by the repose angle for heaped contents.
        """
        dry_height = self.container_height * (1 - fill)
        return sm.atan2(dry_height, self.lip_offset) + self.repose_angle

    def exit_speed(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level.
        :return: Symbolic speed of the stream leaving the lip, ``C_d · √(2 g h)``, in
            metres per second, with the head ``h`` at least :attr:`minimum_head`.
        """
        head = sm.max(
            sm.Scalar(self.minimum_head), self.head_above_lip(context=context)
        )
        return self.discharge_coefficient * sm.sqrt(2 * self.standard_gravity * head)

    def symbolic_velocity(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level.
        :return: Symbolic rate of the normalized fill level.
        """
        return (
            -self.outflow_rate_constant
            * self.head_above_lip(context=context)
            / self.container_height
        )


@dataclass(kw_only=True)
class GatedArticulatedPouringEquation(ArticulatedPouringEquation):
    """
    Articulated drain that flows only while a differentiable gate is open, i.e. while
    the stream lands in the receiver it pours into, so the transfer spills nothing.
    """

    gate: Scalar = field(default_factory=lambda: sm.Scalar(1.0))
    """
    Symbolic transfer gate in ``[0, 1]``; ``1`` while the stream lands in the receiver.
    """

    def symbolic_velocity(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level.
        :return: The drain's rate scaled by the gate; zero while the gate is closed.
        """
        return self.gate * super().symbolic_velocity(context=context)

    def drop_gate(self) -> ArticulatedPouringEquation:
        """
        :return: The drain with this equation's parameters, without the gate.
        """
        return ArticulatedPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            capacity=self.capacity,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
            repose_angle=self.repose_angle,
            minimum_head=self.minimum_head,
        )


@dataclass(kw_only=True)
class InflowEquation(RectangularContainerGeometry, FillEquation):
    """
    Fill equation of a container receiving liquid, converting an inflow volume rate to a
    normalized fill rate by its capacity.
    """

    inflow: Scalar = field(default_factory=lambda: sm.Scalar(0.0))
    """
    The symbolic inflow volume rate entering this container, in cubic metres per second.
    """

    def symbolic_velocity(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level; the inflow does not depend on
            them.
        :return: Symbolic rate of the normalized fill level.
        """
        return self.inflow / self.capacity


# %% gated transfer


@dataclass(kw_only=True)
class TransferGate:
    """
    The two independent factors of the geometric transfer gate, each held by a different
    positioning task.
    """

    height: Scalar
    """
    Vertical factor, closing as the source lip sinks toward the receiver's opening.
    """

    overlap: Scalar
    """
    Horizontal factor, closing as the stream lands further from the opening.
    """

    @property
    def product(self) -> Scalar:
        """
        :return: The gate both factors together describe.
        """
        return self.height * self.overlap


@dataclass(kw_only=True)
class GatedInflowEquation(InflowEquation):
    """
    Inflow of a cup-to-cup transfer: :attr:`inflow` carries the source's drain as a
    volume rate and :attr:`transfer_gate` lets it in only while the stream lands in this
    receiver's opening.
    """

    exit_speed: float
    """
    Fallback speed of the stream leaving the source, in metres per second, which the
    landing point uses where the source models no exit speed of its own.
    """

    transfer_gate: TransferGate = field(
        default_factory=lambda: TransferGate(
            height=sm.Scalar(1.0), overlap=sm.Scalar(1.0)
        )
    )
    """
    The symbolic transfer gate, open while the stream lands in this receiver.
    """

    def symbolic_velocity(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level.
        :return: The fill rate scaled by the gate; zero while the gate is closed.
        """
        return self.transfer_gate.product * self.ungated_velocity(context=context)

    def ungated_velocity(self, *, context: FillContext) -> Scalar:
        """
        :param context: The symbolic tilt and fill level.
        :return: The fill rate the inflow gives while the gate is fully open.
        """
        return super().symbolic_velocity(context=context)
