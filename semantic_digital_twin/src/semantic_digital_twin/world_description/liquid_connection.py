"""
The connection whose passive degree of freedom is a container's fill level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, Any

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import from_json, to_json
from krrood.patterns.called_positionally import called_positionally
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    Scalar,
    VariableParameters,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    FillContext,
    InflowEquation,
    PouringEquation,
)
from semantic_digital_twin.world_description.connections import (
    HasUpdateState,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
)
from semantic_digital_twin.world_description.exceptions import (
    MissingFillLevelLimitsError,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

# %% compiled fill velocity


@dataclass(kw_only=True)
class CompiledFillVelocity:
    """
    The fill velocity of a liquid connection compiled once for the equations and world
    model it was built from.
    """

    outflow_equation: PouringEquation | None
    """
    The drain the function was compiled from.
    """

    inflow_equation: InflowEquation | None
    """
    The inflow the function was compiled from.
    """

    model_version: int
    """
    The world model version the function was compiled at.
    """

    variables: list[FloatVariable]
    """
    The free variables the function reads, in the order it takes them.
    """

    function: CompiledFunction
    """
    The compiled fill velocity.
    """

    def is_compiled_for(
        self,
        *,
        outflow_equation: PouringEquation | None,
        inflow_equation: InflowEquation | None,
        model_version: int,
    ) -> bool:
        """
        :return: Whether the function was compiled from these equations at this world
            model version.
        """
        return (
            self.outflow_equation is outflow_equation
            and self.inflow_equation is inflow_equation
            and self.model_version == model_version
        )

    def evaluate(self) -> float:
        """
        :return: The fill velocity at the current values of the variables.
        """
        values = np.array(
            [variable.resolve() for variable in self.variables], dtype=np.float64
        )
        return float(self.function(values)[0])


# %% liquid connection


@dataclass(eq=False, kw_only=True)
class LiquidConnection(PrismaticConnection, HasUpdateState, FillContext):
    """
    Prismatic connection whose passive degree of freedom is a container's normalized
    fill level, integrated from :attr:`outflow_equation` and :attr:`inflow_equation`
    each physics step; the QP never commands it.
    """

    outflow_equation: PouringEquation | None = field(default=None, init=False)
    """
    Equation of the liquid leaving this container, e.g. as it tilts to pour.
    """

    inflow_equation: InflowEquation | None = field(default=None, init=False)
    """
    Equation of the liquid entering this container from a source.
    """

    coupled_source_drain: PouringEquation | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """
    The source drain :attr:`inflow_equation` was built from; a source carrying another
    drain, e.g. one a synchronized world replaced, marks the coupling stale.
    """

    compiled_fill_velocity: CompiledFillVelocity | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """
    The fill velocity compiled for :meth:`update_state`, recompiled whenever an equation
    or the world model changes.
    """

    @property
    def active_dofs(self) -> list[DegreeOfFreedom]:
        """
        :return: Empty list, so the QP solver cannot command the fill velocity directly.
        """
        return []

    @property
    def passive_dofs(self) -> list[DegreeOfFreedom]:
        """
        :return: The fill-level degree of freedom, which stays in the world state and
            forward kinematics without being a QP variable.
        """
        return [self.raw_dof]

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        :return: The JSON of this connection with its drain and without its inflow,
            which the receiving world rebuilds from the transfer coupling.
        """
        result = super().to_json(**kwargs)
        result["outflow_equation"] = to_json(self.outflow_equation)
        return result

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> LiquidConnection:
        """
        :return: The connection the JSON describes, with its drain.
        """
        instance = super()._from_json(data=data, **kwargs)
        raw_outflow = data.get("outflow_equation")
        if raw_outflow is not None:
            instance.outflow_equation = from_json(raw_outflow)
        return instance

    @called_positionally
    def copy_for_world(self, world: World) -> LiquidConnection:
        """
        :param world: The target world.
        :return: A copy of this connection for ``world`` with the ungated drain; the
            inflow is rebuilt there by the receiver's transfer coupling.
        """
        copy = super().copy_for_world(world)
        if self.outflow_equation is not None:
            copy.outflow_equation = self.outflow_equation.drop_gate()
        return copy

    @property
    def tilt_expression(self) -> Scalar:
        """
        :return: Symbolic tilt angle of the container from the vertical, in radians.
        """
        root_T_child = self._world.compose_forward_kinematics_expression(
            self._world.root, self.child
        )
        return FillContext.tilt_from_vertical(root_T_container=root_T_child)

    @property
    def fill_position(self) -> Scalar:
        """
        :return: Symbolic normalized fill level, the position of the fill-level degree
            of freedom.
        """
        return self.dof.variables.position

    def symbolic_fill_velocity(self) -> Scalar:
        """
        :return: The fill level's rate from :attr:`outflow_equation` and
            :attr:`inflow_equation` together, zero without either.
        """
        velocity = sm.Scalar(0.0)
        for equation in (self.outflow_equation, self.inflow_equation):
            if equation is not None:
                velocity = velocity + equation.symbolic_velocity(context=self)
        return velocity

    def current_compiled_fill_velocity(self) -> CompiledFillVelocity:
        """
        :return: The fill velocity compiled for the current equations and world model,
            compiling it only when either changed since the last call.
        """
        model_version = self._world._model_manager.version
        cached = self.compiled_fill_velocity
        if cached is not None and cached.is_compiled_for(
            outflow_equation=self.outflow_equation,
            inflow_equation=self.inflow_equation,
            model_version=model_version,
        ):
            return cached
        velocity = self.symbolic_fill_velocity()
        variables = velocity.free_variables()
        self.compiled_fill_velocity = CompiledFillVelocity(
            outflow_equation=self.outflow_equation,
            inflow_equation=self.inflow_equation,
            model_version=model_version,
            variables=variables,
            function=velocity.compile(
                parameters=VariableParameters.from_lists(variables)
            ),
        )
        return self.compiled_fill_velocity

    @called_positionally
    def update_state(self, dt: float) -> None:
        """
        Advances the fill level by one physics step, clamped to the fill degree of
        freedom's limits, which the equations do not guard themselves.

        :param dt: Time elapsed since the previous step, in seconds.
        :raises MissingFillLevelLimitsError: If the fill degree of freedom has no
            position limits to clamp to.
        """
        limits = self.raw_dof.limits
        if limits.lower.position is None or limits.upper.position is None:
            raise MissingFillLevelLimitsError(connection_name=self.name)
        state = self._world.state
        velocity = self.current_compiled_fill_velocity().evaluate()
        integrated_position = state[self.raw_dof.id].position + velocity * dt
        state[self.raw_dof.id].position = min(
            max(integrated_position, limits.lower.position), limits.upper.position
        )
