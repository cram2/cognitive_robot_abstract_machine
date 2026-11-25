"""
The connection whose passive degree of freedom is a container's fill level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from krrood.patterns.called_positionally import called_positionally
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    Scalar,
    VariableParameters,
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

# %% fill-level source


class FillLevelSource(ABC):
    """
    What a liquid connection reads its fill velocity from: the semantic annotation of
    the container whose fill level it holds.
    """

    @property
    @abstractmethod
    def fill_connection(self) -> LiquidConnection | None:
        """
        The connection holding this container's fill level.
        """

    @abstractmethod
    def fill_velocity(self) -> Scalar:
        """
        :return: Symbolic rate of the container's normalized fill level.
        """


# %% compiled fill velocity


@dataclass(kw_only=True)
class CompiledFillVelocity:
    """
    The fill velocity of a liquid connection compiled for one world model version.
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
class LiquidConnection(PrismaticConnection, HasUpdateState):
    """
    Prismatic connection whose passive degree of freedom is a container's normalized
    fill level, integrated each physics step from the fill velocity of the
    :class:`FillLevelSource` it belongs to; the QP never commands it.
    """

    compiled_fill_velocity: CompiledFillVelocity | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """
    The fill velocity compiled for :meth:`update_state`, recompiled whenever the world
    model changes.
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

    def symbolic_fill_velocity(self) -> Scalar:
        """
        :return: The fill velocity of the container this connection belongs to; zero if
            it belongs to none.
        """
        for container in self._world.get_semantic_annotations_by_type(FillLevelSource):
            if container.fill_connection is self:
                return container.fill_velocity()
        return sm.Scalar(0.0)

    def current_compiled_fill_velocity(self) -> CompiledFillVelocity:
        """
        :return: The fill velocity compiled for the current world model, compiling it
            only when the model changed since the last call.
        """
        model_version = self._world._model_manager.version
        cached = self.compiled_fill_velocity
        if cached is not None and cached.model_version == model_version:
            return cached
        velocity = self.symbolic_fill_velocity()
        variables = velocity.free_variables()
        self.compiled_fill_velocity = CompiledFillVelocity(
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
        freedom's limits, which the fill velocity does not guard itself.

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
