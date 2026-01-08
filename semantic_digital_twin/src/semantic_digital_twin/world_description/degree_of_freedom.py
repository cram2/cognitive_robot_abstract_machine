from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import Dict, Any

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json
from .world_entity import WorldEntityWithID
from ..datastructures.prefixed_name import PrefixedName
from ..exceptions import UsageError
from ..spatial_types.derivatives import Derivatives, DerivativeMap


@dataclass(eq=False, init=False)
class PositionVariable(sm.FloatVariable):
    """
    Describes the position of a degree of freedom.
    """

    dof: DegreeOfFreedom = field(kw_only=True)
    """ Backreference """

    def __init__(self, name: str, dof: DegreeOfFreedom):
        super().__init__(name)
        self.dof = dof

    def resolve(self) -> float:
        return self.dof._world.state[self.dof.id].position


@dataclass(eq=False)
class VelocityVariable(sm.FloatVariable):
    """
    Describes the velocity of a degree of freedom.
    """

    dof: DegreeOfFreedom = field(kw_only=True)
    """ Backreference """

    def __init__(self, name: str, dof: DegreeOfFreedom):
        super().__init__(name)
        self.dof = dof

    def resolve(self) -> float:
        return self.dof._world.state[self.dof.id].velocity


@dataclass(eq=False)
class AccelerationVariable(sm.FloatVariable):
    """
    Describes the acceleration of a degree of freedom.
    """

    dof: DegreeOfFreedom = field(kw_only=True)
    """ Backreference """

    def __init__(self, name: str, dof: DegreeOfFreedom):
        super().__init__(name)
        self.dof = dof

    def resolve(self) -> float:
        return self.dof._world.state[self.dof.id].acceleration


@dataclass(eq=False)
class JerkVariable(sm.FloatVariable):
    """
    Describes the jerk of a degree of freedom.
    """

    dof: DegreeOfFreedom = field(kw_only=True)
    """ Backreference """

    def __init__(self, name: str, dof: DegreeOfFreedom):
        super().__init__(name)
        self.dof = dof

    def resolve(self) -> float:
        return self.dof._world.state[self.dof.id].jerk


@dataclass(eq=False)
class DegreeOfFreedom(WorldEntityWithID, SubclassJSONSerializer):
    """
    A class representing a degree of freedom in a world model with associated derivatives and limits.

    This class manages a variable that can freely change within specified limits, tracking its position,
    velocity, acceleration, and jerk. It maintains symbolic representations for each derivative order
    and provides methods to get and set limits for these derivatives.
    """

    lower_limits: DerivativeMap[float] = field(default_factory=DerivativeMap)
    upper_limits: DerivativeMap[float] = field(default_factory=DerivativeMap)
    """
    Lower and upper bounds for each derivative
    """

    variables: DerivativeMap[sm.FloatVariable] = field(
        default_factory=DerivativeMap, init=False
    )
    """
    Symbolic representations for each derivative
    """

    has_hardware_interface: bool = False
    """
    Whether this DOF is linked to a controller and can therefore respond to control commands.

    E.g. the caster wheels of a PR2 have dofs, but they are not directly controlled. 
    Instead a the omni drive connection is directly controlled and a low level controller translates these commands
    to commands for the caster wheels.

    A door hinge also has a dof that cannot be controlled.
    """

    def __post_init__(self):
        self.lower_limits = self.lower_limits or DerivativeMap()
        self.upper_limits = self.upper_limits or DerivativeMap()

    def create_variables(self):
        """
        Creates a variable for each derivative, that refer to the corresponding values of this dof.
        """
        assert self._world is not None
        self.variables.data[Derivatives.position] = PositionVariable(
            name=str(PrefixedName("position", prefix=str(self.name))), dof=self
        )
        self.variables.data[Derivatives.velocity] = VelocityVariable(
            name=str(PrefixedName("velocity", prefix=str(self.name))), dof=self
        )
        self.variables.data[Derivatives.acceleration] = AccelerationVariable(
            name=str(PrefixedName("acceleration", prefix=str(self.name))), dof=self
        )
        self.variables.data[Derivatives.jerk] = JerkVariable(
            name=str(PrefixedName("jerk", prefix=str(self.name))), dof=self
        )

    def has_position_limits(self) -> bool:
        try:
            lower_limit = self.lower_limits.position
            upper_limit = self.upper_limits.position
            return lower_limit is not None or upper_limit is not None
        except KeyError:
            return False

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "lower_limits": self.lower_limits.to_json(),
            "upper_limits": self.upper_limits.to_json(),
            "name": self.name.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> DegreeOfFreedom:
        uuid = from_json(data["id"])
        lower_limits = DerivativeMap.from_json(data["lower_limits"], **kwargs)
        upper_limits = DerivativeMap.from_json(data["upper_limits"], **kwargs)
        return cls(
            name=PrefixedName.from_json(data["name"]),
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            id=uuid,
        )

    def __deepcopy__(self, memo):
        result = DegreeOfFreedom(
            lower_limits=deepcopy(self.lower_limits),
            upper_limits=deepcopy(self.upper_limits),
            name=deepcopy(self.name),
            has_hardware_interface=self.has_hardware_interface,
            id=self.id,
        )
        result._world = self._world
        # there can't be two symbols with the same name anyway
        result.variables = self.variables
        return result

    def _overwrite_dof_limits(
        self,
        new_lower_limits: DerivativeMap[float],
        new_upper_limits: DerivativeMap[float],
    ):
        """
        Overwrites the degree-of-freedom (DOF) limits for a range of derivatives. This updates
        lower and upper limits based on the given new limits. For each derivative, if the
        new limit is provided and it is more restrictive than the original limit, the limit
        will be updated accordingly.

        :param new_lower_limits: A mapping of new lower limits for the specified derivatives.
            If a new lower limit is None, no change is applied for that derivative.
        :param new_upper_limits: A mapping of new upper limits for the specified derivatives.
            If a new upper limit is None, no change is applied for that derivative.
        """
        if not isinstance(self.variables.position, sm.FloatVariable):
            raise UsageError(
                message="Cannot overwrite limits of mimic DOFs, use .raw_dof._overwrite_dof_limits instead."
            )
        for derivative in Derivatives.range(Derivatives.position, Derivatives.jerk):
            if new_lower_limits.data[derivative] is not None:
                if self.lower_limits.data[derivative] is None:
                    self.lower_limits.data[derivative] = new_lower_limits.data[
                        derivative
                    ]
                else:
                    self.lower_limits.data[derivative] = max(
                        new_lower_limits.data[derivative],
                        self.lower_limits.data[derivative],
                    )
            if new_upper_limits.data[derivative] is not None:
                if self.upper_limits.data[derivative] is None:
                    self.upper_limits.data[derivative] = new_upper_limits.data[
                        derivative
                    ]
                else:
                    self.upper_limits.data[derivative] = min(
                        new_upper_limits.data[derivative],
                        self.upper_limits.data[derivative],
                    )
