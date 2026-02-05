from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulatorAdditionalProperty:
    """
    Class representing an additional property for a simulator.
    """

    ...


@dataclass
class HasSimulatorProperties:
    """
    Mixin class to add simulator additional properties to a data class.
    """

    simulator_additional_properties: List[SimulatorAdditionalProperty] = field(
        default_factory=list, kw_only=True
    )
    """
    A list of additional properties for the simulator, it can contain properties of multiple simulators.
    """
