from typing import List, Dict, DefaultDict

from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom


class WeightGain:
    _name: str
    gains: List[DefaultDict[Derivatives, Dict[DegreeOfFreedom, float]]]

    def __init__(
        self,
        name: str,
        gains: List[DefaultDict[Derivatives, Dict[DegreeOfFreedom, float]]],
    ):
        self._name = name
        self.gains = gains

    @property
    def name(self) -> str:
        return str(self._name)


class LinearWeightGain(WeightGain):
    def __init__(
        self,
        name: str,
        gains: List[DefaultDict[Derivatives, Dict[DegreeOfFreedom, float]]],
    ):
        super().__init__(name, gains)


class QuadraticWeightGain(WeightGain):
    def __init__(
        self,
        name: str,
        gains: List[DefaultDict[Derivatives, Dict[DegreeOfFreedom, float]]],
    ):
        super().__init__(name, gains)
