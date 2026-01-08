from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Callable, List

from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
    Vector3,
)


@dataclass(eq=False, init=False)
class AuxiliaryVariable(FloatVariable):
    provider: Callable[[], float] = field(kw_only=True)

    def __init__(self, name: str, provider: Callable[[], float]):
        super().__init__(str(name))
        self.provider = provider

    def resolve(self) -> float:
        return float(self.provider())

    def __repr__(self):
        return str(self.name)


def create_point(name: PrefixedName, provider: Callable[[], List[float]]):
    return Point3(
        x=AuxiliaryVariable(
            name=str(PrefixedName("x", str(name))),
            provider=lambda: provider()[0],
        ),
        y=AuxiliaryVariable(
            name=str(PrefixedName("y", str(name))),
            provider=lambda: provider()[1],
        ),
        z=AuxiliaryVariable(
            name=str(PrefixedName("z", str(name))),
            provider=lambda: provider()[2],
        ),
    )


def create_vector3(name: PrefixedName, provider: Callable[[], List[float]]):
    return Vector3(
        x=AuxiliaryVariable(
            name=str(PrefixedName("x", str(name))),
            provider=lambda: provider()[0],
        ),
        y=AuxiliaryVariable(
            name=str(PrefixedName("y", str(name))),
            provider=lambda: provider()[1],
        ),
        z=AuxiliaryVariable(
            name=str(PrefixedName("z", str(name))),
            provider=lambda: provider()[2],
        ),
    )


@dataclass
class AuxiliaryVariableManager:
    variables: List[AuxiliaryVariable] = field(default_factory=list)
    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def add_variable(self, variable: AuxiliaryVariable):
        self.variables.append(variable)
        self.data = np.append(self.data, 0.0)

    def create_float_variable(
        self, name: PrefixedName, provider: Callable[[], float] = None
    ) -> AuxiliaryVariable:
        v = AuxiliaryVariable(name=str(name), provider=provider)
        self.add_variable(v)
        return v

    def create_point3(
        self, name: PrefixedName, provider: Callable[[], List[float]] = None
    ) -> Point3:
        x = AuxiliaryVariable(
            name=str(PrefixedName("x", str(name))), provider=lambda: provider()[0]
        )
        y = AuxiliaryVariable(
            name=str(PrefixedName("y", str(name))), provider=lambda: provider()[1]
        )
        z = AuxiliaryVariable(
            name=str(PrefixedName("z", str(name))), provider=lambda: provider()[2]
        )
        self.add_variable(x)
        self.add_variable(y)
        self.add_variable(z)
        return Point3(x, y, z)

    def create_transformation_matrix(
        self, name: PrefixedName, provider: Callable[[], np.ndarray] = None
    ) -> HomogeneousTransformationMatrix:
        transformation_matrix = HomogeneousTransformationMatrix()
        for row in range(3):
            for column in range(4):
                auxiliary_variable = AuxiliaryVariable(
                    name=str(PrefixedName(f"t[{row},{column}]", str(name))),
                    provider=lambda r=row, c=column: provider()[r, c],
                )
                self.add_variable(auxiliary_variable)
                transformation_matrix[row, column] = auxiliary_variable
        return transformation_matrix

    def resolve_auxiliary_variables(self) -> np.ndarray:
        for i, v in enumerate(self.variables):
            self.data[i] = v.resolve()
        return self.data
