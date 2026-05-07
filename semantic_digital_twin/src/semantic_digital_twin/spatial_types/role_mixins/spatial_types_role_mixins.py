from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Self, Tuple, Union
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from casadi.casadi import SX
    from collections.abc import Iterable
    from krrood.adapters.json_serializer import JSONAttributeDiff
    from krrood.symbolic_math.symbolic_math import (
        CompiledFunction,
        FloatVariable,
        GenericSymbolicType,
        Matrix,
        Scalar,
        VariableParameters,
        Vector,
    )
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
        Point3,
        Pose,
        Quaternion,
        RotationMatrix,
        SpatialType,
        np,
    )
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )
    from types import ScalarData


@dataclass(eq=False)
class RoleForSpatialType(ABC):
    @property
    @abstractmethod
    def role_taker(self) -> SpatialType: ...
    @property
    def casadi_sx(self) -> SX:
        return self.role_taker.casadi_sx

    @casadi_sx.setter
    def casadi_sx(self, value: SX):
        self.role_taker.casadi_sx = value

    @property
    def reference_frame(self) -> Union[KinematicStructureEntity, None]:
        return self.role_taker.reference_frame

    @reference_frame.setter
    def reference_frame(self, value: Union[KinematicStructureEntity, None]):
        self.role_taker.reference_frame = value

    def __deepcopy__(self, memo) -> Self:
        return self.role_taker.__deepcopy__(memo)

    @staticmethod
    def _ensure_consistent_frame(
        spatial_objects: List[Optional[SpatialType]],
    ) -> Optional[KinematicStructureEntity]:
        from semantic_digital_twin.spatial_types.spatial_types import Pose

        return Pose._ensure_consistent_frame(spatial_objects)


@dataclass(eq=False, init=False, repr=False)
class RoleForPose(RoleForSpatialType, ABC):
    @property
    @abstractmethod
    def role_taker(self) -> Pose: ...
    @property
    def casadi_sx(self) -> SX:
        return self.role_taker.casadi_sx

    @casadi_sx.setter
    def casadi_sx(self, value: SX):
        self.role_taker.casadi_sx = value

    @property
    def shape(self) -> tuple[int, int]:
        return self.role_taker.shape

    @property
    def x(self) -> Scalar:
        return self.role_taker.x

    @x.setter
    def x(self, value: Scalar):
        self.role_taker.x = value

    @property
    def y(self) -> Scalar:
        return self.role_taker.y

    @y.setter
    def y(self, value: Scalar):
        self.role_taker.y = value

    @property
    def z(self) -> Scalar:
        return self.role_taker.z

    @z.setter
    def z(self, value: Scalar):
        self.role_taker.z = value

    def __abs__(self) -> Self:
        return self.role_taker.__abs__()

    def __array__(self):
        return self.role_taker.__array__()

    def __copy__(self) -> Self:
        return self.role_taker.__copy__()

    def __getitem__(
        self, item: np.ndarray | int | slice | Tuple[int | slice, int | slice]
    ) -> Scalar | Vector:
        return self.role_taker.__getitem__(item)

    def __hash__(self):
        return self.role_taker.__hash__()

    def __len__(self) -> int:
        return self.role_taker.__len__()

    def __neg__(self) -> Self:
        return self.role_taker.__neg__()

    def __setitem__(
        self,
        key: int | slice | Tuple[int | slice, int | slice],
        value: ScalarData,
    ):
        return self.role_taker.__setitem__(key, value)

    def __str__(self):
        return self.role_taker.__str__()

    def _apply_diff(self, diff: JSONAttributeDiff, **kwargs) -> None:
        return self.role_taker._apply_diff(diff, kwargs)

    def _verify_type(self):
        return self.role_taker._verify_type()

    def compile(
        self,
        parameters: Optional[VariableParameters] = None,
        sparse: bool = False,
    ) -> CompiledFunction:
        return self.role_taker.compile(parameters, sparse)

    def equivalent(self, other: ScalarData) -> bool:
        return self.role_taker.equivalent(other)

    def evaluate(self) -> np.ndarray:
        return self.role_taker.evaluate()

    def flatten(self) -> Vector:
        return self.role_taker.flatten()

    def free_variables(self) -> List[FloatVariable]:
        return self.role_taker.free_variables()

    def is_constant(self) -> bool:
        return self.role_taker.is_constant()

    def is_scalar(self) -> bool:
        return self.role_taker.is_scalar()

    def jacobian(self, variables: Iterable[FloatVariable]) -> Matrix:
        return self.role_taker.jacobian(variables)

    def jacobian_ddot(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
        variables_ddot: Iterable[FloatVariable],
    ) -> Matrix:
        return self.role_taker.jacobian_ddot(variables, variables_dot, variables_ddot)

    def jacobian_dot(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
    ) -> Matrix:
        return self.role_taker.jacobian_dot(variables, variables_dot)

    def pretty_str(self) -> List[List[str]]:
        return self.role_taker.pretty_str()

    def safe_division(
        self,
        other: GenericSymbolicType,
        if_nan: Optional[ScalarData] = None,
    ) -> GenericSymbolicType:
        return self.role_taker.safe_division(other, if_nan)

    def second_order_total_derivative(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
        variables_ddot: Iterable[FloatVariable],
    ) -> Vector:
        return self.role_taker.second_order_total_derivative(
            variables, variables_dot, variables_ddot
        )

    def substitute(
        self,
        old_variables: List[FloatVariable],
        new_variables: List[ScalarData] | Vector,
    ) -> Self:
        return self.role_taker.substitute(old_variables, new_variables)

    def to_homogeneous_matrix(self) -> HomogeneousTransformationMatrix:
        return self.role_taker.to_homogeneous_matrix()

    def to_json(self) -> Dict[str, Any]:
        return self.role_taker.to_json()

    def to_list(self) -> list:
        return self.role_taker.to_list()

    def to_np(self) -> np.ndarray:
        return self.role_taker.to_np()

    def to_position(self) -> Point3:
        return self.role_taker.to_position()

    def to_quaternion(self) -> Quaternion:
        return self.role_taker.to_quaternion()

    def to_rotation_matrix(self) -> RotationMatrix:
        return self.role_taker.to_rotation_matrix()

    def total_derivative(
        self,
        variables: Iterable[FloatVariable],
        variables_dot: Iterable[FloatVariable],
    ) -> Vector:
        return self.role_taker.total_derivative(variables, variables_dot)

    def update_from_json_diff(self, diffs: List[JSONAttributeDiff], **kwargs) -> None:
        return self.role_taker.update_from_json_diff(diffs, kwargs)
