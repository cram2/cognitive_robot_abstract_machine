from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from uuid import UUID

from typing_extensions import (
    Optional,
    List,
    Type,
    TYPE_CHECKING,
    Callable,
    Tuple,
    Union,
    Any,
)

from krrood.adapters.json_serializer import JSONSerializationError
from .datastructures.prefixed_name import PrefixedName

if TYPE_CHECKING:
    from .world import World
    from .world_description.world_entity import (
        SemanticAnnotation,
        WorldEntity,
        KinematicStructureEntity,
    )
    from .spatial_types.spatial_types import FloatVariable, SymbolicType


@dataclass
class UnknownWorldModification(Exception):
    """
    Raised when an unknown world modification is attempted.
    """

    call: Callable
    kwargs: Dict[str, Any]

    def __post_init__(self):
        super().__init__(
            " Make sure that world modifications are atomic and that every atomic modification is "
            "represented by exactly one subclass of WorldModelModification."
            "This module might be incomplete, you can help by expanding it."
        )


class LogicalError(Exception):
    """
    An error that happens due to mistake in the logical operation or usage of the API during runtime.
    """


class DofNotInWorldStateError(KeyError):
    """
    An exception raised when a degree of freedom is not found in the world's state dictionary.
    """

    dof_id: UUID

    def __init__(self, dof_id: UUID):
        self.dof_id = dof_id
        super().__init__(f"Degree of freedom {dof_id} not found in world state.")


class IncorrectWorldStateValueShapeError(ValueError):
    """
    An exception raised when the shape of a value in the world's state dictionary is incorrect.
    """

    dof_id: UUID

    def __init__(self, dof_id: UUID):
        self.dof_id = dof_id
        super().__init__(
            f"Value for '{dof_id}' must be length-4 array (pos, vel, acc, jerk)."
        )


class MismatchingCommandLengthError(ValueError):
    """
    An exception raised when the length of a command does not match the expected length.
    """

    expected_length: int
    actual_length: int

    def __init__(self, expected_length: int, actual_length: int):
        self.expected_length = expected_length
        self.actual_length = actual_length
        super().__init__(
            f"Commands length {self.actual_length} does not match number of free variables {self.expected_length}."
        )


class UsageError(LogicalError):
    """
    An exception raised when an incorrect usage of the API is encountered.
    """


@dataclass
class AddingAnExistingSemanticAnnotationError(UsageError):
    semantic_annotation: SemanticAnnotation

    def __post_init__(self):
        msg = f"Semantic annotation {self.semantic_annotation} already exists."
        super().__init__(msg)


@dataclass
class MissingWorldModificationContextError(UsageError):
    function: Callable

    def __post_init__(self):
        msg = f"World function '{self.function.__name__}' was called without a 'with world.modify_world():' context manager."
        super().__init__(msg)


@dataclass
class DuplicateWorldEntityError(UsageError):
    world_entities: List[WorldEntity]

    def __post_init__(self):
        msg = f"WorldEntities {self.world_entities} are duplicates, while world entity elements should be unique."
        super().__init__(msg)


@dataclass
class DuplicateKinematicStructureEntityError(UsageError):
    names: List[PrefixedName]

    def __post_init__(self):
        msg = f"Kinematic structure entities with names {self.names} are duplicates, while kinematic structure entity names should be unique."
        super().__init__(msg)


class SpatialTypesError(UsageError):
    pass


@dataclass
class ReferenceFrameMismatchError(SpatialTypesError):
    frame1: KinematicStructureEntity
    frame2: KinematicStructureEntity

    def __post_init__(self):
        msg = f"Reference frames {self.frame1.name} and {self.frame2.name} are not the same."
        super().__init__(msg)


@dataclass
class WrongDimensionsError(SpatialTypesError):
    expected_dimensions: Union[Tuple[int, int], str]
    actual_dimensions: Tuple[int, int]

    def __post_init__(self):
        msg = f"Expected {self.expected_dimensions} dimensions, but got {self.actual_dimensions}."
        super().__init__(msg)


@dataclass
class NotSquareMatrixError(SpatialTypesError):
    actual_dimensions: Tuple[int, int]

    def __post_init__(self):
        msg = f"Expected a square matrix, but got {self.actual_dimensions} dimensions."
        super().__init__(msg)


@dataclass
class HasFreeVariablesError(SpatialTypesError):
    """
    Raised when an operation can't be performed on an expression with free variables.
    """

    variables: List[FloatVariable]

    def __post_init__(self):
        msg = f"Operation can't be performed on expression with free variables: {self.variables}."
        super().__init__(msg)


class ExpressionEvaluationError(SpatialTypesError): ...


@dataclass
class WrongNumberOfArgsError(ExpressionEvaluationError):
    expected_number_of_args: int
    actual_number_of_args: int

    def __post_init__(self):
        msg = f"Expected {self.expected_number_of_args} arguments, but got {self.actual_number_of_args}."
        super().__init__(msg)


@dataclass
class DuplicateVariablesError(SpatialTypesError):
    """
    Raised when duplicate variables are found in an operation that requires unique variables.
    """

    variables: List[FloatVariable]

    def __post_init__(self):
        msg = f"Operation failed due to duplicate variables: {self.variables}. All variables must be unique."
        super().__init__(msg)


@dataclass
class ParsingError(Exception):
    """
    An error that happens during parsing of files.
    """

    file_path: Optional[str] = None
    msg: Optional[str] = None

    def __post_init__(self):
        if not self.msg:
            if self.file_path:
                self.msg = f"File {self.file_path} could not be parsed."
            else:
                self.msg = ""
        super().__init__(self.msg)


@dataclass
class WorldEntityNotFoundError(UsageError):
    name_or_hash: Union[PrefixedName, int]

    def __post_init__(self):
        if isinstance(self.name_or_hash, PrefixedName):
            msg = f"WorldEntity with name {self.name_or_hash} not found"
        else:
            msg = f"WorldEntity with hash {self.name_or_hash} not found"
        super().__init__(msg)


@dataclass
class AlreadyBelongsToAWorldError(UsageError):
    world: World
    type_trying_to_add: Type[WorldEntity]

    def __post_init__(self):
        msg = f"Cannot add a {self.type_trying_to_add} that already belongs to another world {self.world.name}."
        super().__init__(msg)


class NotJsonSerializable(JSONSerializationError): ...


@dataclass
class SpatialTypeNotJsonSerializable(NotJsonSerializable):
    spatial_object: SymbolicType

    def __post_init__(self):
        super().__init__(
            f"Object of type '{self.spatial_object.__class__.__name__}' is not JSON serializable, because it has "
            f"free variables: {self.spatial_object.free_variables()}"
        )


@dataclass
class KinematicStructureEntityNotInKwargs(JSONSerializationError):
    kinematic_structure_entity_id: UUID

    def __post_init__(self):
        super().__init__(
            f"Kinematic structure entity '{self.kinematic_structure_entity_id}' is not in the kwargs of the "
            f"method that created it."
        )


class AmbiguousNameError(ValueError):
    """Raised when more than one semantic annotation class matches a given name with the same score."""


class UnresolvedNameError(ValueError):
    """Raised when no semantic annotation class matches a given name."""
