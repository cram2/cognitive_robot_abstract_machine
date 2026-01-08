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

from krrood.adapters.exceptions import JSONSerializationError
from krrood.utils import DataclassException
from .datastructures.prefixed_name import PrefixedName

if TYPE_CHECKING:
    from .world import World
    from .world_description.world_entity import (
        SemanticAnnotation,
        WorldEntity,
        KinematicStructureEntity,
    )
    from .spatial_types.spatial_types import FloatVariable, SymbolicMathType


@dataclass
class UnknownWorldModification(DataclassException):
    """
    Raised when an unknown world modification is attempted.
    """

    call: Callable
    kwargs: Dict[str, Any]

    def __post_init__(self):
        self.message = (
            " Make sure that world modifications are atomic and that every atomic modification is "
            "represented by exactly one subclass of WorldModelModification."
            "This module might be incomplete, you can help by expanding it."
        )


@dataclass
class LogicalError(DataclassException):
    """
    An error that happens due to mistake in the logical operation or usage of the API during runtime.
    """


@dataclass
class DofNotInWorldStateError(DataclassException, KeyError):
    """
    An exception raised when a degree of freedom is not found in the world's state dictionary.
    """

    dof_id: UUID

    def __post_init__(self):
        self.message = f"Degree of freedom {self.dof_id} not found in world state."


@dataclass
class IncorrectWorldStateValueShapeError(DataclassException, ValueError):
    """
    An exception raised when the shape of a value in the world's state dictionary is incorrect.
    """

    dof_id: UUID

    def __post_init__(self):
        self.message = (
            f"Value for '{self.dof_id}' must be length-4 array (pos, vel, acc, jerk)."
        )


@dataclass
class MismatchingCommandLengthError(DataclassException, ValueError):
    """
    An exception raised when the length of a command does not match the expected length.
    """

    expected_length: int
    actual_length: int

    def __post_init__(self):
        self.message = f"Commands length {self.actual_length} does not match number of free variables {self.expected_length}."


@dataclass
class UsageError(LogicalError):
    """
    An exception raised when an incorrect usage of the API is encountered.
    """


@dataclass
class AddingAnExistingSemanticAnnotationError(UsageError):
    semantic_annotation: SemanticAnnotation

    def __post_init__(self):
        self.message = f"Semantic annotation {self.semantic_annotation} already exists."


@dataclass
class MissingWorldModificationContextError(UsageError):
    function: Callable

    def __post_init__(self):
        self.message = f"World function '{self.function.__name__}' was called without a 'with world.modify_world():' context manager."


@dataclass
class DuplicateWorldEntityError(UsageError):
    world_entities: List[WorldEntity]

    def __post_init__(self):
        self.message = f"WorldEntities {self.world_entities} are duplicates, while world entity elements should be unique."


@dataclass
class DuplicateKinematicStructureEntityError(UsageError):
    names: List[PrefixedName]

    def __post_init__(self):
        self.message = f"Kinematic structure entities with names {self.names} are duplicates, while kinematic structure entity names should be unique."


@dataclass
class SpatialTypesError(UsageError):
    pass


@dataclass
class ReferenceFrameMismatchError(SpatialTypesError):
    frame1: KinematicStructureEntity
    frame2: KinematicStructureEntity

    def __post_init__(self):
        self.message = f"Reference frames {self.frame1.name} and {self.frame2.name} are not the same."


@dataclass
class ParsingError(DataclassException, Exception):
    """
    An error that happens during parsing of files.
    """

    file_path: Optional[str] = None

    def __post_init__(self):
        self.message = f"Error parsing file {self.file_path}."


@dataclass
class WorldEntityNotFoundError(UsageError):
    name_or_hash: Union[PrefixedName, int]

    def __post_init__(self):
        if isinstance(self.name_or_hash, PrefixedName):
            self.message = f"WorldEntity with name {self.name_or_hash} not found"
        else:
            self.message = f"WorldEntity with hash {self.name_or_hash} not found"


@dataclass
class AlreadyBelongsToAWorldError(UsageError):
    world: World
    type_trying_to_add: Type[WorldEntity]

    def __post_init__(self):
        self.message = f"Cannot add a {self.type_trying_to_add} that already belongs to another world {self.world.name}."


class NotJsonSerializable(JSONSerializationError): ...


@dataclass
class SpatialTypeNotJsonSerializable(NotJsonSerializable):
    spatial_object: SymbolicMathType

    def __post_init__(self):
        self.message = (
            f"Object of type '{self.spatial_object.__class__.__name__}' is not JSON serializable, because it has "
            f"free variables: {self.spatial_object.free_variables()}"
        )


@dataclass
class KinematicStructureEntityNotInKwargs(JSONSerializationError):
    kinematic_structure_entity_id: UUID

    def __post_init__(self):
        self.message = (
            f"Kinematic structure entity '{self.kinematic_structure_entity_id}' is not in the kwargs of the "
            f"method that created it."
        )


class AmbiguousNameError(ValueError):
    """Raised when more than one semantic annotation class matches a given name with the same score."""


class UnresolvedNameError(ValueError):
    """Raised when no semantic annotation class matches a given name."""
