from __future__ import annotations, absolute_import

from dataclasses import dataclass, field
from typing import Dict, Set
from uuid import UUID

from typing_extensions import (
    Optional,
    List,
    Type,
    TYPE_CHECKING,
    Callable,
    Union,
    Any,
)

from krrood.adapters.exceptions import JSONSerializationError
from krrood.symbolic_math.symbolic_math import SymbolicMathType
from krrood.exceptions import DataclassException
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import (
        AbstractRobot,
        AbstractRobotPart,
    )
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.geometry import Scale
    from semantic_digital_twin.world_description.world_entity import (
        SemanticAnnotation,
        WorldEntity,
        KinematicStructureEntity,
    )
    from semantic_digital_twin.spatial_types.spatial_types import (
        SpatialType,
    )
    from semantic_digital_twin.spatial_types import Vector3, Point3
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedomLimits,
        DegreeOfFreedom,
    )
    from semantic_digital_twin.world_description.world_modification import (
        WorldModification,
    )
    from semantic_digital_twin.collision_checking.collision_matrix import CollisionCheck


@dataclass
class NoJointStateWithType(DataclassException):
    """
    Raised when a JointState type is search which is not defined
    """

    joint_state: JointStateType

    def __post_init__(self):
        self.message = f"There is no JointState with the type: {self.joint_state}"


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
class MismatchingIDsInWorldModification(DataclassException):
    """
    Raised when the UUIDs of a world modification during application are not consistent with the UUIDs assigned during initialization.
    """

    modification_type: Type[WorldModification]

    original_uuids: list[UUID]
    """
    The original UUIDs of the Modification.
    """

    actual_uuids: list[UUID]
    """
    The actual UUIDs of the Modification.
    """

    def __post_init__(self):
        self.message = (
            f"The world modification of type {self.modification_type.__name__} was initialized the following UUIDs: {self.original_uuids}"
            f"But during the application of those modifications, the UUIDs were {self.actual_uuids}."
            f"Somehow the original UUIDs were overridden, which should not happen."
        )


@dataclass
class LogicalError(DataclassException):
    """
    An error that happens due to mistake in the logical operation or usage of the API during runtime.
    """


@dataclass
class NegativeConnectionVelocity(DataclassException):
    """
    An error that happens when a negative velocity limit is provided for a connection.
    """

    connection_name: Union[str, PrefixedName]
    """
    The name of the connection for which the velocity limit is negative.
    """

    velocity: float
    """
    The negative velocity limit.
    """

    def __post_init__(self):
        self.message = f"Velocity limit must be non-negative, got {self.velocity} for joint {self.connection_name}"


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
class WrongWorldModelVersion(LogicalError):
    """
    Raised when a specific world model version is required.
    """

    expected_version: int
    actual_version: int

    def __post_init__(self):
        self.message = f"Expected world model version {self.expected_version}, but got {self.actual_version}."


@dataclass
class NonMonotonicTimeError(LogicalError):
    """
    Raised when attempting to append a world state with a time that is not strictly greater than the last time.
    """

    last_time: float
    attempted_time: float

    def __post_init__(self):
        self.message = f"Time must be strictly increasing. Last time: {self.last_time}, attempted time: {self.attempted_time}"


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
class WorldValidationError(LogicalError):
    """
    Raised when the world fails validation, e.g., when the kinematic structure is not a tree.
    """

    world: World
    """
    The world that failed validation.
    """


@dataclass
class WorldIsNotATreeError(WorldValidationError):
    """
    Raised when the kinematic structure of the world is not a tree during validation.
    """

    def __post_init__(self):
        self.message = (
            f"The world is not a tree: found {len(self.world.kinematic_structure_entities)} kinematic "
            f"structure entities but {len(self.world.connections)} connections."
        )
        super().__post_init__()


@dataclass
class BrokenWorldModificationHistoryError(WorldValidationError):
    """
    Raised when the world's modification history was detected to be broken.
    """

    potential_cause: Optional[DataclassException] = None
    """
    The exception that was thrown and caused the world's modification history to be broken.
    """

    def __post_init__(self):
        self.message = (
            f"The world's modification history was detected to be broken. "
            f"The world is {self.world}"
            f" Potential cause: {self.potential_cause}"
        )
        super().__post_init__()


@dataclass
class WorldContainsOrphanedDegreeOfFreedom(WorldValidationError):
    """
    Raised when the kinematic structure of the world contains orphaned degrees of freedom during validation.
    """

    actual_dofs: Set[DegreeOfFreedom]
    """
    The actual degrees of freedom used in connections.
    """

    def __post_init__(self):
        self.message = (
            "self.degrees_of_freedom does not match the actual dofs used in connections. The orphaned degrees of freedom are: "
            f"{set(self.world.degrees_of_freedom) - self.actual_dofs}"
            " Did you forget to call self.delete_orphaned_dofs()?"
        )
        super().__post_init__()


@dataclass
class InvalidConnectionLimits(UsageError):
    """
    Raised when the lower limit is not less than the upper limit for a degree of freedom.
    """

    name: PrefixedName
    """
    The name of the degree of freedom.
    """

    limits: DegreeOfFreedomLimits
    """
    The invalid limits.
    """

    def __post_init__(self):
        self.message = f"Lower limit for {self.name} must be less than upper limit. Given limits: {self.limits}."


@dataclass
class MismatchingWorld(UsageError):
    """
    Raised when two entities belong to different worlds.
    """

    expected_world: World
    """
    The expected world.
    """

    given_world: World
    """
    The given world.
    """

    def __post_init__(self):
        self.message = f"The two entities have mismatching worlds. Expected world: {self.expected_world}, given world: {self.given_world}"


@dataclass
class SemanticAnnotationCircularDependencyError(UsageError):
    """
    Raised when a circular dependency between semantic annotations is detected.
    """

    semantic_annotations: List[SemanticAnnotation]
    """
    The list of semantic annotations that in which a circular dependency is detected.
    """

    def __post_init__(self):
        self.message = f"The following semantic annotations have circular dependencies: {self.semantic_annotations}"


@dataclass
class MissingSemanticAnnotationError(UsageError):
    """
    Raised when a semantic annotation is required but missing.
    """

    semantic_annotation_class: Type[SemanticAnnotation]
    """
    The semantic annotation class that requires another semantic annotation.
    """

    missing_semantic_annotation_class: Type[SemanticAnnotation]
    """
    The missing semantic annotation class.
    """

    def __post_init__(self):
        self.message = (
            f"The semantic annotation of type {self.missing_semantic_annotation_class.__name__} is required"
            f" by {self.semantic_annotation_class.__name__}, but is missing."
        )


@dataclass
class InvalidPlaneDimensions(UsageError):
    """
    Raised when the depth of a plane is not less than its width or height.
    """

    scale: Scale
    """
    The scale of the plane.
    """

    clazz: Type
    """
    The class for which the dimensions are invalid.
    """

    def __post_init__(self):
        self.message = f"The Dimensions {self.scale} are invalid for the class {self.clazz.__name__}"


@dataclass
class UselessConceptError(UsageError):
    """
    Used to indicate that the operation the user is trying to perform is not useful in the current context, even
    though it might be technically possible.
    """


@dataclass
class InvalidHingeActiveAxis(UsageError):
    """
    Raised when an invalid axis is provided.
    """

    axis: Vector3
    """
    The invalid axis.
    """

    def __post_init__(self):
        self.message = (
            f"Axis {self.axis} provided when trying to calculate the hinge position is invalid. "
            f"If you think this is incorrect, consider extending Door.calculate_world_T_hinge_based_on_handle"
        )


@dataclass
class AddingAnExistingSemanticAnnotationError(UsageError):
    semantic_annotation: SemanticAnnotation

    def __post_init__(self):
        self.message = f"Semantic annotation {self.semantic_annotation} already exists."


@dataclass
class SemanticAnnotationNotInWorldError(UsageError):
    semantic_annotation: SemanticAnnotation

    def __post_init__(self):
        self.message = f"Semantic annotation {self.semantic_annotation} does not belong to a world."


@dataclass
class MissingWorldModificationContextError(UsageError):
    function: Callable

    def __post_init__(self):
        self.message = f"World function '{self.function.__name__}' was called without a 'with world.modify_world():' context manager."


@dataclass
class MismatchingPublishChangesAttribute(UsageError):
    """
    Raised when trying to enter a world modification context with a different publish_changes policy than the currently active world modification context.
    """

    active_publish_changes: bool
    """
    The publish_changes of the currently active world modification context.
    """
    proposed_publish_changes: bool
    """
    The publish_changes of the world modification context that is being entered.
    """

    def __post_init__(self):
        self.message = f"Cannot enter context with publish_changes={self.proposed_publish_changes} when the currently active modification context has publish_changes={self.active_publish_changes}. Make sure to not nest contexts with different publish_changes states."


@dataclass
class MissingPublishChangesKWARG(UsageError):
    kwargs: Dict[str, Any]

    def __post_init__(self):
        self.message = f"publish_changes must be provided as a keyword argument, but got {self.kwargs}. If you see this exception you probably notified a synchronizer without setting publish_changes, which will cause hard to debug issues."


@dataclass
class StateUpdateContainsUnknownDegreesOfFreedomError(UsageError):
    """
    Raised when a WorldStateUpdate is received that contains one or more DOF identifiers
    absent from the world state index.  This indicates a severe model/state desynchronization
    that must be investigated rather than silently ignored.
    """

    unknown_identifiers: List[UUID]
    """
    List of unknown DOF UUIDs that were attempted to update the state of
    """

    def __post_init__(self):
        self.message = (
            f"Received a WorldStateUpdate containing {len(self.unknown_identifiers)} "
            f"DOF identifier(s) absent from the world state index: "
            f"{self.unknown_identifiers}. "
            "This means the world model and state are severely out of sync."
        )


@dataclass
class ApplyMissedMessagesWhileWorldIsBeingModifiedError(UsageError):
    """
    Raised when apply_missed_messages is called while a modify_world context is active on the synchronizer's world.
    Applying missed messages requires entering a modify_world context internally, which would conflict
    with any currently active modify_world context due to mismatching publish_changes policies.
    """

    def __post_init__(self):
        self.message = (
            "apply_missed_messages must not be called while a modify_world context is active on the synchronizer's world. "
            "Call apply_missed_messages after the modify_world context has exited."
        )


@dataclass
class DuplicateWorldEntityError(UsageError):
    world_entities: List[WorldEntity]

    def __post_init__(self):
        self.message = f"WorldEntities {self.world_entities} are duplicates, while world entity elements should be unique."


@dataclass
class DuplicateRobotAssignmentsError(UsageError):
    """
    Raised when a robot part is assigned to multiple robots, which should not happen.
    """

    robot_part: AbstractRobotPart
    """
    The robot part that is assigned to multiple robots.
    """

    robots: list[AbstractRobot]
    """
    The robots that are already assigned to the robot part.
    """

    def __post_init__(self):
        self.message = (
            f"Robot part {self.robot_part} is assigned to multiple robots: {self.robots}."
            f" Each robot part should be assigned to at most one robot."
        )


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
class MissingReferenceFrameError(SpatialTypesError):
    """
    Represents an error that occurs when a spatial type lacks a reference frame, even though its required for the
    current operation
    """

    spatial_type: SpatialType
    """
    Spatial type that lacks a reference frame.
    """

    def __post_init__(self):
        self.message = f"Spatial type {self.spatial_type} has no reference frame."


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
        super().__post_init__()


@dataclass
class MissingDefaultCameraError(UsageError):
    """
    Raised when trying to access the default camera of a robot that does not have a default camera.
    """

    robot: Type[AbstractRobot]
    """
    The robot that does not have a default camera.
    """

    def __post_init__(self):
        self.message = f"Robot {self.robot.name} does not have a default camera."


@dataclass
class MissingWorldError(UsageError):
    """
    Raised when trying to access a world that is None, but a world is required for the operation.
    """

    def __post_init__(self):
        self.message = f"The world you are trying to access is None."


@dataclass
class WorldEntityWithIDNotFoundError(WorldEntityNotFoundError):
    name_or_hash: UUID = None

    def __post_init__(self):
        self.message = f"WorldEntity with id {self.name_or_hash} not found"

    @property
    def id(self) -> UUID:
        return self.name_or_hash


@dataclass
class AlreadyBelongsToAWorldError(UsageError):
    world: World
    type_trying_to_add: Type[WorldEntity]

    def __post_init__(self):
        self.message = f"Cannot add a {self.type_trying_to_add} that already belongs to another world {self.world.name}."


@dataclass
class DoesNotBelongToAWorldError(UsageError):
    """
    Raised when trying to use a world entity that does not belong to any world in a context where it must belong to a world.
    """

    world_entity: WorldEntity
    """
    The world entity that does not belong to a world.
    """

    def __post_init__(self):
        self.message = f"WorldEntity {self.world_entity} does not belong to a world."


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
class WorldEntityWithIDNotInKwargs(JSONSerializationError):
    world_entity_id: UUID

    def __post_init__(self):
        self.message = (
            f"World entity '{self.world_entity_id}' is not in the kwargs of the "
            f"method that created it."
        )


class AmbiguousNameError(ValueError):
    """Raised when more than one semantic annotation class matches a given name with the same score."""


class UnresolvedNameError(ValueError):
    """Raised when no semantic annotation class matches a given name."""


@dataclass
class RootNodeNotFoundError(DataclassException):
    """
    Raised when the root node cannot be found or is ambiguous in a scene graph.
    """

    candidates: List[str]
    """The candidate node names that were considered as potential roots."""

    def __post_init__(self):
        self.message = (
            f"Could not determine unique root node. Candidates: {self.candidates}"
        )


@dataclass
class CollisionCheckingError(DataclassException):
    message: str = field(kw_only=True, default=None, init=False)


@dataclass
class InvalidCollisionCheckError(CollisionCheckingError):
    collision_check: CollisionCheck


@dataclass
class NegativeCollisionCheckingDistanceError(InvalidCollisionCheckError):
    def __post_init__(self):
        super().__post_init__()
        self.message = f"Distance must be positive, got {self.collision_check.distance}"


@dataclass
class InvalidBodiesInCollisionCheckError(InvalidCollisionCheckError):
    def __post_init__(self):
        super().__post_init__()
        self.message = f"Body_a and body_b must be different, got {self.collision_check.body_a} and {self.collision_check.body_b}"


@dataclass
class BodyHasNoGeometryError(InvalidCollisionCheckError):
    def __post_init__(self):
        super().__post_init__()
        self.message = ""
        if not self.collision_check.body_a.has_collision():
            self.message += (
                f"Body {self.collision_check.body_a.name} has collision geometry."
            )
        if not self.collision_check.body_b.has_collision():
            self.message += (
                f"Body {self.collision_check.body_b.name} has collision geometry."
            )


@dataclass
class AtomicWorldModificationNotAtomic(DataclassException):
    """
    Exception raised when atomic world modifications are overlapping.
    If this exception is raised, it means that somewhere in the code a function decorated with @atomic_world_modification
    triggered another function decorated with it. This must not happen ever!
    """

    modification: Callable
    """
    The callable that tried to atomically modify the world.
    """

    world: World
    """
    The world where this happened.
    """

    def __post_init__(self):
        self.message = (
            f"World {self.world} is already being modified atomically by "
            f"{self.world._current_active_atomic_world_modification.__name__}.\n"
            f"{self.modification.__name__} tried to perform an atomic world modification anyways."
        )
        super().__post_init__()


@dataclass
class PointOccupiedError(DataclassException):
    """
    Error that is raised when a pose is occupied or not in the search space of a Connectivity Graphs.
    """

    point: Point3
    """
    The point that is occupied.
    """

    def __post_init__(self):
        self.message = f"The point {self.point} is occupied."
