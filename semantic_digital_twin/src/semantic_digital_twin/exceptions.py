from __future__ import annotations, absolute_import

from dataclasses import dataclass, field, Field
from typing import Dict, Set, Any
from uuid import UUID

import mujoco
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
    from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
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
    Raised when a JointState type is search which is not defined.
    """

    joint_state: JointStateType

    def error_message(self) -> str:
        return f"There is no JointState with the type: {self.joint_state}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnknownWorldModification(DataclassException):
    """
    Raised when an unknown world modification is attempted.
    """

    call: Callable
    kwargs: Dict[str, Any]

    def error_message(self) -> str:
        call_name = getattr(self.call, "__name__", repr(self.call))
        return f"No WorldModification subclass is registered for the call '{call_name}' with kwargs {self.kwargs}."

    def suggest_correction(self) -> str:
        return (
            "make sure that world modifications are atomic and that every atomic modification is "
            "represented by exactly one subclass of WorldModelModification. "
            "This module might be incomplete, you can help by expanding it."
        )


@dataclass
class MismatchingIDsInWorldModification(DataclassException):
    """
    Raised when the UUIDs of a world modification during application are not consistent
    with the UUIDs assigned during initialization.
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

    def error_message(self) -> str:
        return (
            f"The world modification of type {self.modification_type.__name__} was initialized with the following UUIDs: {self.original_uuids}. "
            f"But during the application of those modifications, the UUIDs were {self.actual_uuids}. "
            f"Somehow the original UUIDs were overridden, which should not happen unless a user deliberately does so."
        )

    def suggest_correction(self) -> str:
        return (
            "if you are sure you did not override the UUIDs yourself, this may be an internal error in "
            "semantic_digital_twin. In that case, please open a bug report with a script reproducing the error."
        )


@dataclass
class LogicalError(DataclassException):
    """
    An error that happens due to mistake in the logical operation or usage of the API
    during runtime.
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

    def error_message(self) -> str:
        return f"Velocity limit must be non-negative, got {self.velocity} for joint {self.connection_name}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class DofNotInWorldStateError(DataclassException, KeyError):
    """
    An exception raised when a degree of freedom is not found in the world's state
    dictionary.
    """

    dof_id: UUID

    def error_message(self) -> str:
        return f"Degree of freedom {self.dof_id} not found in world state."

    def suggest_correction(self) -> str:
        return (
            "did you possibly try to merge states of two different world models? "
            "States of two different worlds can only be merged if other._world.degrees_of_freedom "
            "is a subset of self._world.degrees_of_freedom."
        )


@dataclass
class IncorrectWorldStateValueShapeError(DataclassException, ValueError):
    """
    An exception raised when the shape of a value in the world's state dictionary is
    incorrect.
    """

    dof_id: UUID

    def error_message(self) -> str:
        return (
            f"Value for '{self.dof_id}' must be length-4 array (pos, vel, acc, jerk)."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class WrongWorldModelVersion(LogicalError):
    """
    Raised when a specific world model version is required.
    """

    expected_version: int
    actual_version: int

    def error_message(self) -> str:
        return (
            f"Expected world model version {self.expected_version}, but got {self.actual_version}. "
            f"The model version increments with every structural change (adding/removing bodies, "
            f"connections or degrees of freedom), so the world model changed since this object was created."
        )

    def suggest_correction(self) -> str:
        return "recreate whatever was derived from the old world model."


@dataclass
class NonMonotonicTimeError(LogicalError):
    """
    Raised when attempting to append a world state with a time that is not strictly
    greater than the last time.
    """

    last_time: float
    attempted_time: float

    def error_message(self) -> str:
        return f"Time must be strictly increasing. Last time: {self.last_time}, attempted time: {self.attempted_time}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MismatchingCommandLengthError(DataclassException, ValueError):
    """
    An exception raised when the length of a command does not match the expected length.
    """

    expected_length: int
    actual_length: int

    def error_message(self) -> str:
        return f"Commands length {self.actual_length} does not match number of free variables {self.expected_length}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UsageError(LogicalError):
    """
    An exception raised when an incorrect usage of the API is encountered.
    """


@dataclass
class WorldValidationError(LogicalError):
    """
    Raised when the world fails validation, e.g., when the kinematic structure is not a
    tree.
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

    def error_message(self) -> str:
        return (
            f"The world is not a tree: found {len(self.world.kinematic_structure_entities)} kinematic "
            f"structure entities but {len(self.world.connections)} connections."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class BrokenWorldModificationHistoryError(WorldValidationError):
    """
    Raised when the world's modification history was detected to be broken.
    """

    potential_cause: Optional[DataclassException] = None
    """
    The exception that was thrown and caused the world's modification history to be
    broken.
    """

    def error_message(self) -> str:
        return (
            f"The world's modification history was detected to be broken. "
            f"The world is {self.world}"
            f" Potential cause: {self.potential_cause}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class WorldContainsOrphanedDegreeOfFreedom(WorldValidationError):
    """
    Raised when the kinematic structure of the world contains orphaned degrees of
    freedom during validation.
    """

    actual_dofs: Set[DegreeOfFreedom]
    """
    The actual degrees of freedom used in connections.
    """

    def error_message(self) -> str:
        return (
            "self.degrees_of_freedom does not match the actual dofs used in connections. The orphaned degrees of freedom are: "
            f"{set(self.world.degrees_of_freedom) - self.actual_dofs}"
        )

    def suggest_correction(self) -> str:
        return "did you forget to call self.delete_orphaned_dofs()?"


@dataclass
class InvalidConnectionLimits(UsageError):
    """
    Raised when the lower limit is not less than the upper limit for a degree of
    freedom.
    """

    name: PrefixedName
    """
    The name of the degree of freedom.
    """

    limits: DegreeOfFreedomLimits
    """
    The invalid limits.
    """

    def error_message(self) -> str:
        return f"Lower limit for {self.name} must be less than upper limit. Given limits: {self.limits}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MimicDofLimitOverwriteError(UsageError):
    """
    Raised when trying to overwrite the limits of a mimic degree of freedom.
    """

    dof_name: PrefixedName
    """
    The name of the mimic degree of freedom.
    """

    def error_message(self) -> str:
        return (
            f"Cannot overwrite limits of the mimic DOF '{self.dof_name}'. "
            f"Mimic DOF limits are derived from the DOF they mimic."
        )

    def suggest_correction(self) -> str:
        return "use .raw_dof._overwrite_dof_limits instead."


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

    def error_message(self) -> str:
        return (
            f"The two entities have mismatching worlds: expected world '{self.expected_world.name}', "
            f"given world '{self.given_world.name}'. Entities can only reference each other when they belong to "
            f"the same world."
        )

    def suggest_correction(self) -> str:
        return (
            "bring everything into one world first, e.g. with "
            "world.merge_world(other_world) or world.merge_world_at_pose(other_world, pose)."
        )


@dataclass
class CannotBeAPartOf(UsageError):
    """
    Raised when ``add`` is called with a part that no part-whole relationship field of
    the annotation accepts.
    """

    annotation: SemanticAnnotation
    """
    The annotation the part was being added to.
    """

    part: SemanticAnnotation
    """
    The part that could not be added.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.part).__name__} cannot be added as a part of "
            f"{type(self.annotation).__name__}: no part-whole relationship field accepts it."
        )

    def suggest_correction(self) -> str:
        return (
            f"Check out the superclasses of {type(self.annotation).__name__}, to find the currently valid part-whole "
            f"relationships, and if you think its currently incomplete, feel free to adjust it and make a PR for it"
        )


@dataclass
class AmbiguousPart(UsageError):
    """
    Raised when ``add`` is called with a part whose type matches more than one part-
    whole relationship field of the annotation, so the target field is ambiguous.
    """

    annotation: SemanticAnnotation
    """
    The annotation the part was being added to.
    """

    part: SemanticAnnotation
    """
    The part that could not be unambiguously added.
    """

    fields: List[Field]
    """
    The names of the part-whole relationship fields whose element type accepts the part.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.part).__name__} cannot be unambiguously added as a part of "
            f"{type(self.annotation).__name__}: it matches multiple part-whole relationship fields "
            f"({', '.join([field_.name for field_ in self.fields])})."
        )

    def suggest_correction(self) -> str:
        return (
            f"consider if its practical to use the 'field_name' keyword argument for the `add` method to"
            f" disambiguate the matching cases."
        )


@dataclass
class UnknownPartWholeRelationshipField(UsageError):
    """
    Raised when ``add`` is called with a ``field_name`` that is not a part-whole
    relationship field of the annotation.
    """

    annotation: HasRootBody
    """
    The annotation the part was being added to.
    """

    field_name: str
    """
    The field name that was requested but does not exist as a part-whole relationship
    field.
    """

    available_fields: List[str]
    """
    The names of the annotation's part-whole relationship fields.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.annotation).__name__} has no part-whole relationship field "
            f"'{self.field_name}."
        )

    def suggest_correction(self) -> str:
        return (
            f"the available fields are:"
            f" {', '.join(self.available_fields) or '(none)'}"
        )


@dataclass
class MechanicalJointAlreadyMounted(UsageError):
    """
    Raised when a mechanical joint that already connects a child is mounted onto a
    different whole.

    If you think a single Mechanical Joint should be able to have multiple children,
    contact @LucaKro.
    """

    joint: SemanticAnnotation
    """
    The mechanical joint being mounted.
    """

    main_has_root_body_annotation: SemanticAnnotation
    """
    The annotation (the whole) the joint was being mounted onto.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.joint).__name__} already connects a child and cannot be mounted onto "
            f"{type(self.main_has_root_body_annotation).__name__}: a mechanical joint connects exactly one child."
        )

    def suggest_correction(self) -> str:
        return f"if you think that you found a case where this error does not apply, please contact @LucaKro"


@dataclass
class SemanticAnnotationCircularDependencyError(UsageError):
    """
    Raised when a circular dependency between semantic annotations is detected.
    """

    semantic_annotations: List[SemanticAnnotation]
    """
    The list of semantic annotations that in which a circular dependency is detected.
    """

    def error_message(self) -> str:
        return f"The following semantic annotations have circular dependencies: {self.semantic_annotations}"

    def suggest_correction(self) -> str:
        return ""


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

    def error_message(self) -> str:
        return (
            f"The semantic annotation of type {self.missing_semantic_annotation_class.__name__} is required"
            f" by {self.semantic_annotation_class.__name__}, but is missing."
        )

    def suggest_correction(self) -> str:
        return ""


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

    def error_message(self) -> str:
        return f"The Dimensions {self.scale} are invalid for the class {self.clazz.__name__}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UselessConceptError(UsageError):
    """
    Used to indicate that the operation the user is trying to perform is not useful in
    the current context, even though it might be technically possible.
    """

    reason: str
    """
    Why the operation is not useful in this context.
    """

    def error_message(self) -> str:
        return self.reason

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidHingeActiveAxis(UsageError):
    """
    Raised when an invalid axis is provided.
    """

    axis: Vector3
    """
    The invalid axis.
    """

    def error_message(self) -> str:
        return f"Axis {self.axis} provided when trying to calculate the hinge position is invalid."

    def suggest_correction(self) -> str:
        return "if you think this is incorrect, consider extending Door.calculate_world_T_hinge_based_on_handle."


@dataclass
class SemanticAnnotationNotInWorldError(UsageError):
    semantic_annotation: SemanticAnnotation

    def error_message(self) -> str:
        return (
            f"Semantic annotation {self.semantic_annotation} does not belong to a world, "
            f"but this operation requires one."
        )

    def suggest_correction(self) -> str:
        return "add it first via world.add_semantic_annotation(annotation)."


@dataclass
class MissingWorldModificationContextError(UsageError):
    function: Callable

    def _public_function_name(self) -> str:
        # Strip leading underscores so the hint shows the public method, not the internal one it delegates to.
        return getattr(self.function, "__name__", repr(self.function)).lstrip("_")

    def error_message(self) -> str:
        return (
            f"'{self._public_function_name()}' modifies the world model and must be called inside "
            f"a world modification context."
        )

    def suggest_correction(self) -> str:
        function_name = self._public_function_name()
        return (
            f"wrap the call like this:\n"
            f"    with world.modify_world():\n"
            f"        world.{function_name}(...)"
        )


@dataclass
class MismatchingPublishChangesAttribute(UsageError):
    """
    Raised when trying to enter a world modification context with a different
    publish_changes policy than the currently active world modification context.
    """

    active_publish_changes: bool
    """
    The publish_changes of the currently active world modification context.
    """

    proposed_publish_changes: bool
    """
    The publish_changes of the world modification context that is being entered.
    """

    def error_message(self) -> str:
        return (
            f"Cannot enter context with publish_changes={self.proposed_publish_changes} when the currently active "
            f"modification context has publish_changes={self.active_publish_changes}."
        )

    def suggest_correction(self) -> str:
        return "make sure to not nest contexts with different publish_changes states."


@dataclass
class MissingPublishChangesKWARG(UsageError):
    kwargs: Dict[str, Any]

    def error_message(self) -> str:
        return (
            f"publish_changes must be provided as a keyword argument, but got {self.kwargs}. "
            f"If you see this exception you probably notified a synchronizer without setting publish_changes, "
            f"which will cause hard to debug issues."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class StateUpdateContainsUnknownDegreesOfFreedomError(UsageError):
    """
    Raised when a WorldStateUpdate is received that contains one or more DOF identifiers
    absent from the world state index.

    This indicates a severe model/state desynchronization that must be investigated
    rather than silently ignored.
    """

    unknown_identifiers: List[UUID]
    """
    List of unknown DOF UUIDs that were attempted to update the state of.
    """

    def error_message(self) -> str:
        return (
            f"Received a WorldStateUpdate containing {len(self.unknown_identifiers)} "
            f"DOF identifier(s) absent from the world state index: "
            f"{self.unknown_identifiers}. "
            "This means the world model and state are severely out of sync."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ApplyMissedMessagesWhileWorldIsBeingModifiedError(UsageError):
    """
    Raised when apply_missed_messages is called while a modify_world context is active
    on the synchronizer's world.

    Applying missed messages requires entering a modify_world context internally, which
    would conflict with any currently active modify_world context due to mismatching
    publish_changes policies.
    """

    def error_message(self) -> str:
        return "apply_missed_messages must not be called while a modify_world context is active on the synchronizer's world."

    def suggest_correction(self) -> str:
        return "call apply_missed_messages after the modify_world context has exited."


@dataclass
class DuplicateWorldEntityError(UsageError):
    world_entities: List[WorldEntity]

    def error_message(self) -> str:
        names = [str(world_entity.name) for world_entity in self.world_entities]
        return f"Multiple world entities match: {names}, but the result must be unique."

    def suggest_correction(self) -> str:
        return (
            "if this came from a lookup with a plain string name, disambiguate by passing a "
            "PrefixedName with the desired prefix, or use the plural get_..._by_name variant "
            "to retrieve all matches."
        )


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

    def error_message(self) -> str:
        return (
            f"Robot part {self.robot_part} is assigned to multiple robots: {self.robots}."
            f" Each robot part should be assigned to at most one robot."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class SpatialTypesError(UsageError):
    pass


@dataclass
class InsufficientVectorsError(SpatialTypesError):
    """
    Raised when a rotation matrix is constructed from fewer than two vectors.
    """

    x: Optional[Vector3]
    y: Optional[Vector3]
    z: Optional[Vector3]

    def error_message(self) -> str:
        return (
            f"from_vectors requires at least two of the vectors x, y, z to be provided; "
            f"the third is computed via the cross product."
            f"X Vector was: {self.x}"
            f"Y Vector was: {self.y}"
            f"Z Vector was: {self.z}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ReferenceFrameMismatchError(SpatialTypesError):
    expected_frame: KinematicStructureEntity
    """
    The frame the operation requires.
    """

    actual_frame: KinematicStructureEntity
    """
    The frame that was found instead.
    """

    context: Optional[str] = None
    """
    Description of the value whose frame was checked, e.g. the argument name.
    """

    def error_message(self) -> str:
        checked_value = self.context or "the given value"
        return (
            f"Expected {checked_value} to be expressed in frame '{self.expected_frame.name}', "
            f"but it is expressed in frame '{self.actual_frame.name}'."
        )

    def suggest_correction(self) -> str:
        return (
            "transform it into the expected frame first, e.g. via "
            "world.transform(value, target_frame), or construct it with the expected frame."
        )


@dataclass
class MissingReferenceFrameError(SpatialTypesError):
    """
    Represents an error that occurs when a spatial type lacks a reference frame, even
    though its required for the current operation.
    """

    spatial_type: SpatialType
    """
    Spatial type that lacks a reference frame.
    """

    def error_message(self) -> str:
        return f"Spatial type {self.spatial_type} has no reference frame, but this operation requires one."

    def suggest_correction(self) -> str:
        return "construct it with an explicit frame, e.g. Point3(x, y, z, reference_frame=some_body)."


@dataclass
class ParsingError(DataclassException, Exception):
    """
    An error that happens during parsing of files.
    """

    file_path: Optional[str] = None

    def error_message(self) -> str:
        return f"Error parsing file {self.file_path}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class PackageResolutionError(ParsingError):
    """
    Raised when a ROS package name cannot be resolved to a directory.
    """

    package_name: str = field(kw_only=True)
    """
    The package name that could not be resolved.
    """

    details: Optional[str] = field(kw_only=True, default=None)
    """
    Details about why the resolution failed.
    """

    def error_message(self) -> str:
        message = f"Could not resolve package '{self.package_name}'."
        if self.details:
            message += f" Details: {self.details}"
        return message

    def suggest_correction(self) -> str:
        return ""


@dataclass
class PathResolutionError(ParsingError):
    """
    Raised when a URI cannot be resolved to a local file path.
    """

    uri: str = field(kw_only=True)
    """
    The URI that could not be resolved.
    """

    details: Optional[str] = field(kw_only=True, default=None)
    """
    Details about why the resolution failed.
    """

    def error_message(self) -> str:
        message = f"Could not resolve path '{self.uri}'."
        if self.details:
            message += f" Details: {self.details}"
        return message

    def suggest_correction(self) -> str:
        return ""


@dataclass
class WorldEntityNotFoundError(UsageError):
    name_or_hash: Union[str, PrefixedName, int]

    suggestions: List[Union[str, PrefixedName]] = field(default_factory=list)
    """
    Names of existing world entities that closely match the searched name.
    """

    def error_message(self) -> str:
        if isinstance(self.name_or_hash, (str, PrefixedName)):
            return f"No world entity with name '{self.name_or_hash}' found."
        return f"No world entity with hash {self.name_or_hash} found."

    def suggest_correction(self) -> str:
        if not self.suggestions:
            return ""
        formatted_suggestions = ", ".join(
            f"'{suggestion}'" for suggestion in self.suggestions
        )
        return f"did you mean: {formatted_suggestions}?"


@dataclass
class MissingDefaultCameraError(UsageError):
    """
    Raised when trying to access the default camera of a robot that does not have a
    default camera.
    """

    robot: Type[AbstractRobot]
    """
    The robot that does not have a default camera.
    """

    def error_message(self) -> str:
        return f"Robot {self.robot.name} does not have a default camera."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MissingWorldError(UsageError):
    """
    Raised when trying to access a world that is None, but a world is required for the
    operation.
    """

    def error_message(self) -> str:
        return "A world is required for this operation, but None was found."

    def suggest_correction(self) -> str:
        return "make sure the involved entities have been added to a world before calling this."


@dataclass
class WorldEntityWithIDNotFoundError(WorldEntityNotFoundError):
    name_or_hash: UUID = None

    def error_message(self) -> str:
        return f"WorldEntity with id {self.name_or_hash} not found"

    def suggest_correction(self) -> str:
        return ""

    @property
    def id(self) -> UUID:
        return self.name_or_hash


@dataclass
class AlreadyBelongsToAWorldError(UsageError):
    world: World
    type_trying_to_add: Type[WorldEntity]

    def error_message(self) -> str:
        return (
            f"Cannot add this {self.type_trying_to_add.__name__} because it already belongs to the world "
            f"'{self.world.name}'. A world entity can belong to at most one world."
        )

    def suggest_correction(self) -> str:
        return "to combine two worlds, use world.merge_world(other_world); to reuse an entity, create a fresh copy."


@dataclass
class DoesNotBelongToAWorldError(UsageError):
    """
    Raised when trying to use a world entity that does not belong to any world in a
    context where it must belong to a world.
    """

    world_entity: WorldEntity
    """
    The world entity that does not belong to a world.
    """

    def error_message(self) -> str:
        return f"WorldEntity '{self.world_entity.name}' does not belong to a world, but this operation requires one."

    def suggest_correction(self) -> str:
        return (
            "add it to a world first, e.g.:\n"
            "    with world.modify_world():\n"
            "        world.add_kinematic_structure_entity(entity)"
        )


class NotJsonSerializable(JSONSerializationError): ...


@dataclass
class SpatialTypeNotJsonSerializable(NotJsonSerializable):
    spatial_object: SymbolicMathType

    def error_message(self) -> str:
        return (
            f"Object of type '{self.spatial_object.__class__.__name__}' is not JSON serializable, because it has "
            f"free variables: {self.spatial_object.free_variables()}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class WorldEntityWithIDNotInKwargs(JSONSerializationError):
    world_entity_id: UUID

    def error_message(self) -> str:
        return (
            f"World entity '{self.world_entity_id}' is not in the kwargs of the "
            f"method that created it."
        )

    def suggest_correction(self) -> str:
        return ""


class AmbiguousNameError(ValueError):
    """
    Raised when more than one semantic annotation class matches a given name with the
    same score.
    """


class UnresolvedNameError(ValueError):
    """
    Raised when no semantic annotation class matches a given name.
    """


@dataclass
class RootNodeNotFoundError(DataclassException):
    """
    Raised when the root node cannot be found or is ambiguous in a scene graph.
    """

    candidates: List[str]
    """
    The candidate node names that were considered as potential roots.
    """

    def error_message(self) -> str:
        return f"Could not determine unique root node. Candidates: {self.candidates}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class CollisionCheckingError(DataclassException):
    """
    Base class for errors during collision checking.
    """


@dataclass
class InvalidCollisionCheckError(CollisionCheckingError):
    collision_check: CollisionCheck


@dataclass
class NegativeCollisionCheckingDistanceError(InvalidCollisionCheckError):
    def error_message(self) -> str:
        return f"Distance must be positive, got {self.collision_check.distance}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidBodiesInCollisionCheckError(InvalidCollisionCheckError):
    def error_message(self) -> str:
        return f"Body_a and body_b must be different, got {self.collision_check.body_a} and {self.collision_check.body_b}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class BodyHasNoGeometryError(InvalidCollisionCheckError):
    def error_message(self) -> str:
        bodies_without_geometry = [
            body
            for body in (self.collision_check.body_a, self.collision_check.body_b)
            if not body.has_collision()
        ]
        return " ".join(
            f"Body {body.name} has no collision geometry."
            for body in bodies_without_geometry
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class AtomicWorldModificationNotAtomic(DataclassException):
    """
    Exception raised when atomic world modifications are overlapping.

    If this exception is raised, it means that somewhere in the code a function
    decorated with @atomic_world_modification triggered another function decorated with
    it. This must not happen ever!
    """

    modification: Callable
    """
    The callable that tried to atomically modify the world.
    """

    world: World
    """
    The world where this happened.
    """

    def error_message(self) -> str:
        return (
            f"World '{self.world.name}' is already being modified atomically by "
            f"'{self.world._current_active_atomic_world_modification.__name__}' while "
            f"'{self.modification.__name__}' attempted another atomic world modification. "
            f"Atomic world modifications must never trigger each other."
        )

    def suggest_correction(self) -> str:
        return (
            "this is an internal error in semantic_digital_twin, not a usage error - "
            "please open a bug report with the stack trace."
        )


@dataclass
class PointOccupiedError(DataclassException):
    """
    Error that is raised when a pose is occupied or not in the search space of a
    Connectivity Graphs.
    """

    point: Point3
    """
    The point that is occupied.
    """

    def error_message(self) -> str:
        return f"The point {self.point} is occupied."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MultiSimError(DataclassException):
    """
    Base class for all MultiSim-related exceptions.
    """


@dataclass
class QuaternionConversionError(MultiSimError):
    """
    Raised when a rotation matrix cannot be converted to a quaternion.
    """

    rotation_matrix: Any
    """
    The rotation matrix that could not be converted.
    """

    reason: str
    """
    The error message of the underlying conversion failure.
    """

    def error_message(self) -> str:
        return (
            f"Error converting rotation matrix to quaternion. "
            f"Rotation matrix:\n{self.rotation_matrix}\nError message: {self.reason}"
        )

    def suggest_correction(self) -> str:
        return ""


class MujocoError(MultiSimError):
    """
    Base class for all MuJoCo-related exceptions.
    """


class MujocoEntityNotFoundError(MujocoError):
    """
    Raised when a MuJoCo entity of a given type and name cannot be found.
    """

    entity_name: str
    entity_type: mujoco.mjtObj
    action: str = "find"

    def error_message(self) -> str:
        return f"Failed to {self.action}: type={self.entity_type}, name='{self.entity_name}'"

    def suggest_correction(self) -> str:
        return ""
