from __future__ import absolute_import
from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps, lru_cache, cached_property
from itertools import combinations_with_replacement
from uuid import UUID

import numpy as np
import rustworkx as rx
import rustworkx.visualization
from lxml import etree
from rustworkx import NoEdgeBetweenNodes
from typing_extensions import (
    Dict,
    Tuple,
    Optional,
    TypeVar,
    Union,
    Callable,
    Any,
    Iterable,
    TYPE_CHECKING,
)
from typing_extensions import List
from typing_extensions import Type, Set

from .callbacks.callback import ModelChangeCallback
from .collision_checking.collision_detector import CollisionDetector
from .collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from .datastructures.prefixed_name import PrefixedName
from .datastructures.types import NpMatrix4x4
from .exceptions import (
    DuplicateWorldEntityError,
    WorldEntityNotFoundError,
    AlreadyBelongsToAWorldError,
    MissingWorldModificationContextError,
    WorldEntityWithIDNotFoundError,
)
from .robots.abstract_robot import AbstractRobot
from .spatial_computations.forward_kinematics import ForwardKinematicsManager
from .spatial_computations.ik_solver import InverseKinematicsSolver
from .spatial_computations.raytracer import RayTracer
from .spatial_types import HomogeneousTransformationMatrix, Quaternion
from .spatial_types.derivatives import Derivatives
from .utils import IDGenerator
from .world_description.connections import (
    Connection6DoF,
    ActiveConnection1DOF,
    FixedConnection,
    ActiveConnection,
)
from .world_description.connections import HasUpdateState
from .world_description.degree_of_freedom import DegreeOfFreedom
from .world_description.visitors import CollisionBodyCollector, ConnectionCollector
from .world_description.world_entity import (
    Connection,
    SemanticAnnotation,
    WorldEntityWithID,
    KinematicStructureEntity,
    Region,
    GenericKinematicStructureEntity,
    GenericConnection,
    CollisionCheckingConfig,
    Body,
    WorldEntity,
    GenericWorldEntity,
    Actuator,
)
from .world_description.world_modification import (
    WorldModelModification,
    WorldModelModificationBlock,
    SetDofHasHardwareInterface,
    AddDegreeOfFreedomModification,
    RemoveDegreeOfFreedomModification,
    AddKinematicStructureEntityModification,
    AddConnectionModification,
    RemoveConnectionModification,
    RemoveBodyModification,
    AddSemanticAnnotationModification,
    RemoveSemanticAnnotationModification,
    AddActuatorModification,
    RemoveActuatorModification,
)
from .world_description.world_state import WorldState

if TYPE_CHECKING:
    from .spatial_types import GenericSpatialType

logger = logging.getLogger(__name__)

id_generator = IDGenerator()

GenericSemanticAnnotation = TypeVar(
    "GenericSemanticAnnotation", bound=SemanticAnnotation
)

FunctionStack = List[Tuple[Callable, Dict[str, Any]]]


class PlotAlignment(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


class ResetStateContextManager:
    """
    A context manager for resetting the state of a given `World` instance.

    This class is designed to allow operations to be performed on a `World`
    object, ensuring that its state can be safely returned to its previous
    condition upon leaving the context. If no exceptions occur within the
    context, the original state of the `World` instance is restored, and the
    state change is notified.
    """

    def __init__(self, world: World):
        self.world = world

    def __enter__(self) -> None:
        self.state = deepcopy(self.world.state)

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[type],
    ) -> None:
        if exc_type is None:
            self.world.state = self.state
            self.world.notify_state_change()


@dataclass
class WorldModelUpdateContextManager:
    """
    Context manager for updating the state of a given `World` instance.
    This class manages that updates to the world within the context of this class only trigger recomputations after all
    desired updates have been performed.
    """

    world: World = field(kw_only=True, repr=False)
    """
    The world to manage updates for.
    """

    first: bool = True
    """
    First time flag.
    """

    def __enter__(self):
        if self.world.world_is_being_modified:
            self.first = False
        self.world.world_is_being_modified = True

        if self.first:
            self.world.get_world_model_manager().current_model_modification_block = (
                WorldModelModificationBlock()
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.first:
            self.world.delete_orphaned_dofs()
            self.world.get_world_model_manager().model_modification_blocks.append(
                self.world.get_world_model_manager().current_model_modification_block
            )
            self.world.get_world_model_manager().current_model_modification_block = None
            if exc_type is None:
                self.world._notify_model_change()
            self.world.world_is_being_modified = False


class AtomicWorldModificationNotAtomic(Exception):
    """
    Exception raised when atomic world modifications are overlapping.
    If this exception is raised, it means that somewhere in the code a function decorated with @atomic_world_modification
    triggered another function decorated with it. This must not happen ever!
    """


def atomic_world_modification(
    func=None, modification: Type[WorldModelModification] = None
):
    """
    Decorator for ensuring atomicity in world modification operations.

    This decorator ensures that no other atomic world modification is in progress when the decorated function is executed.
    It records the function call along with its arguments for potential replay or tracking purposes.
    If an operation is attempted when the world is locked, it raises an appropriate exception.

    Raises:
        AtomicWorldModificationNotAtomic: If the world is already locked during the execution of another atomic operation.
    """

    def _decorate(func):

        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(current_world: World, *args, **kwargs):
            if current_world._atomic_modification_is_being_executed:
                raise AtomicWorldModificationNotAtomic(
                    f"World {current_world} is locked."
                )
            current_world._atomic_modification_is_being_executed = True

            # bind args and kwargs
            bound = sig.bind_partial(
                current_world, *args, **kwargs
            )  # use bind() if you require all args
            bound.apply_defaults()  # fill in default values

            # record function call
            # Build a dict with all arguments (including positional), excluding 'self'
            bound_args = dict(bound.arguments)
            bound_args.pop("self", None)
            if (
                current_world.get_world_model_manager().current_model_modification_block
                is None
            ):
                raise MissingWorldModificationContextError(func)
            current_world.get_world_model_manager().current_model_modification_block.append(
                modification.from_kwargs(bound_args)
            )

            result = func(current_world, *args, **kwargs)

            current_world._atomic_modification_is_being_executed = False
            return result

        return wrapper

    if func is None:
        return _decorate

    return _decorate(func)


@dataclass
class CollisionPairManager:
    """
    Manages disabled collision pairs in the world.
    """

    world: World
    """
    The world to manage collision pairs for.
    """

    _disabled_collision_pairs: Set[Tuple[Body, Body]] = field(
        default_factory=set, repr=False
    )
    """
    Collisions for these Body pairs is disabled.f
    """

    _temp_disabled_collision_pairs: Set[Tuple[Body, Body]] = field(
        default_factory=set, repr=False
    )
    """
    A set of Body pairs for which collisions are temporarily disabled.
    """

    def reset_temporary_collision_config(self):
        self._temp_disabled_collision_pairs = set()
        for body in self.world.bodies_with_enabled_collision:
            body.reset_temporary_collision_config()

    @property
    def disabled_collision_pairs(
        self,
    ) -> Set[Tuple[Body, Body]]:
        return self._disabled_collision_pairs | self._temp_disabled_collision_pairs

    @property
    def enabled_collision_pairs(self) -> Set[Tuple[Body, Body]]:
        """
        The complement of disabled_collision_pairs with respect to all possible body combinations with enabled collision.
        """
        all_combinations = set(
            combinations_with_replacement(self.world.bodies_with_enabled_collision, 2)
        )
        return all_combinations - self.disabled_collision_pairs

    def add_temp_disabled_collision_pair(
        self, body_a: KinematicStructureEntity, body_b: KinematicStructureEntity
    ):
        """
        Disable collision checking between two bodies
        """
        pair = tuple(sorted([body_a, body_b], key=lambda b: b.id))
        self._temp_disabled_collision_pairs.add(pair)

    def load_collision_srdf(self, file_path: str):
        """
        Creates a CollisionConfig instance from an SRDF file.

        Parse an SRDF file to configure disabled collision pairs or bodies for a given world.
        Process SRDF elements like `disable_collisions`, `disable_self_collision`,
        or `disable_all_collisions` to update collision configuration
        by referencing bodies in the provided `world`.

        :param file_path: The path to the SRDF file used for collision configuration.
        """
        SRDF_DISABLE_ALL_COLLISIONS: str = "disable_all_collisions"
        SRDF_DISABLE_SELF_COLLISION: str = "disable_self_collision"
        SRDF_MOVEIT_DISABLE_COLLISIONS: str = "disable_collisions"

        srdf = etree.parse(file_path)
        srdf_root = srdf.getroot()

        children_with_tag = [child for child in srdf_root if hasattr(child, "tag")]

        child_disable_collisions = [
            c for c in children_with_tag if c.tag == SRDF_DISABLE_ALL_COLLISIONS
        ]

        for c in child_disable_collisions:
            body = self.world.get_body_by_name(c.attrib["link"])
            body.set_static_collision_config(CollisionCheckingConfig(disabled=True))

        child_disable_moveit_and_self_collision = [
            c
            for c in children_with_tag
            if c.tag in {SRDF_MOVEIT_DISABLE_COLLISIONS, SRDF_DISABLE_SELF_COLLISION}
        ]

        disabled_collision_pairs = [
            (body_a, body_b)
            for child in child_disable_moveit_and_self_collision
            if (
                body_a := self.world.get_body_by_name(child.attrib["link1"])
            ).has_collision()
            and (
                body_b := self.world.get_body_by_name(child.attrib["link2"])
            ).has_collision()
        ]

        for body_a, body_b in disabled_collision_pairs:
            self.add_disabled_collision_pair(body_a, body_b)

    def disable_collisions_for_adjacent_bodies(self):
        """
        Computes pairs of bodies that should not be collision checked because they have no controlled connections
        between them.

        When all connections between two bodies are not controlled, these bodies cannot move relative to each
        other, so collision checking between them is unnecessary.

        :return: Set of body pairs that should have collisions disabled
        """

        body_combinations = combinations_with_replacement(
            self.world.bodies_with_enabled_collision, 2
        )

        for body_a, body_b in (
            (a, b)
            for a, b in body_combinations
            if not self.world.is_controlled_connection_in_chain(a, b)
        ):
            self.add_disabled_collision_pair(body_a, body_b)

    def disable_non_robot_collisions(self) -> None:
        """
        Disable collision checks between bodies that do not belong to any robot.
        """
        # Bodies that are part of any robot and participate in collisions
        robot_bodies: Set[Body] = {
            body
            for robot in self.world.get_semantic_annotations_by_type(AbstractRobot)
            for body in robot.bodies_with_collisions
        }

        # Bodies with collisions that are NOT part of a robot
        non_robot_bodies: Set[Body] = (
            set(self.world.bodies_with_enabled_collision) - robot_bodies
        )
        if not non_robot_bodies:
            return

        # Disable every unordered pair (including self-collisions) exactly once
        for a, b in combinations_with_replacement(non_robot_bodies, 2):
            self.add_disabled_collision_pair(a, b)

    def add_disabled_collision_pair(self, body_a: Body, body_b: Body):
        """
        Disable collision checking between two bodies
        """
        pair = tuple(sorted([body_a, body_b], key=lambda body: body.id))
        self._disabled_collision_pairs.add(pair)


@dataclass
class WorldModelManager:
    """
    Manages the world model version and modification blocks.
    """

    version: int = 0
    """
    The version of the model. This increases whenever a change to the kinematic model is made. Mostly triggered
    by adding/removing bodies and connections.
    """

    model_modification_blocks: List[WorldModelModificationBlock] = field(
        default_factory=list, repr=False, init=False
    )
    """
    All atomic modifications applied to the world. Tracked by @atomic_world_modification.
    The field itself is a list of lists. The outer lists indicates when to trigger the model/state change callbacks.
    The inner list is a block of modifications where change callbacks must not be called in between.
    """

    current_model_modification_block: Optional[WorldModelModificationBlock] = field(
        default=None, repr=False, init=False
    )
    """
    The current modification block called within one context of @atomic_world_modification.
    """

    model_change_callbacks: List[ModelChangeCallback] = field(
        default_factory=list, repr=False
    )
    """
    Callbacks to be called when the model of the world changes.
    """

    def update_model_version_and_notify_callbacks(self) -> None:
        """
        Notifies the system of a model change and updates necessary states, caches,
        and forward kinematics expressions while also triggering registered callbacks
        for model changes.
        """
        self.version += 1
        for callback in self.model_change_callbacks:
            callback.notify()


_LRU_CACHE_SIZE: int = 2048


@dataclass
class World:
    """
    A class representing the world.
    The world manages a set of kinematic structure entities and connections represented as a tree-like graph.
    The nodes represent kinematic structure entities in the world, and the edges represent joins between them.
    """

    kinematic_structure: rx.PyDAG[KinematicStructureEntity] = field(
        default_factory=lambda: rx.PyDAG(multigraph=False), kw_only=True, repr=False
    )
    """
    The kinematic structure of the world.
    The kinematic structure is a tree shaped directed graph where the nodes represent kinematic structure entities
     in the world, and the edges represent connections between them.
    """

    semantic_annotations: List[SemanticAnnotation] = field(
        default_factory=list, repr=False
    )
    """
    All semantic annotations the world is aware of.
    """

    degrees_of_freedom: List[DegreeOfFreedom] = field(default_factory=list)
    """
    All degrees of freedom in the world.
    """

    actuators: List[Actuator] = field(default_factory=list)
    """
    All actuators in the world.
    """

    state: WorldState = field(init=False)
    """
    2d array where rows are derivatives and columns are dof values for that derivative.
    """

    world_is_being_modified: bool = False
    """
    Is set to True, when a world.modify_world context is used.
    """

    name: Optional[str] = None
    """
    Name of the world. May act as default namespace for all bodies and semantic annotations in the world which do not have a prefix.
    """

    _atomic_modification_is_being_executed: bool = field(init=False, default=False)
    """
    Flag that indicates if an atomic world operation is currently being executed.
    See `atomic_world_modification` for more information.
    """

    _collision_pair_manager: CollisionPairManager = field(init=False, repr=False)
    """
    Manages disabled collision pairs in the world.
    """

    _model_manager: WorldModelManager = field(
        default_factory=WorldModelManager, repr=False
    )
    """
    Manages the world model version and modification blocks.
    """

    _forward_kinematic_manager: ForwardKinematicsManager = field(
        init=False, default=None, repr=False
    )
    """
    Manages forward kinematics computations for the world.
    """

    _world_entity_hash_table: Dict = field(init=False, default_factory=dict)
    """
    Lookup table to get a world entity by its hash
    """

    def __post_init__(self):
        self._collision_pair_manager = CollisionPairManager(self)
        self.state = WorldState(_world=self)

    def __hash__(self):
        return hash((id(self), self._model_manager.version))

    def __str__(self):
        return f"{self.__class__.name} v{self._model_manager.version}.{self.state.version}."

    def validate(self) -> bool:
        """
        Validate the world.

        The world must be a tree.
        :return: True if the world is valid, raises an AssertionError otherwise.
        """
        if self.is_empty():
            return True
        assert len(self.kinematic_structure_entities) == (len(self.connections) + 1)
        assert rx.is_weakly_connected(self.kinematic_structure)
        self._validate_dofs()
        return True

    def _validate_dofs(self):
        actual_dofs = {
            dof for connection in self.connections for dof in connection.dofs
        }
        assert actual_dofs == set(
            self.degrees_of_freedom
        ), "self.degrees_of_freedom does not match the actual dofs used in connections. Did you forget to call self.delete_orphaned_dofs()?"

    # %% Properties
    @property
    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def root(self) -> Optional[KinematicStructureEntity]:
        """
        The root of the world is the unique node with in-degree 0.

        :return: The root of the world.
        """
        if self.is_empty():
            return None

        possible_roots = [
            node
            for node in self.kinematic_structure_entities
            if self.kinematic_structure.in_degree(node.index) == 0
        ]
        assert (
            len(possible_roots) == 1
        ), f"A World must have exactly one root. Found {len(possible_roots)} possible roots: {possible_roots}."

        return possible_roots[0]

    @property
    def active_degrees_of_freedom(self) -> Set[DegreeOfFreedom]:
        return {
            dof for connection in self.connections for dof in connection.active_dofs
        }

    @property
    def passive_degrees_of_freedom(self) -> Set[DegreeOfFreedom]:
        return {
            dof for connection in self.connections for dof in connection.passive_dofs
        }

    @property
    def regions(self) -> List[Region]:
        """
        :return: A list of all regions in the world.
        """
        return self.get_kinematic_structure_entity_by_type(Region)

    @property
    def bodies(self) -> List[Body]:
        """
        :return: A list of all bodies in the world.
        """
        return self.get_kinematic_structure_entity_by_type(Body)

    @property
    def bodies_with_enabled_collision(self) -> List[Body]:
        return [
            b
            for b in self.bodies
            if b.has_collision()
            and b.get_collision_config
            and not b.get_collision_config().disabled
        ]

    @property
    def bodies_topologically_sorted(self) -> List[Body]:
        return [
            body
            for body in self.kinematic_structure_entities_topologically_sorted
            if isinstance(body, Body)
        ]

    @property
    def kinematic_structure_entities(self) -> List[KinematicStructureEntity]:
        """
        :return: A list of all bodies in the world.
        """
        return list(self.kinematic_structure.nodes())

    @property
    def kinematic_structure_entities_topologically_sorted(
        self,
    ) -> List[KinematicStructureEntity]:
        """
        Return a list of all kinematic_structure_entities in the world, sorted topologically.
        """
        indices = rx.topological_sort(self.kinematic_structure)
        return [self.kinematic_structure[index] for index in indices]

    @property
    def connections(self) -> List[Connection]:
        """
        :return: A list of all connections in the world.
        """
        return list(self.kinematic_structure.edges())

    @property
    def controlled_connections(self) -> List[ActiveConnection]:
        """
        A subset of the robot's connections that are controlled by a controller.
        """
        return [
            connection for connection in self.connections if connection.is_controlled
        ]

    # %% Adding WorldEntities to the World
    def add_connection(self, connection: Connection) -> None:
        """
        Add a connection and the entities it connects to the world.

        :param connection: The connection to add.
        """
        logger.debug(
            f"Adding connection with name {connection.name} between parent {connection.parent.name} and child {connection.child.name}"
        )
        self._raise_error_if_belongs_to_other_world(connection)
        if not self.is_connection_in_world(connection):
            self.add_kinematic_structure_entity(connection.parent)
            self.add_kinematic_structure_entity(connection.child)
            self._add_connection(connection)

    @atomic_world_modification(modification=AddConnectionModification)
    def _add_connection(self, connection: Connection):
        """
        Adds a connection to the kinematic structure.

        The method updates the connection instance to associate it with the current
        world instance and reflects the connection in the kinematic structure.
        Do not call this function directly, use add_connection instead.

        :param connection: The connection to be added to the kinematic structure.
        """
        connection.add_to_world(self)
        self.kinematic_structure.add_edge(
            connection.parent.index, connection.child.index, connection
        )

    def add_body(
        self,
        body: KinematicStructureEntity,
    ):
        return self.add_kinematic_structure_entity(body)

    def add_kinematic_structure_entity(
        self,
        kinematic_structure_entity: KinematicStructureEntity,
    ):
        """
        Add a kinematic_structure_entity to the world if it does not exist already.

        :param kinematic_structure_entity: The kinematic_structure_entity to add.
        """
        logger.info(
            f"Trying to add kinematic_structure_entity with name {kinematic_structure_entity.name}"
        )
        self._raise_error_if_belongs_to_other_world(kinematic_structure_entity)
        if not self.is_kinematic_structure_entity_in_world(kinematic_structure_entity):
            self._add_kinematic_structure_entity(kinematic_structure_entity)

    @atomic_world_modification(modification=AddKinematicStructureEntityModification)
    def _add_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ):
        """
        Add a kinematic_structure_entity to the world.
        Do not call this function directly, use add_kinematic_structure_entity instead.

        :param kinematic_structure_entity: The kinematic_structure_entity to add.
        """
        kinematic_structure_entity.add_to_world(self)
        kinematic_structure_entity.index = self.kinematic_structure.add_node(
            kinematic_structure_entity
        )

    def add_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        """
        Adds degree of freedom in the world.
        This is used to register DoFs that are not created by the world, but are part of the world model.
        :param dof: The degree of freedom to register.
        """
        self._raise_error_if_belongs_to_other_world(dof)
        if not self.is_degree_of_freedom_in_world(dof):
            self._add_degree_of_freedom(dof)

    @atomic_world_modification(modification=AddDegreeOfFreedomModification)
    def _add_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        """
        Adds a degree of freedom to the current system and initializes its state.

        This method modifies the internal state of the system by adding a new
        degree of freedom (DOF). It sets the initial position of the DOF based
        on its configured lower and upper position limits, ensuring it respects
        both constraints. The DOF is then added to the list of degrees of freedom
        in the system.

        :param dof: The degree of freedom to be added to the system.
        :return: None
        """
        dof.add_to_world(self)
        self.state.add_degree_of_freedom(dof)
        self.degrees_of_freedom.append(dof)

    def add_semantic_annotation(self, semantic_annotation: SemanticAnnotation) -> None:
        """
        Adds a semantic annotation to the current list of semantic annotations if it doesn't already exist

        :param semantic_annotation: The semantic annotation instance to be added. Its name must be unique within
            the current context.

        :raises AddingAnExistingSemanticAnnotationError: If the semantic annotation already exists
        """
        logger.debug(f"Adding semantic annotation with name {semantic_annotation.name}")
        self._raise_error_if_belongs_to_other_world(semantic_annotation)
        if not self.is_semantic_annotation_in_world(semantic_annotation):
            self._add_semantic_annotation(semantic_annotation)

    def add_semantic_annotations(
        self,
        semantic_annotations: List[SemanticAnnotation],
    ) -> None:
        """
        Adds a list of semantic annotations to the current list of semantic annotations if they don't already exist.
        :param semantic_annotations: The list of semantic annotations to be added.
        :param skip_duplicates: Whether to raise an error or not when a semantic annotation already exists.
        """
        for semantic_annotation in semantic_annotations:
            self.add_semantic_annotation(
                semantic_annotation,
            )

    @atomic_world_modification(modification=AddSemanticAnnotationModification)
    def _add_semantic_annotation(self, semantic_annotation: SemanticAnnotation):
        """
        The atomic method that adds a semantic annotation to the current list of semantic annotations.
        """
        semantic_annotation.add_to_world(self)
        self.semantic_annotations.append(semantic_annotation)
        self._world_entity_hash_table[hash(semantic_annotation)] = semantic_annotation

    def add_actuator(self, actuator: Actuator) -> None:
        """
        Adds an actuator in the world.
        This is used to register Actuators that are not created by the world, but are part of the world model.

        :param actuator: The actuator to register.
        """
        if actuator._world is self and actuator in self.actuators:
            return
        if actuator._world is not None:
            raise AlreadyBelongsToAWorldError(
                world=actuator._world, type_trying_to_add=Actuator
            )
        self._add_actuator(actuator)

    @atomic_world_modification(modification=AddActuatorModification)
    def _add_actuator(self, actuator: Actuator) -> None:
        """
        Adds an actuator to the current system.

        This method modifies the internal state of the system by adding a new
        actuator. The actuator is then added to the list of actuators
        in the system.

        :param actuator: The actuator to be added to the system.
        :return: None
        """
        actuator.add_to_world(self)
        self.actuators.append(actuator)

    def _raise_error_if_belongs_to_other_world(self, world_entity: WorldEntity):
        """
        Raises an AlreadyBelongsToAWorldError if the world_entity already belongs to another world.
        :param world_entity:
        """
        if world_entity._world is not None and world_entity._world is not self:
            raise AlreadyBelongsToAWorldError(
                world=world_entity._world, type_trying_to_add=type(world_entity)
            )

    # %% Remove WorldEntities from the World
    def remove_connection(self, connection: Connection) -> None:
        """
        Removes a connection.
        Might create disconnected entities, so make sure to add a new connection or delete the child kinematic_structure_entity.

        :param connection: The connection to be removed

        .. warning::

            The reason self.is_connection_in_world is not checked before removing the connection, is because it is using
            the self.connections internally, which accesses the live rustworkx kinematic_structure. The problem arises
            if we want to remove the parent or child from the world, before removing the connection from the world.
            In that case, rustworkx automatically removes the edge representing the connection, which results in
            self.is_connection_in_world returning False, even though we have not cleaned up the connection properly on
            our side.
        """
        self._remove_connection(connection)

    @atomic_world_modification(modification=RemoveConnectionModification)
    def _remove_connection(self, connection: Connection) -> None:
        parent_index = connection.parent.index
        child_index = connection.child.index
        if parent_index is not None and child_index is not None:
            try:
                self.kinematic_structure.remove_edge(parent_index, child_index)
            except NoEdgeBetweenNodes:
                pass
        connection.remove_from_world()

    def remove_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> None:
        """
        Removes a kinematic_structure_entity from the world.

        :param kinematic_structure_entity: The kinematic_structure_entity to remove.
        """
        if self.is_kinematic_structure_entity_in_world(kinematic_structure_entity):
            self._remove_kinematic_structure_entity(kinematic_structure_entity)

    @atomic_world_modification(modification=RemoveBodyModification)
    def _remove_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> None:
        """
        Removes a kinematic_structure_entity from the world.

        Do not call this function directly, use `remove_kinematic_structure_entity` instead.

        :param kinematic_structure_entity: The kinematic_structure_entity to remove.
        """
        self.kinematic_structure.remove_node(kinematic_structure_entity.index)
        kinematic_structure_entity.remove_from_world()

    def remove_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        if self.is_degree_of_freedom_in_world(dof):
            self._remove_degree_of_freedom(dof)
            self.get_degree_of_freedom_by_name.cache_clear()

    @atomic_world_modification(modification=RemoveDegreeOfFreedomModification)
    def _remove_degree_of_freedom(self, dof: DegreeOfFreedom) -> None:
        dof.remove_from_world()
        self.degrees_of_freedom.remove(dof)
        del self.state[dof.id]

    def remove_semantic_annotation(
        self, semantic_annotation: SemanticAnnotation
    ) -> None:
        """
        Removes a semantic annotation from the current list of semantic annotations if it exists.

        :param semantic_annotation: The semantic annotation instance to be removed.
        """
        if self.is_semantic_annotation_in_world(semantic_annotation):
            self._remove_semantic_annotation(semantic_annotation)

    @atomic_world_modification(modification=RemoveSemanticAnnotationModification)
    def _remove_semantic_annotation(self, semantic_annotation: SemanticAnnotation):
        """
        The atomic method that removes a semantic annotation from the current list of semantic annotations.
        """
        semantic_annotation.remove_from_world()
        self.semantic_annotations.remove(semantic_annotation)

    def remove_actuator(self, actuator: Actuator) -> None:
        """
        Removes an actuator from the current list of actuators if it exists.

        :param actuator: The actuator instance to be removed.
        """
        if self.is_actuator_in_world(actuator):
            self._remove_actuator(actuator)

    @atomic_world_modification(modification=RemoveActuatorModification)
    def _remove_actuator(self, actuator: Actuator) -> None:
        """
        The atomic method that removes an actuator from the current list of actuators.
        """
        actuator.remove_from_world()
        self.actuators.remove(actuator)

    # %% Other Atomic World Modifications
    @atomic_world_modification(modification=SetDofHasHardwareInterface)
    def set_dofs_has_hardware_interface(
        self, dofs: Iterable[DegreeOfFreedom], value: bool
    ):
        """
        Sets whether the specified degrees of freedom (DOFs) have a hardware interface or not.

        This method allows controlling the presence of a hardware interface for multiple
        DOFs at once. The modification is atomic, ensuring that all DOFs are updated as
        a single operation and the state remains consistent. The method iterates through
        the given DOFs and updates their `has_hardware_interface` attribute to the provided
        value.

        :param dofs: An iterable collection of DegreeOfFreedom instances whose
                     `has_hardware_interface` attribute is to be updated.
        :param value: A boolean value indicating whether the DOFs should have a hardware
                      interface (True) or not (False).
        """
        for dof in dofs:
            dof.has_hardware_interface = value

    # %% Getter
    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_connection(
        self, parent: KinematicStructureEntity, child: KinematicStructureEntity
    ) -> Connection:
        """
        Retrieves the connection between a parent and child kinematic_structure_entity in the kinematic structure.
        """
        return self.kinematic_structure.get_edge_data(parent.index, child.index)

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_connections_by_type(
        self, connection_type: Type[GenericConnection]
    ) -> List[GenericConnection]:
        """
        Retrieves the connections of a given type.

        :param connection_type: The type of connection to retrieve.
        :return: A list of connections of the given type.
        """
        return self._get_world_entity_by_type_from_iterable(
            connection_type, self.connections
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_semantic_annotations_by_type(
        self, semantic_annotation_type: Type[GenericSemanticAnnotation]
    ) -> List[GenericSemanticAnnotation]:
        """
        Retrieves all semantic annotations of a specific type from the world.

        :param semantic_annotation_type: The class (type) of the semantic annotations to search for.
        :return: A list of `SemanticAnnotation` objects that match the given type.
        """
        return self._get_world_entity_by_type_from_iterable(
            semantic_annotation_type, self.semantic_annotations
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_kinematic_structure_entity_by_type(
        self, entity_type: Type[GenericKinematicStructureEntity]
    ) -> List[GenericKinematicStructureEntity]:
        """
        Retrieves all kinematic structure entities of a specific type from the world.

        :param entity_type: The class (type) of the kinematic structure entities to search for.
        :return: A list of `KinematicStructureEntity` objects that match the given type.
        """
        return self._get_world_entity_by_type_from_iterable(
            entity_type, self.kinematic_structure_entities
        )

    @staticmethod
    def _get_world_entity_by_type_from_iterable(
        world_entity_type: Type[GenericWorldEntity], iterable: Iterable[WorldEntity]
    ) -> List[GenericWorldEntity]:
        """
        Helper function to retrieve all world entities of a specific type from an iterable.
        :param world_entity_type: The type of the world entity.
        :param iterable: The iterable to search for the world entity, for example self.connections or self.kinematic_structure_entities.
        :return: A list of `WorldEntity` objects that match the given type.
        """
        return [entity for entity in iterable if isinstance(entity, world_entity_type)]

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_semantic_annotation_by_name(
        self, name: Union[str, PrefixedName]
    ) -> SemanticAnnotation:
        semantic_annotation: SemanticAnnotation = (
            self._get_world_entity_by_name_from_iterable(
                name, self.semantic_annotations
            )
        )
        return semantic_annotation

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_kinematic_structure_entity_by_name(
        self, name: Union[str, PrefixedName]
    ) -> KinematicStructureEntity:
        return self._get_world_entity_by_name_from_iterable(
            name, self.kinematic_structure_entities
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_body_by_name(self, name: Union[str, PrefixedName]) -> Body:
        return self._get_world_entity_by_name_from_iterable(name, self.bodies)

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_degree_of_freedom_by_name(
        self, name: Union[str, PrefixedName]
    ) -> DegreeOfFreedom:
        return self._get_world_entity_by_name_from_iterable(
            name, self.degrees_of_freedom
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_connection_by_name(self, name: Union[str, PrefixedName]) -> Connection:
        return self._get_world_entity_by_name_from_iterable(name, self.connections)

    def _get_world_entity_by_name_from_iterable(
        self,
        name: Union[str, PrefixedName],
        world_entity_iterable: Iterable[GenericWorldEntity],
    ) -> GenericWorldEntity:
        """
        If more than one world entity matches the specified name, or if no world entity is found,
        an exception is raised.
        :param name: The name of the entity to retrieve. Can be a string or
            a `PrefixedName` instance.
        :param world_entity_iterable:
        :return: The `WorldEntity` object that matches the given name.
        :raises WorldEntityNotFoundError: If no world entity with the given name exists.
        :raises DuplicateWorldEntityError: If multiple world entities with the given name exist.
        """
        matches = self._get_world_entities_by_name_from_iterable(
            name, world_entity_iterable
        )
        match matches:
            case []:
                if isinstance(name, PrefixedName):
                    logger.warning(
                        f"No world entity with PrefixedName {name} found. Did you want a general matching of {name.name}?"
                        f"If so, please provide only the string name."
                    )
                raise WorldEntityNotFoundError(name)
            case [entity]:
                return entity
            case _:
                raise DuplicateWorldEntityError(matches)

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_semantic_annotations_by_name(
        self, name: Union[str, PrefixedName]
    ) -> List[SemanticAnnotation]:
        return self._get_world_entities_by_name_from_iterable(
            name, self.semantic_annotations
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_kinematic_structure_entities_by_name(
        self, name: Union[str, PrefixedName]
    ) -> List[KinematicStructureEntity]:
        return self._get_world_entities_by_name_from_iterable(
            name, self.kinematic_structure_entities
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_bodies_by_name(self, name: Union[str, PrefixedName]) -> List[Body]:
        return self._get_world_entities_by_name_from_iterable(name, self.bodies)

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_degrees_of_freedom_by_name(
        self, name: Union[str, PrefixedName]
    ) -> List[DegreeOfFreedom]:
        return self._get_world_entities_by_name_from_iterable(
            name, self.degrees_of_freedom
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_connections_by_name(
        self, name: Union[str, PrefixedName]
    ) -> List[Connection]:
        return self._get_world_entities_by_name_from_iterable(name, self.connections)

    @staticmethod
    def _get_world_entities_by_name_from_iterable(
        name: Union[str, PrefixedName],
        world_entity_iterable: Iterable[GenericWorldEntity],
    ) -> List[GenericWorldEntity]:
        """
        Retrieve a world entity by its name from an iterable of world entities.
        This iterable would, for example, be self.connections or self.kinematic_structure_entities.
        This method accepts either a string or a `PrefixedName` instance.
        It searches through the provided iterable and returns the list of world entities
        that matches the given name.
        If only a string was provided, it matches against the name without prefix.
        If a `PrefixedName` was provided, it matches against the full name including prefix.
        :param name: The name of the world entity to search for.
        :param world_entity_iterable: The iterable to search for the world entity, for example self.connections or self.kinematic_structure_entities.
        :return: The list of `WorldEntity` that match the given name.
        """

        match name:
            case PrefixedName():
                return [
                    world_entity
                    for world_entity in world_entity_iterable
                    if world_entity.name == name
                ]
            case str():
                return [
                    world_entity
                    for world_entity in world_entity_iterable
                    if world_entity.name.name == name
                ]

    def get_degree_of_freedom_by_id(self, id: UUID) -> DegreeOfFreedom:
        return self._get_world_entity_by_hash(hash(id))

    def get_world_entity_with_id_by_id(self, id: UUID) -> WorldEntityWithID:
        result = [
            v
            for v in self._world_entity_hash_table.values()
            if isinstance(v, WorldEntityWithID) and v.id == id
        ]
        if len(result) == 0:
            raise WorldEntityWithIDNotFoundError(id)
        else:
            return result[0]

    def get_kinematic_structure_entity_by_id(
        self, id: UUID
    ) -> KinematicStructureEntity:
        return self._get_world_entity_by_hash(hash(id))

    def get_actuator_by_id(self, id: UUID) -> Actuator:
        return self._get_world_entity_by_hash(hash(id))

    def get_semantic_annotation_by_id(self, id: UUID) -> SemanticAnnotation:
        return [s for s in self.semantic_annotations if s.id == id][0]

    def _get_world_entity_by_hash(self, entity_hash: int) -> GenericWorldEntity:
        """
        Retrieve a WorldEntity by its hash.

        :param entity_hash: The hash of the entity to retrieve.
        :return:
        """
        entity = self._world_entity_hash_table.get(entity_hash, None)
        if entity is None:
            raise WorldEntityNotFoundError(entity_hash)
        return entity

    # %% Existence Checks
    def is_semantic_annotation_in_world(
        self, semantic_annotation: SemanticAnnotation
    ) -> bool:
        return (
            semantic_annotation._world == self
            and semantic_annotation in self.semantic_annotations
        )

    def is_body_in_world(self, body: Body) -> bool:
        return self._is_world_entity_with_hash_in_world_from_iterable(hash(body))

    def is_kinematic_structure_entity_in_world(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> bool:
        return self._is_world_entity_with_hash_in_world_from_iterable(
            hash(kinematic_structure_entity)
        )

    def is_connection_in_world(self, connection: Connection) -> bool:
        return self._is_world_entity_with_hash_in_world_from_iterable(hash(connection))

    def is_degree_of_freedom_in_world(self, degree_of_freedom: DegreeOfFreedom) -> bool:
        return self._is_world_entity_with_hash_in_world_from_iterable(
            hash(degree_of_freedom)
        )

    def is_actuator_in_world(self, actuator: Actuator) -> bool:
        return self._is_world_entity_with_hash_in_world_from_iterable(hash(actuator))

    def _is_world_entity_with_hash_in_world_from_iterable(
        self, entity_hash: int
    ) -> bool:
        """
        Check if a world entity with a given hash exists in the world based on a given iterable.
        :param entity_hash: The hash of the entity to retrieve.
        :return: True if the entity exists, False otherwise.
        """
        return entity_hash in self._world_entity_hash_table

    # %% World Merging
    def merge_world_at_pose(
        self, other: World, pose: HomogeneousTransformationMatrix
    ) -> None:
        """
        Merge another world into the existing one, creates a 6DoF connection between the root of this world and the root
        of the other world.
        :param other: The world to be added.
        :param pose: world_root_T_other_root, the pose of the other world's root with respect to the current world's root
        """
        with self.modify_world():
            root_connection = Connection6DoF.create_with_dofs(
                parent=self.root, child=other.root, world=self
            )
            self.merge_world(other, root_connection)
            root_connection.origin = pose

    def merge_world(
        self,
        other: World,
        root_connection: Connection = None,
    ) -> None:
        """
        Merge a world into the existing one by merging degrees of freedom, states, connections, and bodies.
        This removes all bodies and connections from `other`.

        :param other: The world to be added.
        :param root_connection: If provided, this connection will be used to connect the two worlds. Otherwise, a new Connection6DoF will be created
        :return: None
        """
        assert other is not self, "Cannot merge a world with itself."

        with self.modify_world(), other.modify_world():
            self_root = self.root
            other_root = other.root
            self._merge_dofs_with_state_of_world(other)
            self._merge_connections_of_world(other)
            self._remove_kinematic_structure_entities_of_world(other)
            self._merge_semantic_annotations_of_world(other)

            if not root_connection and self_root:
                root_connection = Connection6DoF.create_with_dofs(
                    parent=self_root, child=other_root, world=self
                )

            if root_connection:
                self.add_connection(root_connection)

    def _merge_dofs_with_state_of_world(self, other: World):
        old_state = deepcopy(other.state)
        for dof in other.degrees_of_freedom.copy():
            other.remove_degree_of_freedom(dof)
            self.add_degree_of_freedom(dof)
        for dof_id in old_state.keys():
            self.state[dof_id] = old_state[dof_id]

    def _merge_connections_of_world(self, other: World):
        other_root = other.root
        other_connections = other.connections
        for connection in other_connections:
            other.remove_connection(connection)
            other.remove_kinematic_structure_entity(connection.parent)
            other.remove_kinematic_structure_entity(connection.child)
            self.add_connection(connection)
        other.remove_kinematic_structure_entity(other_root)
        self.add_kinematic_structure_entity(other_root)

    @staticmethod
    def _remove_kinematic_structure_entities_of_world(other: World):
        other_kse_with_world = [
            kse for kse in other.kinematic_structure_entities if kse._world is not None
        ]
        for kinematic_structure_entity in other_kse_with_world:
            other.remove_kinematic_structure_entity(kinematic_structure_entity)

    def _merge_semantic_annotations_of_world(self, other: World):
        other_semantic_annotations = [
            semantic_annotation for semantic_annotation in other.semantic_annotations
        ]
        for semantic_annotation in other_semantic_annotations:
            other.remove_semantic_annotation(semantic_annotation)
            self.add_semantic_annotation(semantic_annotation)

    # %% Subgraph Targeting
    def get_connections_of_branch(
        self, root: KinematicStructureEntity
    ) -> List[Connection]:
        """
        Collect all connections that are below root in the tree.

        :param root: The root body of the branch
        :return: List of all connections in the subtree rooted at the given body
        """
        visitor = ConnectionCollector(self)
        self._travel_branch(root, visitor)
        return visitor.connections

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def get_kinematic_structure_entities_of_branch(
        self, root: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Collect all bodies that are below root in the tree.

        :param root: The root body of the branch
        :return: List of all bodies in the subtree rooted at the given body (including the root)
        """
        descendants_indices = rx.descendants(self.kinematic_structure, root.index)
        return [root] + [
            self.kinematic_structure[index] for index in descendants_indices
        ]

    def get_direct_child_bodies_with_collision(
        self, connection: Connection
    ) -> Set[Body]:
        """
        Collect all child Bodies until a movable connection is found.

        :param connection: The connection from the kinematic structure whose child bodies will be traversed.
        :return: A set of Bodies that are moved directly by only this connection.
        """
        visitor = CollisionBodyCollector(self)
        self._travel_branch(connection.child, visitor)
        return visitor.bodies

    def _travel_branch(
        self,
        root_kinematic_structure_entity: KinematicStructureEntity,
        visitor: rustworkx.visit.DFSVisitor,
    ) -> None:
        """
        Apply a DFS Visitor to a subtree of the kinematic structure.

        :param root_kinematic_structure_entity: Starting point of the search
        :param visitor: This visitor to apply.
        """
        rx.dfs_search(
            self.kinematic_structure, [root_kinematic_structure_entity.index], visitor
        )

    def move_branch(
        self,
        branch_root: KinematicStructureEntity,
        new_parent: KinematicStructureEntity,
    ) -> None:
        """
        Destroys the connection between branch_root and its parent, and moves it to a new parent using a new connection
        of the same type. The pose of body with respect to root stays the same.

        :param branch_root: The root of the branch to be moved.
        :param new_parent: The new parent of the branch.
        """
        new_connection = None
        new_parent_T_root = self.compute_forward_kinematics(new_parent, branch_root)
        old_connection = branch_root.parent_connection

        assert isinstance(
            old_connection, (FixedConnection, Connection6DoF)
        ), "The branch root must be connected to a Connection6DoF or FixedConnection."

        match old_connection:
            case FixedConnection():
                new_connection = FixedConnection(
                    parent=new_parent,
                    child=branch_root,
                    _world=self,
                    parent_T_connection_expression=new_parent_T_root,
                )

            case Connection6DoF():
                new_connection = Connection6DoF.create_with_dofs(
                    parent=new_parent,
                    child=branch_root,
                    world=self,
                )

        with self.modify_world():
            self.add_connection(new_connection)
            self.remove_connection(old_connection)

        if isinstance(new_connection, Connection6DoF):
            new_connection.origin = new_parent_T_root

    def move_branch_to_new_world(self, new_root: KinematicStructureEntity) -> World:
        """
        Copies the subgraph of the kinematic structure from the root body to a new world and removes it from the old world.

        :param new_root: The root body of the subgraph to be copied.
        :return: A new `World` instance containing the copied subgraph.
        """
        new_world = World(name=self.name)
        child_bodies = self.compute_descendent_child_kinematic_structure_entities(
            new_root
        )
        root_connection = new_root.parent_connection

        if not child_bodies:
            with self.modify_world(), new_world.modify_world():
                self.remove_connection(root_connection)
                self.remove_kinematic_structure_entity(new_root)

                new_world.add_kinematic_structure_entity(new_root)
                return new_world

        child_body_parent_connections = [
            body.parent_connection for body in child_bodies
        ]
        child_body_dofs = [
            dof
            for connection in child_body_parent_connections
            for dof in connection.dofs
        ]

        with self.modify_world(), new_world.modify_world():
            for dof in child_body_dofs:
                self.remove_degree_of_freedom(dof)
                new_world.add_degree_of_freedom(dof)
            for connection in child_body_parent_connections:
                self.remove_kinematic_structure_entity(connection.parent)
                self.remove_kinematic_structure_entity(connection.child)
                new_world.remove_connection(connection)
                new_world.add_connection(connection)
            self.remove_connection(root_connection)

        return new_world

    # %% Change Notifications
    def notify_state_change(self) -> None:
        """
        If you have changed the state of the world, call this function to trigger necessary events and increase
        the state version.
        """
        if not self.is_empty():
            self._forward_kinematic_manager.recompute()
        self.state._notify_state_change()

    def _notify_model_change(self) -> None:
        """
        Notifies the system of a model change and updates the necessary states, caches,
        and forward kinematics expressions while also triggering registered callbacks
        for model changes.
        """
        self._model_manager.update_model_version_and_notify_callbacks()
        self._compile_forward_kinematics_expressions()
        self.notify_state_change()

        for callback in self.state.state_change_callbacks:
            callback.update_previous_world_state()

        self.validate()
        self._collision_pair_manager.disable_non_robot_collisions()
        self._collision_pair_manager.disable_collisions_for_adjacent_bodies()

    def delete_orphaned_dofs(self):
        actual_dofs = {
            dof for connection in self.connections for dof in connection.dofs
        }

        removed_dofs = set(self.degrees_of_freedom) - actual_dofs

        for dof in removed_dofs:
            self.remove_degree_of_freedom(dof)

    # %% Kinematic Structure Computations
    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_descendent_child_kinematic_structure_entities(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Computes all child entities of a given KinematicStructureEntity in the world recursively.
        :param kinematic_structure_entity: The KinematicStructureEntity for which to compute children.
        :return: A list of all child KinematicStructureEntities.
        """
        children = self.compute_child_kinematic_structure_entities(
            kinematic_structure_entity
        )
        for child in children:
            children.extend(
                self.compute_descendent_child_kinematic_structure_entities(child)
            )
        return children

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_child_kinematic_structure_entities(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Computes the child entities of a given KinematicStructureEntity in the world.
        :param kinematic_structure_entity: The KinematicStructureEntity for which to compute children.
        :return: A list of child KinematicStructureEntities.
        """
        return list(
            self.kinematic_structure.successors(kinematic_structure_entity.index)
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_parent_connection(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> Optional[Connection]:
        """
        Computes the parent connection of a given KinematicStructureEntity in the world.
        :param kinematic_structure_entity: The entityKinematicStructureEntity for which to compute the parent connection.
        :return: The parent connection of the given KinematicStructureEntity.
        """
        parent = self.compute_parent_kinematic_structure_entity(
            kinematic_structure_entity
        )
        return (
            None
            if parent is None
            else self.kinematic_structure.get_edge_data(
                parent.index, kinematic_structure_entity.index
            )
        )

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_parent_kinematic_structure_entity(
        self, kinematic_structure_entity: KinematicStructureEntity
    ) -> Optional[KinematicStructureEntity]:
        """
        Computes the parent KinematicStructureEntity of a given KinematicStructureEntity in the world.
        :param kinematic_structure_entity: The KinematicStructureEntity for which to compute the parent KinematicStructureEntity.
        :return: The parent KinematicStructureEntity of the given KinematicStructureEntity.
         If the given KinematicStructureEntity is the root, None is returned.
        """
        parent = self.kinematic_structure.predecessors(kinematic_structure_entity.index)
        return parent[0] if parent else None

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_chain_of_connections(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> List[Connection]:
        """
        Computes the chain of connections between root and tip. Can handle chains that start and end anywhere in the tree.
        """
        entity_chain = self.compute_chain_of_kinematic_structure_entities(root, tip)
        return [
            self.get_connection(entity_chain[i], entity_chain[i + 1])
            for i in range(len(entity_chain) - 1)
        ]

    def is_body_controlled(self, body: KinematicStructureEntity) -> bool:
        return self.is_controlled_connection_in_chain(self.root, body)

    def is_controlled_connection_in_chain(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> bool:
        root_part, tip_part = self.compute_split_chain_of_connections(root, tip)
        connections = root_part + tip_part
        return any(connection.is_controlled for connection in connections)

    def compute_chain_reduced_to_controlled_connections(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> Tuple[KinematicStructureEntity, KinematicStructureEntity]:
        """
        Removes root and tip links until they are both connected with a controlled connection.
        Useful for implementing collision avoidance.

        1. Compute the kinematic chain of bodies between root and tip.
        2. Remove all entries from link_a downward until one is connected with a connection from this semantic annotation.
        2. Remove all entries from link_b upward until one is connected with a connection from this semantic annotation.

        :param root: start of the chain
        :param tip: end of the chain
        :return: start and end link of the reduced chain
        """
        downward_chain, upward_chain = self.compute_split_chain_of_connections(
            root=root, tip=tip
        )
        chain = downward_chain + upward_chain

        new_root = next(
            (conn for conn in chain if conn.is_controlled),
            None,
        )

        new_tip = next(
            (conn for conn in reversed(chain) if conn.is_controlled),
            None,
        )
        assert (
            new_root is not None and new_tip is not None
        ), f"no controlled connection in chain between {root} and {tip}"

        # if new_root is in the downward chain, we need to "flip" it by returning its child
        new_root_body = new_root.parent if new_root in upward_chain else new_root.child

        # if new_tip is in the downward chain, we need to "flip" it by returning its parent
        new_tip_body = new_tip.parent if new_tip in downward_chain else new_tip.child
        return new_root_body, new_tip_body

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_split_chain_of_connections(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> Tuple[List[Connection], List[Connection]]:
        """
        Computes split chains of connections between 'root' and 'tip' bodies. Returns tuple of two Connection lists:
        (root->common ancestor, tip->common ancestor). Returns empty lists if root==tip.

        :param root: The starting `KinematicStructureEntity` object for the chain of connections.
        :param tip: The ending `KinematicStructureEntity` object for the chain of connections.
        :return: A tuple of two lists: the first list contains `Connection` objects from the `root` to
            the common ancestor, and the second list contains `Connection` objects from the `tip` to the
            common ancestor.
        """
        if root == tip:
            return [], []
        root_chain, common_ancestor, tip_chain = (
            self.compute_split_chain_of_kinematic_structure_entities(root, tip)
        )
        root_chain = root_chain + [common_ancestor[0]]
        tip_chain = [common_ancestor[0]] + tip_chain

        root_connections = [
            self.get_connection(root_chain[i + 1], root_chain[i])
            for i in range(len(root_chain) - 1)
        ]

        tip_connections = [
            self.get_connection(tip_chain[i], tip_chain[i + 1])
            for i in range(len(tip_chain) - 1)
        ]
        return root_connections, tip_connections

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_split_chain_of_kinematic_structure_entities(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> Tuple[
        List[KinematicStructureEntity],
        List[KinematicStructureEntity],
        List[KinematicStructureEntity],
    ]:
        """
        Computes the chain between root and tip. Can handle chains that start and end anywhere in the tree.
        :param root: The root KinematicStructureEntity to start the chain from
        :param tip: The tip KinematicStructureEntity to end the chain at
        :return: tuple containing
                    1. chain from root to the common ancestor (excluding common ancestor)
                    2. list containing just the common ancestor
                    3. chain from common ancestor to tip (excluding common ancestor)
        """
        if root == tip:
            return [], [root], []

        # Paths from the tree's true root to each endpoint (inclusive).
        root_path = self._compute_chain_of_kinematic_structure_entities_indexes(
            self.root, root
        )
        tip_path = self._compute_chain_of_kinematic_structure_entities_indexes(
            self.root, tip
        )

        # Find the lowest common ancestor (LCA) index in the paths.
        LCA_index = self._find_lowest_common_ancestor(root_path, tip_path)

        # The last common ancestor is the last common node.
        common_ancestor_node_index = root_path[LCA_index]
        common_ancestor = self.kinematic_structure[common_ancestor_node_index]

        # 1) From `root` up to just below lowest common ancestor.
        up_from_root_index = list(reversed(root_path[LCA_index + 1 :]))
        up_from_root = [self.kinematic_structure[index] for index in up_from_root_index]

        # 3) From just below lowest common ancestor down to `tip` (in CA->tip order, excluding CA).
        down_to_tip_index = tip_path[LCA_index + 1 :]
        down_to_tip = [self.kinematic_structure[index] for index in down_to_tip_index]

        return up_from_root, [common_ancestor], down_to_tip

    def _find_lowest_common_ancestor(
        self, root_path: List[int], tip_path: List[int]
    ) -> int:
        """
        Find the index of the lowest common ancestor, which is the index where the two paths diverge, minus 1.

        :param root_path: The path from the root to the first entity.
        :param tip_path: The path from the root to the second entity.
        :return: The index where the paths diverge.
        """
        max_index = min(len(root_path), len(tip_path))
        root_path = np.array(root_path[:max_index])
        tip_path = np.array(tip_path[:max_index])
        differing_indices = np.where(root_path != tip_path)[0]
        divergence_index = (
            differing_indices[0] if len(differing_indices) > 0 else max_index
        )

        return divergence_index - 1

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def compute_chain_of_kinematic_structure_entities(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> List[KinematicStructureEntity]:
        """
        Computes the chain between root and tip. Can handle chains that start and end anywhere in the tree.
        """
        path_indeces = self._compute_chain_of_kinematic_structure_entities_indexes(
            root, tip
        )
        return [self.kinematic_structure[index] for index in path_indeces]

    @lru_cache(maxsize=_LRU_CACHE_SIZE)
    def _compute_chain_of_kinematic_structure_entities_indexes(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> List[int]:
        """
        Computes the chain between root and tip. Can handle chains that start and end anywhere in the tree.
        """
        if root == tip:
            return [root.index]
        shortest_paths = rx.all_shortest_paths(
            self.kinematic_structure, root.index, tip.index, as_undirected=False
        )

        assert len(shortest_paths), f"No path found from {root} to {tip}"

        return shortest_paths[0]

    # %% Forward Kinematics
    def _compile_forward_kinematics_expressions(self) -> None:
        """
        Traverse the kinematic structure and compile forward kinematics expressions for fast evaluation.
        """

        if self.is_empty():
            return
        if self._forward_kinematic_manager is None:
            self._forward_kinematic_manager = ForwardKinematicsManager(self)
        self._forward_kinematic_manager.recompile()

    def compute_forward_kinematics(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        """
        Compute the forward kinematics from the root KinematicStructureEntity to the tip KinematicStructureEntity.

        Calculate the transformation matrix representing the pose of the
        tip KinematicStructureEntity relative to the root KinematicStructureEntity.

        :param root: Root KinematicStructureEntity, for which the kinematics are computed.
        :param tip: Tip KinematicStructureEntity, to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip KinematicStructureEntity with respect to the root KinematicStructureEntity.
        """
        return self._forward_kinematic_manager.compute(root, tip)

    def compose_forward_kinematics_expression(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        """
        :param root: The root KinematicStructureEntity in the kinematic chain.
            It determines the starting point of the forward kinematics calculation.
        :param tip: The tip KinematicStructureEntity in the kinematic chain.
            It determines the endpoint of the forward kinematics calculation.
        :return: An expression representing the computed forward kinematics of the tip KinematicStructureEntity relative to the root KinematicStructureEntity.
        """
        return self._forward_kinematic_manager.compose_expression(root, tip)

    def compute_forward_kinematics_np(
        self, root: KinematicStructureEntity, tip: KinematicStructureEntity
    ) -> NpMatrix4x4:
        """
        Compute the forward kinematics from the root KinematicStructureEntity to the tip KinematicStructureEntity, root_T_tip and return it as a 4x4 numpy ndarray.

        Calculate the transformation matrix representing the pose of the
        tip KinematicStructureEntity relative to the root KinematicStructureEntity, expressed as a numpy ndarray.

        :param root: Root KinematicStructureEntity, for which the kinematics are computed.
        :param tip: Tip KinematicStructureEntity, to which the kinematics are computed.
        :return: Transformation matrix representing the relative pose of the tip KinematicStructureEntity with respect to the root KinematicStructureEntity.
        """
        return self._forward_kinematic_manager.compute_np(root, tip).copy()

    def compute_forward_kinematics_of_all_collision_bodies(self) -> np.ndarray:
        """
        Computes a 4 by X matrix, with the forward kinematics of all collision bodies stacked on top each other.
        The entries are sorted by name of body.
        """
        return self._forward_kinematic_manager.collision_fks

    # %% Inverse Kinematics
    def compute_inverse_kinematics(
        self,
        root: KinematicStructureEntity,
        tip: KinematicStructureEntity,
        target: HomogeneousTransformationMatrix,
        dt: float = 0.05,
        max_iterations: int = 200,
        translation_velocity: float = 0.2,
        rotation_velocity: float = 0.2,
    ) -> Dict[DegreeOfFreedom, float]:
        """
        Compute inverse kinematics using quadratic programming.

        :param root: Root KinematicStructureEntity of the kinematic chain.
        :param tip: Tip KinematicStructureEntity of the kinematic chain.
        :param target: Desired tip pose relative to the root KinematicStructureEntity.
        :param dt: Time step for integration.
        :param max_iterations: Maximum number of iterations.
        :param translation_velocity: Maximum translation velocity.
        :param rotation_velocity: Maximum rotation velocity.
        :return: Dictionary mapping DOF names to their computed positions.
        """
        ik_solver = InverseKinematicsSolver(self)
        return ik_solver.solve(
            root,
            tip,
            target,
            dt,
            max_iterations,
            translation_velocity,
            rotation_velocity,
        )

    # %% World Utils
    def clear(self):
        """
        Clears all stored data and resets the state of the instance.
        """
        kse = self.kinematic_structure_entities
        with self.modify_world():
            for body in kse:
                self.remove_kinematic_structure_entity(body)

            self.semantic_annotations.clear()
            self.degrees_of_freedom.clear()
            self.state = WorldState(_world=self)

    def is_empty(self):
        """
        :return: Returns True if the world contains no kinematic_structure_entities, else False.
        """
        return not bool(len(self.kinematic_structure))

    def transform(
        self,
        spatial_object: GenericSpatialType,
        target_frame: KinematicStructureEntity,
    ) -> GenericSpatialType:
        """
        Transform a given spatial object from its reference frame to a target frame.

        Calculate the transformation from the reference frame of the provided
        spatial object to the specified target frame. Apply the transformation
        differently depending on the type of the spatial object:

        - If the object is a Quaternion, compute its rotation matrix, transform it, and
          convert back to a Quaternion.
        - For other types, apply the transformation matrix directly.

        :param spatial_object: The spatial object to be transformed.
        :param target_frame: The target KinematicStructureEntity frame to which the spatial object should
            be transformed.
        :return: The spatial object transformed to the target frame. If the input object
            is a Quaternion, the returned object is a Quaternion. Otherwise, it is the
            transformed spatial object.
        """
        target_frame_T_reference_frame = self.compute_forward_kinematics(
            root=target_frame, tip=spatial_object.reference_frame
        )

        match spatial_object:
            case Quaternion():
                reference_frame_R = spatial_object.to_rotation_matrix()
                target_frame_R = target_frame_T_reference_frame @ reference_frame_R
                return target_frame_R.to_quaternion()
            case _:
                return target_frame_T_reference_frame @ spatial_object

    def __deepcopy__(self, memo):
        memo = {} if memo is None else memo
        me_id = id(self)
        if me_id in memo:
            return memo[me_id]

        new_world = World(name=self.name)
        memo[me_id] = new_world

        with new_world.modify_world():
            for body in self.bodies:
                new_body = Body(
                    name=body.name,
                    id=body.id,
                )
                new_world.add_kinematic_structure_entity(new_body)
                new_body.visual = body.visual.copy_for_world(new_world)
                new_body.collision = body.collision.copy_for_world(new_world)
            for region in self.regions:
                new_region = Region(
                    name=region.name,
                    area=region.area,
                    id=region.id,
                )
                new_world.add_kinematic_structure_entity(new_region)
            for dof in self.degrees_of_freedom:
                new_dof = DegreeOfFreedom(
                    name=dof.name,
                    lower_limits=dof.lower_limits,
                    upper_limits=dof.upper_limits,
                    id=dof.id,
                )
                new_world.add_degree_of_freedom(new_dof)
                new_world.state[dof.id] = self.state[dof.id].data
            for connection in self.connections:
                new_connection = connection.copy_for_world(new_world)
                new_world.add_connection(new_connection)
        return new_world

    # %% Associations
    def load_collision_srdf(self, file_path: str):
        self._collision_pair_manager.load_collision_srdf(file_path)

    def modify_world(self) -> WorldModelUpdateContextManager:
        return WorldModelUpdateContextManager(world=self)

    def reset_state_context(self) -> ResetStateContextManager:
        return ResetStateContextManager(self)

    def get_world_model_manager(self) -> WorldModelManager:
        return self._model_manager

    @cached_property
    def collision_detector(self) -> CollisionDetector:
        """
        A collision detector for the world.
        :return: A collision detector for the world.
        """
        return TrimeshCollisionDetector(self)

    @cached_property
    def ray_tracer(self) -> RayTracer:
        """
        A ray tracer for the world.
        :return: A ray tracer for the world.
        """
        return RayTracer(self)

    def apply_control_commands(
        self, commands: np.ndarray, dt: float, derivative: Derivatives
    ) -> None:
        """
        Updates the state of a system by applying control commands at a specified derivative level,
        followed by backward integration to update lower derivatives.

        :param commands: Control commands to be applied at the specified derivative
            level. The array length must match the number of free variables
            in the system.
        :param dt: Time step used for the integration of lower derivatives.
        :param derivative: The derivative level to which the control commands are
            applied.
        """
        self.state._apply_control_commands(commands, dt, derivative)
        for connection in self.connections:
            match connection:
                case HasUpdateState():
                    connection.update_state(dt)
                case _:
                    pass
        self.notify_state_change()

    def set_positions_1DOF_connection(
        self, new_state: Dict[ActiveConnection1DOF, float]
    ) -> None:
        """
        Set the positions of 1DOF connections and notify the world of the state change.
        """
        for connection, value in new_state.items():
            connection.position = value
        self.notify_state_change()
