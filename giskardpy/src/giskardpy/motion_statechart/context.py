from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Self, Dict, List, Optional, Type, TypeVar, TYPE_CHECKING

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.collision_checking.collision_manager import CollisionManager
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    BaseCollisionVariableManager,
    SelfCollisionVariableManager,
    ExternalCollisionVariableManager,
)
from giskardpy.motion_statechart.exceptions import (
    MissingContextExtensionError,
    DuplicateContextExtensionError,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig

from semantic_digital_twin.world import World


@dataclass
class ContextExtension:
    """
    Context extension for build context.

    Used together with require_extension to augment BuildContext with custom data.
    """


GenericContextExtension = TypeVar("GenericContextExtension", bound=ContextExtension)


@dataclass
class MotionStatechartContext:
    """
    Context used during the build phase of a MotionStatechartNode.
    """

    world: World
    """
    There world in which to execute the Motion Statechart.
    """

    control_cycle_variable: FloatVariable = field(init=False)
    """
    Auxiliary variable used to count control cycles, can be used my Motion
    StatechartNodes to implement time-dependent actions.
    """

    float_variable_data: FloatVariableData = field(default_factory=FloatVariableData)
    """
    Data structure used to store auxiliary variables.
    """

    qp_controller_config: QPControllerConfig = field(
        default_factory=QPControllerConfig.create_with_simulation_defaults
    )
    """
    Optional configuration for the QP Controller.

    Is only needed when constraints are present in the motion statechart.
    """

    extensions: Dict[Type[ContextExtension], ContextExtension] = field(
        default_factory=dict, repr=False, init=False
    )
    """
    Dictionary of extensions used to augment the build context.

    Ros2 extensions are automatically added to the build context when using the
    Ros2Executor.
    """

    _self_collision_manager: Optional[SelfCollisionVariableManager] = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Backs :attr:`self_collision_manager`, None until a node requests it.
    """

    _external_collision_manager: Optional[ExternalCollisionVariableManager] = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Backs :attr:`external_collision_manager`, None until a node requests it.
    """

    @property
    def collision_manager(self) -> CollisionManager:
        return self.world.collision_manager

    @property
    def self_collision_manager(self) -> SelfCollisionVariableManager:
        """
        SelfCollisionVariableManager shared by all self collision avoidance nodes,
        created on first access.
        """
        if self._self_collision_manager is None:
            self._self_collision_manager = SelfCollisionVariableManager(
                self.float_variable_data
            )
            self.collision_manager.add_collision_consumer(self._self_collision_manager)
        return self._self_collision_manager

    @property
    def external_collision_manager(self) -> ExternalCollisionVariableManager:
        """
        ExternalCollisionVariableManager shared by all external collision avoidance
        nodes, created on first access.
        """
        if self._external_collision_manager is None:
            self._external_collision_manager = ExternalCollisionVariableManager(
                self.float_variable_data
            )
            self.collision_manager.add_collision_consumer(
                self._external_collision_manager
            )
        return self._external_collision_manager

    @property
    def _registered_collision_variable_managers(
        self,
    ) -> List[BaseCollisionVariableManager]:
        """
        :return: The collision variable managers that nodes have requested so far.
        """
        return [
            manager
            for manager in (
                self._self_collision_manager,
                self._external_collision_manager,
            )
            if manager is not None
        ]

    @property
    def requires_collision_checking(self) -> bool:
        """
        :return: True if a node requested a collision variable manager and therefore
            needs collisions to be computed in every control cycle.
        """
        return len(self._registered_collision_variable_managers) > 0

    def require_extension(
        self, extension_type: Type[GenericContextExtension]
    ) -> GenericContextExtension:
        """
        Return an extension instance or raise ``MissingContextExtensionError``.
        """
        extension = self.extensions.get(extension_type)
        if extension is None:
            raise MissingContextExtensionError(expected_extension=extension_type)
        return extension

    def add_extension(self, extension: GenericContextExtension):
        """
        Extend the build context with a custom extension.
        """
        extension_type = type(extension)
        if extension_type in self.extensions:
            raise DuplicateContextExtensionError(extension_type=extension_type)
        self.extensions[extension_type] = extension

    def cleanup(self):
        """
        Removes the lazy-initialized collision managers from the collision manager.
        """
        for manager in self._registered_collision_variable_managers:
            self.collision_manager.remove_collision_consumer(manager)
        self._self_collision_manager = None
        self._external_collision_manager = None

    @classmethod
    def empty(cls) -> Self:
        return cls(
            world=World(),
            float_variable_data=None,
            qp_controller_config=None,
        )
