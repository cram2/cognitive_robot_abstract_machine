from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Self, Dict, Type, TypeVar, TYPE_CHECKING

from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from .auxilary_variable_manager import AuxiliaryVariableManager, AuxiliaryVariable
    from .exceptions import MissingContextExtensionError
    from ..qp.qp_controller_config import QPControllerConfig
    from ..model.collision_world_syncer import CollisionWorldSynchronizer


@dataclass
class ContextExtension:
    """
    Context extension for build context.
    Used together with require_extension to augment BuildContext with custom data.
    """


GenericContextExtension = TypeVar("GenericContextExtension", bound=ContextExtension)


@dataclass
class BuildContext:
    """
    Context used during the build phase of a MotionStatechartNode.
    """

    world: World
    """There world in which to execute the Motion Statechart."""
    auxiliary_variable_manager: AuxiliaryVariableManager
    """Auxiliary variable manager used by nodes to create auxiliary variables."""
    collision_scene: CollisionWorldSynchronizer
    """Synchronization of the collision world with the world in which the Motion Statechart is executed."""
    qp_controller_config: QPControllerConfig
    """Configuration of the QP controller used to solve the QP problem."""
    control_cycle_variable: AuxiliaryVariable
    """Auxiliary variable used to count control cycles, can be used my Motion StatechartNodes to implement time-dependent actions."""
    extensions: Dict[Type[ContextExtension], ContextExtension] = field(
        default_factory=dict, repr=False, init=False
    )
    """
    Dictionary of extensions used to augment the build context.
    Ros2 extensions are automatically added to the build context when using the Ros2Executor.
    """

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
        self.extensions[type(extension)] = extension

    @classmethod
    def empty(cls) -> Self:
        return cls(
            world=World(),
            auxiliary_variable_manager=None,
            collision_scene=None,
            qp_controller_config=None,
            control_cycle_variable=None,
        )


@dataclass
class ExecutionContext:
    world: World
    external_collision_data_data: np.ndarray
    self_collision_data_data: np.ndarray
    auxiliar_variables_data: np.ndarray
    control_cycle_counter: int
