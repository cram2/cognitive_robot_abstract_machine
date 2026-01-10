from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Self, Dict, Type, TypeVar, Optional

from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
    AuxiliaryVariable,
)
from giskardpy.motion_statechart.exceptions import MissingContextExtensionError
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
class BuildContext:
    """
    Context used during the build phase of a MotionStatechartNode.
    """

    world: World
    auxiliary_variable_manager: AuxiliaryVariableManager
    collision_scene: CollisionWorldSynchronizer
    qp_controller_config: QPControllerConfig
    control_cycle_variable: AuxiliaryVariable
    extensions: Dict[Type[ContextExtension], ContextExtension] = field(
        default_factory=dict, repr=False, init=False
    )

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
