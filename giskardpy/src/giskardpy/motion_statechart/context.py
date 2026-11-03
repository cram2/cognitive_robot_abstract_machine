from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from cramph.context import ContextExtension
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    BaseCollisionVariableManager,
    SelfCollisionVariableManager,
    ExternalCollisionVariableManager,
)

if TYPE_CHECKING:
    from cramph.context import StatechartContext


@dataclass
class MotionControlContext(ContextExtension):
    """
    What motion statechart nodes need from motion control while they are built and
    ticked.
    """

    qp_controller_config: QPControllerConfig
    """
    The configuration of the QP controller that turns the constraints of the nodes into
    commands.
    """

    self_collision_manager: SelfCollisionVariableManager | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    The self collision variable manager shared by all self collision avoidance nodes,
    None until one of them requests it.
    """

    external_collision_manager: ExternalCollisionVariableManager | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    The external collision variable manager shared by all external collision avoidance
    nodes, None until one of them requests it.
    """

    def get_or_create_self_collision_manager(
        self, context: StatechartContext
    ) -> SelfCollisionVariableManager:
        """
        :param context: The context this extension belongs to.
        :return: :attr:`self_collision_manager`, created in `context` on first request.
        """
        if self.self_collision_manager is None:
            self.self_collision_manager = SelfCollisionVariableManager(
                context.float_variable_data
            )
            context.world.collision_manager.add_collision_consumer(
                self.self_collision_manager
            )
        return self.self_collision_manager

    def get_or_create_external_collision_manager(
        self, context: StatechartContext
    ) -> ExternalCollisionVariableManager:
        """
        :param context: The context this extension belongs to.
        :return: :attr:`external_collision_manager`, created in `context` on first
            request.
        """
        if self.external_collision_manager is None:
            self.external_collision_manager = ExternalCollisionVariableManager(
                context.float_variable_data
            )
            context.world.collision_manager.add_collision_consumer(
                self.external_collision_manager
            )
        return self.external_collision_manager

    @property
    def _registered_collision_variable_managers(
        self,
    ) -> list[BaseCollisionVariableManager]:
        """
        :return: The collision variable managers that nodes have requested so far.
        """
        return [
            manager
            for manager in (
                self.self_collision_manager,
                self.external_collision_manager,
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

    def cleanup(self) -> None:
        """
        Removes the collision variable managers from the collision manager they were
        added to.
        """
        for manager in self._registered_collision_variable_managers:
            manager.collision_manager.remove_collision_consumer(manager)
        self.self_collision_manager = None
        self.external_collision_manager = None
