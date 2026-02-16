from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from typing_extensions import TYPE_CHECKING, Callable, Dict
from uuid import UUID

from ..datastructures.prefixed_name import PrefixedName

if TYPE_CHECKING:
    from ..world import World

logger = logging.getLogger(__name__)


@dataclass
class Callback(ABC):
    """
    Callback is an abstract base class (ABC)
    reacting to changes in the associated `world`.
    It provides a flexible  mechanism for subclasses to implement custom behaviors to be triggered
    whenever a change occurs.

    The primary purpose of this class is to encapsulate logic that needs to be
    executed as a response to certain events or changes within the `world` object.
    """

    world: World
    """
    The world this callback is listening on.
    """

    _is_paused = False
    """
    Flag that indicates if the callback is paused.
    """

    def notify(self, **kwargs):
        """
        Notify the callback of a change in the world.
        """
        if self._is_paused:
            pass
        else:
            self._notify(**kwargs)

    @abstractmethod
    def _notify(self, **kwargs):
        """
        Notify the callback of a change in the world.
        Override this method to implement custom behaviors.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """
        Stop the callback.
        """
        raise NotImplementedError

    def pause(self):
        """
        Pause the callback such that notify does not trigger anymore.
        """
        self._is_paused = True

    def resume(self):
        """
        Resume the callback such that notify does trigger again.
        """
        self._is_paused = False


@dataclass
class StateChangeCallback(Callback, ABC):
    """
    Callback for handling state changes.
    """

    previous_world_state_data: np.ndarray = field(init=False)
    """
    The previous world state data used to check if something changed.
    """

    def __post_init__(self):
        self.world.state.state_change_callbacks.append(self)
        self.update_previous_world_state()

    def stop(self):
        try:
            self.world.state.state_change_callbacks.remove(self)
        except ValueError:
            pass

    def update_previous_world_state(self):
        """
        Update the previous world state to reflect the current world positions.
        """
        self.previous_world_state_data = np.copy(self.world.state.positions)


@dataclass
class ModelChangeCallback(Callback, ABC):
    """
    Callback for handling model changes.
    """

    def __post_init__(self):
        self.world.get_world_model_manager().model_change_callbacks.append(self)

    def stop(self):
        try:
            self.world.get_world_model_manager().model_change_callbacks.remove(self)
        except ValueError:
            pass
