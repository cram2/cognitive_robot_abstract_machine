from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from typing_extensions import Dict

from krrood.adapters.json_serializer import SubclassJSONSerializer
from ..world_description.world_entity import WorldEntityWithClassBasedID

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Callback(WorldEntityWithClassBasedID, SubclassJSONSerializer, ABC):
    """
    Callback is an abstract base class (ABC)
    reacting to changes in the associated `_world`.
    It provides a flexible mechanism for subclasses to implement custom behaviors to be triggered
    whenever a change occurs.

    The primary purpose of this class is to encapsulate logic that needs to be
    executed as a response to certain events or changes within the `_world` object.
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

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "is_paused": self._is_paused}

    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        instance = super()._from_json(data, **kwargs)
        instance._is_paused = data.get("is_paused", False)
        return instance


@dataclass(eq=False)
class StateChangeCallback(Callback, ABC):
    """
    Callback for handling state changes.
    """

    previous_world_state_data: np.ndarray = field(init=False)
    """
    The previous world state data used to check if something changed.
    """

    def __post_init__(self):
        self._world.state.state_change_callbacks.append(self)
        self.update_previous_world_state()

    def stop(self):
        try:
            self._world.state.state_change_callbacks.remove(self)
        except ValueError:
            pass

    def update_previous_world_state(self):
        """
        Update the previous world state to reflect the current world positions.
        """
        self.previous_world_state_data = np.copy(self._world.state.positions)


@dataclass(eq=False)
class ModelChangeCallback(Callback, ABC):
    """
    Callback for handling model changes.
    """

    def __post_init__(self):
        super().__post_init__()
        self._world.get_world_model_manager().model_change_callbacks.append(self)

    def stop(self):
        try:
            self._world.get_world_model_manager().model_change_callbacks.remove(self)
        except ValueError:
            pass
