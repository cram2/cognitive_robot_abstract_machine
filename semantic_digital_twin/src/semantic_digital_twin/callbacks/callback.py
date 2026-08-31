from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Self, TYPE_CHECKING

import numpy as np
from typing_extensions import Dict, List

from krrood.adapters.json_serializer import SubclassJSONSerializer
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityWithClassBasedID,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Callback(WorldEntityWithClassBasedID, SubclassJSONSerializer, ABC):
    """
    Callback is an abstract base class (ABC) reacting to changes in the associated
    `_world`. It provides a flexible mechanism for subclasses to implement custom
    behaviors to be triggered whenever a change occurs.

    The primary purpose of this class is to encapsulate logic that needs to be executed
    as a response to certain events or changes within the `_world` object.
    """

    _is_paused = False
    """
    Flag that indicates if the callback is paused.
    """

    @classmethod
    @abstractmethod
    def _all_callbacks_from_world(cls, world: World) -> List[Callback]:
        """
        The list a world notifies when the change this kind of callback reacts to
        happens.

        A world keeps one such list per kind of change. A callback puts itself on the
        list of its own kind when it is created and takes itself off again when it is
        stopped, so the list is also what decides whether a callback is still live.

        :param world: The world whose list is asked for.
        :return: the callbacks on that list, of whatever class, in the order they were
            created.
        """
        raise NotImplementedError

    @classmethod
    def all_callbacks_of_this_type_from_world(cls, world: World) -> List[Self]:
        """
        The callbacks of this class that the given world still notifies.

        Two parts that share a world can find each other this way instead of one being
        handed a reference to the other, which is what lets them be created in either
        order and by unrelated code. A callback that has been stopped is no longer among
        them, and neither is one attached to a different world.

        :param world: The world to search.
        :return: every callback of this class the world notifies, in the order they were
            created.
        """
        return [
            callback
            for callback in cls._all_callbacks_from_world(world)
            if isinstance(callback, cls)
        ]

    def stop(self):
        """
        Stop the callback.

        Should be overridden by the Subclasses. Subclasses should call super().stop()
        after their own cleanup.
        """
        pass

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

    @classmethod
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

    @classmethod
    def _all_callbacks_from_world(cls, world: World) -> List[Callback]:
        """
        :param world: The world whose state change callbacks are asked for.
        :return: everything the world notifies whenever its state changes.
        """
        return world.state.state_change_callbacks

    def notify_state_change(self, **kwargs):
        if not self._is_paused:
            self.on_state_change(**kwargs)

    @abstractmethod
    def on_state_change(self, **kwargs):
        raise NotImplementedError

    def stop(self):
        try:
            self._world.state.state_change_callbacks.remove(self)
        except ValueError:
            pass
        super().stop()

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

    @classmethod
    def _all_callbacks_from_world(cls, world: World) -> List[Callback]:
        """
        :param world: The world whose model change callbacks are asked for.
        :return: everything the world notifies whenever its model changes.
        """
        return world.get_world_model_manager().model_change_callbacks

    def notify_model_change(self, **kwargs):
        if not self._is_paused:
            self.on_model_change(**kwargs)

    @abstractmethod
    def on_model_change(self, **kwargs):
        raise NotImplementedError

    def stop(self):
        try:
            self._world.get_world_model_manager().model_change_callbacks.remove(self)
        except ValueError:
            pass
        super().stop()
