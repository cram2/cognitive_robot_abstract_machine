"""
Objects whose attributes or elements are set by code of their own class, which may
change more than the value it stores, next to objects that only store what they are
given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any

# %% setting that runs code


@dataclass
class AttributeSetThroughProperty:
    """
    A lamp whose state is set through a property that also counts the switches.
    """

    switch_count: int = 0
    """
    How often the lamp was switched.
    """

    _is_lit: bool = False
    """
    Whether the lamp is lit.
    """

    @property
    def is_lit(self) -> bool:
        """
        :return: Whether the lamp is lit.
        """
        return self._is_lit

    @is_lit.setter
    def is_lit(self, value: bool) -> None:
        """
        :param value: Whether the lamp is lit from now on.
        """
        self._is_lit = value
        self.switch_count += 1


@dataclass
class AttributesSetThroughSetattr:
    """
    A record that remembers which of its attributes were set.
    """

    status: str = "new"
    """
    Where the record stands.
    """

    changed_attributes: list[str] = field(default_factory=list)
    """
    The names of the attributes that were set, in order.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "changed_attributes" and "changed_attributes" in self.__dict__:
            self.changed_attributes.append(name)
        super().__setattr__(name, value)


class ElementsSetThroughSetitem(dict):
    """
    A dictionary that remembers which keys were set.
    """

    def __init__(self, *arguments: Any, **keyword_arguments: Any) -> None:
        super().__init__(*arguments, **keyword_arguments)
        self.changed_keys: list[Any] = []
        """
        The keys that were set, in order.
        """

    def __setitem__(self, key: Any, value: Any) -> None:
        self.changed_keys.append(key)
        super().__setitem__(key, value)


# %% setting that only stores the value


@dataclass(slots=True)
class SlottedAttributes:
    """
    A point whose coordinate is stored in a slot.
    """

    coordinate: float
    """
    Where the point lies on a line.
    """


@dataclass(frozen=True)
class FrozenAttributes:
    """
    A point whose coordinate never changes on the live object.
    """

    coordinate: float
    """
    Where the point lies on a line.
    """


@dataclass
class ElementsInAnImmutableContainer:
    """
    A segment whose end points are stored in a tuple, which cannot be changed on the
    live object.
    """

    end_points: tuple[float, float]
    """
    Where the segment starts and ends on a line.
    """


@dataclass
class AttributeReadThroughProperty:
    """
    A circle whose diameter is computed and cannot be set on the live object.
    """

    radius: float
    """
    The radius of the circle.
    """

    @property
    def diameter(self) -> float:
        """
        :return: Twice the radius.
        """
        return 2 * self.radius
