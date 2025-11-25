from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    Generic,
    Optional,
    Iterable,
    Dict,
    Any,
    Callable,
    List,
)
from typing_extensions import TypeVar, ClassVar

from .utils import make_list, ALL

T = TypeVar("T")


@dataclass
class HashedValue(Generic[T]):
    """
    Value wrapper carrying a stable hash identifier.
    """

    value: T
    """
    The wrapped value.
    """
    id_: Optional[int] = field(default=None)
    """
    Optional explicit identifier; if omitted, derived from value.
    """

    def __post_init__(self) -> None:
        """
        Initialize the identifier from the wrapped value when not provided.
        """
        if self.id_ is not None:
            return
        if isinstance(self.value, HashedValue):
            # General nested HashedValue: unwrap
            self.id_ = self.value.id_
            self.value = self.value.value
            return
        if hasattr(self.value, "_id_"):
            self.id_ = self.value._id_
        else:
            self.id_ = id(self.value)

    def __hash__(self) -> int:
        """Hash of the identifier."""
        return hash(self.id_)

    def __eq__(self, other: object) -> bool:
        """
        Equality based on identifier, with ALL sentinel matching any value.
        """
        if isinstance(other, ALL):
            return True
        if not isinstance(other, HashedValue):
            return False
        return self.id_ == other.id_

    def __bool__(self) -> bool:
        return bool(self.value)


@dataclass
class HashedIterable(Generic[T]):
    """
    A wrapper for an iterable that hashes its items.
    This is useful for ensuring that the items in the iterable are unique and can be used as keys in a dictionary.
    """

    iterable: Iterable[HashedValue[T]] = field(default_factory=list)
    values: Dict[int, HashedValue[T]] = field(default_factory=dict)

    def __post_init__(self):
        if self.iterable and not isinstance(self.iterable, HashedIterable):
            self.iterable = (
                HashedValue(v) if not isinstance(v, HashedValue) else v
                for v in self.iterable
            )

    def set_iterable(self, iterable):
        if iterable and not isinstance(iterable, HashedIterable):
            self.iterable = (
                HashedValue(v) if not isinstance(v, HashedValue) else v
                for v in iterable
            )

    def get(self, key: int, default: Any) -> HashedValue[T]:
        return self.values.get(key, default)

    def add(self, value: Any):
        if not isinstance(value, HashedValue):
            value = HashedValue(value)
        if value.id_ not in self.values:
            self.values[value.id_] = value
        return self

    def update(self, iterable: Iterable[Any]):
        for v in iterable:
            self.add(v)

    def map(
        self,
        func: Callable[[HashedValue], HashedValue],
        ids: Optional[List[int]] = None,
    ) -> HashedIterable[T]:
        if ids:
            func = lambda v: func(v) if v.id_ in ids else v
        return HashedIterable(map(func, self))

    def filter(self, func: Callable[[HashedValue], bool]) -> HashedIterable[T]:
        return HashedIterable(filter(func, self))

    @property
    def unwrapped_values(self) -> List[T]:
        return [v.value for v in self]

    @property
    def first_value(self) -> HashedValue:
        """
        Return the first value in the iterable.
        """
        for v in self:
            return v
        raise ValueError("Tried to get a value from empty iterable")

    def clear(self):
        self.values.clear()

    def __iter__(self):
        """
        Iterate over the hashed values.

        :return: An iterator over the hashed values.
        """
        yield from self.values.values()
        for v in self.iterable:
            self.values[v.id_] = v
            yield v

    def __or__(self, other) -> HashedIterable[T]:
        return self.union(other)

    def __and__(self, other) -> HashedIterable[T]:
        return self.intersection(other)

    def intersection(self, other):
        common_keys = self.values.keys() & other.values.keys()
        common_values = {k: self.values[k] for k in common_keys}
        return HashedIterable(values=common_values)

    def difference(self, other):
        left_keys = self.values.keys() - other.values.keys()
        values = {k: self.values[k] for k in left_keys}
        return HashedIterable(values=values)

    def union(self, other):
        if not isinstance(other, HashedIterable):
            other = HashedIterable(
                values={HashedValue(v).id_: HashedValue(v) for v in make_list(other)}
            )
        all_keys = self.values.keys() | other.values.keys()
        all_values = {k: self.values.get(k, other.values.get(k)) for k in all_keys}
        return HashedIterable(values=all_values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, id_: Any) -> HashedValue:
        """
        Get the HashedValue by its id.

        :param id_: The id of the HashedValue to get.
        :return: The HashedValue with the given id.
        :raises KeyError: If the given id is unknown.
        """
        if isinstance(id_, HashedValue):
            id_ = id_.id_
        elif not isinstance(id_, int):
            id_ = HashedValue(id_).id_
        try:
            return self.values[id_]
        except KeyError:
            for v in self.iterable:
                self.values[v.id_] = v
                if v.id_ == id_:
                    return v
            raise KeyError(id_)

    def __setitem__(self, id_: int, value: HashedValue[T]):
        """
        Set the HashedValue by its id.

        :param id_: The id of the HashedValue to set.
        :param value: The HashedValue to set.
        """
        self.values[id_] = value

    def __copy__(self):
        """
        Create a shallow copy of the HashedIterable.

        :return: A new HashedIterable instance with the same values.
        """
        return HashedIterable(values=self.values.copy())

    def __contains__(self, item):
        return item in self.values

    def __hash__(self):
        return hash(tuple(sorted(self.values.keys())))

    def __eq__(self, other):
        keys_are_equal = self.values.keys() == other.values.keys()
        if not keys_are_equal:
            return False
        values_are_equal = all(
            my_v == other_v
            for my_v, other_v in zip(self.values.values(), other.values.values())
        )
        return values_are_equal

    def __bool__(self):
        return bool(self.values) or bool(self.iterable)
