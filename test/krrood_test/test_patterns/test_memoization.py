"""
Storage and ownership contract of :mod:`krrood.patterns.caching`'s memoization.
"""

from __future__ import annotations

import gc
import pickle
import weakref
from copy import deepcopy
from dataclasses import dataclass
from typing_extensions import Any, ClassVar, Generic, TypeVar, get_origin

import pytest

from krrood.patterns.caching import (
    MemoizationCache,
    clear_memoization_cache,
    memoize,
)
from krrood.patterns.exceptions import UnmemoizableOwnerError

# %% mimics


@dataclass(eq=False)
class SquaringOwner:
    """
    An owner of a memoized method that is also copied and pickled.

    Defined at module level because :mod:`pickle` resolves a class by name.
    """

    computation_count: int = 0
    """
    Number of times :meth:`square` ran its body rather than returning a cached value.
    """

    @memoize
    def square(self, value: int) -> int:
        self.computation_count += 1
        return value * value


@dataclass(eq=False)
class SelfReferencingOwner:
    """
    An owner whose memoized result refers back to the owner itself.
    """

    @memoize
    def identity(self) -> SelfReferencingOwner:
        return self


@memoize
def name_tokens(name: str) -> tuple[str, ...]:
    """
    A memoized function whose receiver is a ``str``, which cannot carry a cache.
    """
    return tuple(name.split("_"))


ElementType = TypeVar("ElementType")


@memoize
def origin_name(alias: Any) -> str:
    """
    A memoized function whose receiver is a parametrized generic rather than an
    instance.
    """
    return get_origin(alias).__name__


# %% the cache does not travel with the owner


def test_owner_of_a_memoized_method_can_be_deep_copied():
    """
    Memoizing a call must not make its owner uncopyable, and the copy must start with an
    empty cache instead of inheriting the original's cached values.
    """
    owner = SquaringOwner()
    assert owner.square(3) == 9

    copied_owner = deepcopy(owner)

    assert copied_owner.square(3) == 9
    assert copied_owner.computation_count == owner.computation_count + 1
    assert owner.computation_count == 1


def test_owner_of_a_memoized_method_can_be_pickled():
    """
    Memoizing a call must not make its owner unpicklable.
    """
    owner = SquaringOwner()
    assert owner.square(4) == 16

    restored_owner = pickle.loads(pickle.dumps(owner))

    assert restored_owner.square(4) == 16
    assert restored_owner.computation_count == 2


def test_owner_is_collected_even_when_its_cached_value_references_it():
    """
    A memoized result that refers back to its owner, which is what a world query
    returns, must not keep that owner alive once nothing else refers to it.
    """
    owner = SelfReferencingOwner()
    owner.identity()
    assert MemoizationCache.existing_for(owner) is not None
    reference = weakref.ref(owner)

    del owner
    gc.collect()

    assert reference() is None


# %% one cache per owner, never shared along the inheritance chain


def test_clearing_a_subclass_cache_leaves_the_base_class_cache_intact():
    """
    A memoized classmethod caches per class, so clearing a subclass must not invalidate
    what its base class computed.
    """

    class TokenCountingBase:
        computation_counts: ClassVar[dict[type, int]] = {}

        @classmethod
        @memoize
        def lowercase_name(cls) -> str:
            cls.computation_counts[cls] = cls.computation_counts.get(cls, 0) + 1
            return cls.__name__.lower()

    class TokenCountingSubclass(TokenCountingBase): ...

    assert TokenCountingBase.lowercase_name() == "tokencountingbase"

    clear_memoization_cache(TokenCountingSubclass)

    assert TokenCountingBase.lowercase_name() == "tokencountingbase"
    assert TokenCountingBase.computation_counts[TokenCountingBase] == 1


def test_clearing_an_instance_cache_leaves_its_class_cache_intact():
    """
    An instance and its class are separate owners, so clearing the instance must not
    invalidate what a memoized classmethod computed.
    """

    class MixedLevelMemoizingOwner:
        class_level_computation_count: ClassVar[int] = 0

        @classmethod
        @memoize
        def lowercase_name(cls) -> str:
            MixedLevelMemoizingOwner.class_level_computation_count += 1
            return cls.__name__.lower()

        @memoize
        def identity(self) -> MixedLevelMemoizingOwner:
            return self

    owner = MixedLevelMemoizingOwner()
    MixedLevelMemoizingOwner.lowercase_name()
    owner.identity()

    clear_memoization_cache(owner)
    MixedLevelMemoizingOwner.lowercase_name()

    assert MixedLevelMemoizingOwner.class_level_computation_count == 1


# %% owners that cannot carry a cache


def test_owner_that_cannot_hold_a_cache_is_rejected():
    """
    A receiver that cannot hold attributes of its own has nowhere to keep a cache, and
    saying so is the whole contract of the decorator.
    """
    with pytest.raises(UnmemoizableOwnerError):
        name_tokens("some_name")


def test_generic_alias_is_a_supported_owner():
    """
    A parametrized generic holds its own cache instead of pushing it onto the class it
    parametrizes, so parametrizations do not share one.
    """

    class ParametrizedContainer(Generic[ElementType]): ...

    assert origin_name(ParametrizedContainer[int]) == "ParametrizedContainer"
    assert origin_name(ParametrizedContainer[float]) == "ParametrizedContainer"
