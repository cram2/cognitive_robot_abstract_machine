"""
Thread-safety of :func:`krrood.patterns.caching.memoize` and
:func:`krrood.patterns.caching.copy_memoize`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from typing_extensions import Any, Callable

from krrood.patterns.caching import MemoizationCache, copy_memoize, memoize

BARRIER_TIMEOUT_IN_SECONDS = 5.0
"""
How long a thread waits at a barrier for the other thread to reach it before declaring
that the two computations cannot overlap.
"""

# %% running calls concurrently


def run_concurrently(
    calls: list[Callable[[], Any]],
) -> tuple[list[Any], list[BaseException]]:
    """
    Runs every call in its own thread and collects the results and the errors raised.

    Threads are daemons joined with a timeout, so a deadlock fails the test instead of
    hanging the suite.
    """
    results: list[Any] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(call: Callable[[], Any]) -> None:
        try:
            result = call()
            with lock:
                results.append(result)
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [
        threading.Thread(target=run, args=(call,), daemon=True) for call in calls
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=BARRIER_TIMEOUT_IN_SECONDS * 2)

    assert not any(thread.is_alive() for thread in threads)
    return results, errors


# %% mimics for the hazard: a value whose copy is unsafe to run concurrently


@dataclass
class _ReentrancyDetectingValue:
    """
    A value whose copy detects if another thread is copying the same value at the same
    time, mimicking a CasADi-backed object whose ``deepcopy`` is unsafe against
    concurrent execution.
    """

    payload: int
    """
    Arbitrary data carried by this value, used only to check that copies stay value-
    equal to the original.
    """

    _copy_in_progress: list[bool] = field(default_factory=lambda: [False])
    """
    One-element flag shared by every copy made directly from this instance, marking
    whether some thread is currently inside :meth:`__deepcopy__` on it.
    """

    def __deepcopy__(self, memo: dict) -> _ReentrancyDetectingValue:
        if self._copy_in_progress[0]:
            raise RuntimeError("concurrent deepcopy detected on the same value")
        self._copy_in_progress[0] = True
        try:
            time.sleep(0.01)
            return _ReentrancyDetectingValue(payload=self.payload)
        finally:
            self._copy_in_progress[0] = False


@dataclass(eq=False)
class _CopyMemoizingOwner:
    """
    Minimal host object exercising :func:`~krrood.patterns.caching.copy_memoize` under
    concurrency.
    """

    build_count: int = 0
    """
    Number of times :meth:`build_value` actually ran its body, rather than returning a
    cached copy.
    """

    @copy_memoize
    def build_value(self, payload: int) -> _ReentrancyDetectingValue:
        self.build_count += 1
        return _ReentrancyDetectingValue(payload=payload)


@dataclass(eq=False)
class _MemoizingOwner:
    """
    Minimal host object exercising :func:`~krrood.patterns.caching.memoize` under
    concurrency.
    """

    call_count: int = 0
    """
    Number of times :meth:`compute` actually ran its body, rather than returning a
    cached value.
    """

    @memoize
    def compute(self, key: int) -> int:
        self.call_count += 1
        return key * key


# %% mimics for lock granularity


@dataclass(eq=False)
class _RendezvousingOwner:
    """
    An owner whose memoized computations meet at a barrier, so a computation only
    finishes if another key on the same owner is computed at the same time.
    """

    rendezvous: threading.Barrier
    """
    The barrier both concurrent computations wait at.
    """

    @memoize
    def compute(self, key: int) -> int:
        self.rendezvous.wait(timeout=BARRIER_TIMEOUT_IN_SECONDS)
        return key * key


@dataclass(eq=False)
class _NestingOwner:
    """
    An owner whose memoized method calls a memoized method of another owner, mimicking
    how forward kinematics composes memoized world queries.
    """

    rendezvous: threading.Barrier
    """
    The barrier every outer call waits at, so both threads hold their own outer call
    before either descends into the other owner.
    """

    nested_owner: _NestingOwner | None = None
    """
    The owner whose :meth:`inner` the outer call descends into.
    """

    @memoize
    def outer(self, key: int) -> int:
        self.rendezvous.wait(timeout=BARRIER_TIMEOUT_IN_SECONDS)
        return self.nested_owner.inner(key)

    @memoize
    def inner(self, key: int) -> int:
        return key * 2


# %% one computation and one copy at a time per key


def test_concurrent_copy_memoize_hits_on_the_same_key_do_not_race():
    """
    Many threads calling a `@copy_memoize`-decorated method with the same arguments at
    once must never call ``deepcopy`` on the same cached value concurrently, must run
    the underlying computation exactly once, and must each get back their own,
    independent, value-equal copy.
    """
    owner = _CopyMemoizingOwner()
    thread_count = 16

    results, errors = run_concurrently(
        [lambda: owner.build_value(42) for _ in range(thread_count)]
    )

    assert errors == []
    assert owner.build_count == 1
    assert len(results) == thread_count
    assert all(result.payload == 42 for result in results)
    assert len({id(result) for result in results}) == thread_count


def test_concurrent_memoize_misses_on_distinct_keys_each_compute_once():
    """
    Many threads calling a `@memoize`-decorated method with distinct arguments at once
    must compute each distinct key's result exactly once, and the cache must end up
    holding exactly one entry per distinct key.
    """
    owner = _MemoizingOwner()
    distinct_keys = range(50)
    callers_per_key = 4

    results, errors = run_concurrently(
        [
            (lambda captured_key=key: owner.compute(captured_key))
            for key in distinct_keys
            for _ in range(callers_per_key)
        ]
    )

    assert errors == []
    assert owner.call_count == len(distinct_keys)
    assert len(results) == len(distinct_keys) * callers_per_key
    assert len(MemoizationCache.existing_for(owner)) == len(distinct_keys)
    assert all(owner.compute(key) == key * key for key in distinct_keys)


# %% distinct keys never wait for each other


def test_distinct_keys_on_one_owner_are_computed_concurrently():
    """
    Two threads computing different keys of the same owner must be able to run their
    computations at the same time, rather than queueing behind one owner-wide lock.
    """
    owner = _RendezvousingOwner(rendezvous=threading.Barrier(2))

    results, errors = run_concurrently(
        [lambda: owner.compute(2), lambda: owner.compute(3)]
    )

    assert errors == []
    assert sorted(results) == [4, 9]


def test_memoized_calls_nested_into_two_owners_in_opposite_orders_do_not_deadlock():
    """
    Two threads descending through the same pair of owners in opposite order must both
    finish: their four cache keys are distinct, so they never wait for each other.
    """
    rendezvous = threading.Barrier(2)
    first_owner = _NestingOwner(rendezvous=rendezvous)
    second_owner = _NestingOwner(rendezvous=rendezvous)
    first_owner.nested_owner = second_owner
    second_owner.nested_owner = first_owner

    results, errors = run_concurrently(
        [lambda: first_owner.outer(1), lambda: second_owner.outer(2)]
    )

    assert errors == []
    assert sorted(results) == [2, 4]
