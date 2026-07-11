"""Thread-safety and reentrancy behaviour of :class:`~krrood.singleton.SingletonMeta`."""

from __future__ import annotations

import threading
import time

from krrood.singleton import SingletonMeta


def test_concurrent_first_construction_does_not_raise_circular_dependency():
    """Several threads racing to construct the same singleton for the first time must not see each
    other's in-progress construction as a circular dependency: exactly one instance is built and every
    thread observes it, with no exception."""

    class ConcurrentlyConstructedSingleton(metaclass=SingletonMeta):
        def __init__(self):
            time.sleep(0.05)

    errors = []
    instances = []

    def build():
        try:
            instances.append(ConcurrentlyConstructedSingleton())
        except RuntimeError as error:
            errors.append(error)

    threads = [threading.Thread(target=build) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ConcurrentlyConstructedSingleton.clear_instance()

    assert errors == []
    assert len(instances) == 8
    assert len(set(map(id, instances))) == 1


def test_recursive_construction_within_the_same_thread_still_raises():
    """A singleton whose own construction recursively constructs itself again, on the same thread, is
    a genuine circular dependency and must still raise."""

    class SelfRecursingSingleton(metaclass=SingletonMeta):
        def __init__(self, recurse: bool = True):
            if recurse:
                SelfRecursingSingleton(recurse=False)

    try:
        raised = False
        try:
            SelfRecursingSingleton()
        except RuntimeError:
            raised = True
        assert raised
    finally:
        SelfRecursingSingleton.clear_instance()
