import pytest

from krrood.adapters.exceptions import UntrackedObjectError
from krrood.adapters.json_serializer import from_json, to_json

from ..dataset.shared_references import (
    ElementReferencedTwice,
    KeyedElement,
    KeyedElementTracker,
    OtherKeyedElementTracker,
)


def test_element_referenced_twice_deserializes_to_one_instance():
    element = KeyedElement(key="shared")
    data = to_json(ElementReferencedTwice(first=element, second=element))

    result = from_json(data, **KeyedElementTracker().create_kwargs())

    assert result.first is result.second


def test_from_kwargs_returns_the_tracker_already_in_the_kwargs():
    tracker = KeyedElementTracker()

    assert KeyedElementTracker.from_kwargs(tracker.create_kwargs()) is tracker


def test_trackers_of_different_types_keep_separate_kwargs():
    tracker = KeyedElementTracker()
    kwargs = tracker.create_kwargs()

    assert (
        type(OtherKeyedElementTracker.from_kwargs(kwargs)) is OtherKeyedElementTracker
    )
    assert KeyedElementTracker.from_kwargs(kwargs) is tracker


def test_has_is_true_only_after_add():
    tracker = KeyedElementTracker()
    element = KeyedElement(key="element")

    assert not tracker.has(element.key)
    tracker.add(element.key, element)
    assert tracker.has(element.key)


def test_get_returns_the_added_object():
    tracker = KeyedElementTracker()
    element = KeyedElement(key="element")
    tracker.add(element.key, element)

    assert tracker.get(element.key) is element


def test_get_of_an_untracked_key_raises():
    with pytest.raises(UntrackedObjectError):
        KeyedElementTracker().get("untracked")
