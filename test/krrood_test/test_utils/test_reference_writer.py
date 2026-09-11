import json

import pytest

from krrood.adapters.json_serializer import ReferenceWriter, from_json, to_json

from ..dataset.referenced_objects import (
    ElementHolder,
    ElementHolderWithOwnSerialization,
    ElementInEveryPosition,
    ReaderOwnedElement,
    ReaderOwnedElementReferenceWriter,
    ReaderOwnedElementTracker,
)


@pytest.fixture
def element() -> ReaderOwnedElement:
    return ReaderOwnedElement(key="landmark")


def create_element_in_every_position(
    element: ReaderOwnedElement,
) -> ElementInEveryPosition:
    return ElementInEveryPosition(
        direct=element,
        nested=ElementHolder(element=element),
        in_own_serialization=ElementHolderWithOwnSerialization(element=element),
        in_list=[element],
        in_dict={"entry": element},
    )


def test_a_referenced_object_is_written_as_its_reference(element):
    writer = ReaderOwnedElementReferenceWriter()

    data = to_json(create_element_in_every_position(element), **writer.create_kwargs())

    assert data["direct"] == writer.write_reference(element)


def test_every_position_refers_to_the_object_the_reader_has(element):
    """
    Written in full, an element would be read back as a new instance; a reference is
    read back as the reader's own element.
    """
    writer = ReaderOwnedElementReferenceWriter()
    data = json.loads(
        json.dumps(
            to_json(create_element_in_every_position(element), **writer.create_kwargs())
        )
    )
    tracker = ReaderOwnedElementTracker()
    tracker.add(element.key, element)

    result = from_json(data, **tracker.create_kwargs())

    assert result.direct is element
    assert result.nested.element is element
    assert result.in_own_serialization.element is element
    assert result.in_list[0] is element
    assert result.in_dict["entry"] is element


def test_without_a_reference_writer_an_object_is_written_in_full(element):
    data = to_json(create_element_in_every_position(element))

    assert data["direct"] == element.to_json()


def test_a_reference_writer_is_found_only_for_its_referenced_type(element):
    writer = ReaderOwnedElementReferenceWriter()
    kwargs = writer.create_kwargs()

    assert ReferenceWriter.find_for(element, kwargs) is writer
    assert ReferenceWriter.find_for(ElementHolder(element=element), kwargs) is None
