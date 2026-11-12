from __future__ import annotations

import json

import pytest

from uuid import uuid4

from krrood.adapters.exceptions import UntrackedObjectError
from krrood.adapters.json_serializer import to_json
from krrood.patterns.identifier_and_name import HasID, HasName, NameT
from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin

from ..dataset.identified_objects import (
    Landmark,
    LandmarkReference,
    LandmarkTracker,
    UnnamedMarker,
)

# %% having an identifier and a name


@pytest.mark.parametrize("trait", [HasID, HasName])
def test_an_identifier_and_a_name_are_mixins(trait):
    assert issubclass(trait, Mixin)


@pytest.mark.parametrize("trait", [HasID, HasName])
def test_an_identifier_and_a_name_are_interfaces(trait):
    assert Interface.is_interface(class_=trait)


def test_an_identifier_does_not_require_a_name():
    assert not issubclass(HasID, HasName)


def test_a_named_object_binds_the_kind_of_its_name():
    assert Landmark.get_type_of_generic_parameter(parameter=NameT) is str


# %% writing and resolving references


@pytest.fixture
def lighthouse() -> Landmark:
    return Landmark(name="lighthouse")


def test_a_reference_carries_the_identifier_and_name_under_its_subject(lighthouse):
    data = {}

    LandmarkReference(subject="harbour").write(data=data, identified_object=lighthouse)

    assert data == {
        "harbour_id": to_json(lighthouse.id),
        "harbour_name": lighthouse.name,
    }


def test_a_reference_to_an_unnamed_object_carries_only_its_identifier():
    marker = UnnamedMarker()
    data = {}

    LandmarkReference(subject="harbour").write(data=data, identified_object=marker)

    assert data == {"harbour_id": to_json(marker.id)}


def test_a_written_reference_resolves_to_the_object_it_names(lighthouse):
    reference = LandmarkReference(subject="harbour")
    data = {}
    reference.write(data=data, identified_object=lighthouse)
    tracker = LandmarkTracker()
    tracker.add(key=lighthouse.id, tracked_object=lighthouse)

    resolved = reference.resolve(
        data=json.loads(json.dumps(data)), **tracker.create_kwargs()
    )

    assert resolved is lighthouse


def test_an_object_missing_from_the_document_is_reported_with_its_name():
    with pytest.raises(UntrackedObjectError) as raised:
        LandmarkTracker().get(key=uuid4(), name="lighthouse")

    assert raised.value.name == "lighthouse"
