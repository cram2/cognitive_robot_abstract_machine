from __future__ import annotations

import json

import pytest

from krrood.adapters.json_serializer import to_json
from krrood.patterns.identifier_and_name import HasID, HasName
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
    assert Interface.is_interface(trait)


def test_an_identifier_does_not_require_a_name():
    assert not issubclass(HasID, HasName)


# %% writing and resolving references


@pytest.fixture
def lighthouse() -> Landmark:
    return Landmark(name="lighthouse")


def test_a_reference_carries_the_identifier_and_name_under_its_subject(lighthouse):
    data = {}

    LandmarkReference("harbour").write(data, lighthouse)

    assert data == {
        "harbour_id": to_json(lighthouse.id),
        "harbour_name": lighthouse.name,
    }


def test_a_reference_to_an_unnamed_object_carries_only_its_identifier():
    marker = UnnamedMarker()
    data = {}

    LandmarkReference("harbour").write(data, marker)

    assert data == {"harbour_id": to_json(marker.id)}


def test_a_written_reference_resolves_to_the_object_it_names(lighthouse):
    reference = LandmarkReference("harbour")
    data = {}
    reference.write(data, lighthouse)
    tracker = LandmarkTracker()
    tracker.add(lighthouse.id, lighthouse)

    resolved = reference.resolve(
        json.loads(json.dumps(data)), **tracker.create_kwargs()
    )

    assert resolved is lighthouse
