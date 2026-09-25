"""
Tests for what a query box is told it may name.
"""

from __future__ import annotations

import pytest

from cramera.knowledge.entities import BenchObject
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.query_runner import EqlQueryRunner
from cramera.knowledge.query_vocabulary import (
    UnknownVocabularyName,
    VocabularyKind,
)
from cramera.knowledge.workspace_classes import WorkspaceClassIndex

from .test_workspace_classes import scanned_class


@pytest.fixture()
def vocabulary():
    """
    The vocabulary of a runner with one domain and one workspace class.
    """
    index = WorkspaceClassIndex.of_scanned_classes(
        [
            scanned_class(
                "Body",
                "semantic_digital_twin",
                "semantic_digital_twin.src.semantic_digital_twin"
                ".world_description.world_entity",
            ),
            scanned_class("Filter", "coraplex", "coraplex.src.coraplex.filter"),
            scanned_class("Filter", "robokudo", "robokudo.src.robokudo.filter"),
        ]
    )
    return EqlQueryRunner(
        domains=[QueryDomain("scene_object", BenchObject, [])],
        extra_names={"objects": [], "BenchObject": BenchObject},
        class_index=index,
    ).vocabulary()


def entry_named(vocabulary, name: str):
    """
    The one vocabulary entry of this name.

    :param vocabulary: The vocabulary to look in.
    :param name: Name of the entry.
    """
    return next(entry for entry in vocabulary.entries() if entry.name == name)


# %% what a query may name


class TestVocabularyEntries:
    """
    Which names a query box offers, and what it says about each.
    """

    def test_a_ready_made_variable_carries_the_type_its_members_come_from(
        self, vocabulary
    ):
        entry = entry_named(vocabulary, "scene_object")

        assert entry.kind is VocabularyKind.VARIABLE
        assert entry.type_name == "BenchObject"

    def test_an_entity_type_is_offered_with_its_module(self, vocabulary):
        entry = entry_named(vocabulary, "BenchObject")

        assert entry.kind is VocabularyKind.ENTITY_TYPE
        assert entry.module == BenchObject.__module__
        assert entry.type_name == "BenchObject"

    def test_an_eql_factory_is_offered(self, vocabulary):
        entry = entry_named(vocabulary, "entity")

        assert entry.kind is VocabularyKind.FACTORY

    def test_a_value_put_in_reach_by_hand_is_offered_as_a_value(self, vocabulary):
        entry = entry_named(vocabulary, "objects")

        assert entry.kind is VocabularyKind.VALUE

    def test_a_workspace_class_is_offered_with_its_module_and_summary(self, vocabulary):
        entry = entry_named(vocabulary, "Body")

        assert entry.kind is VocabularyKind.CLASS
        assert entry.module == "semantic_digital_twin.world_description.world_entity"
        assert entry.detail == "Body of semantic_digital_twin."

    def test_a_name_several_modules_define_is_offered_once_by_its_winner(
        self, vocabulary
    ):
        offered = [entry for entry in vocabulary.entries() if entry.name == "Filter"]

        assert len(offered) == 1
        assert offered[0].module == "coraplex.filter"
        assert offered[0].further_modules == 1

    def test_a_class_defined_once_says_so(self, vocabulary):
        assert entry_named(vocabulary, "Body").further_modules == 0

    def test_every_offered_name_is_offered_only_once(self, vocabulary):
        names = [entry.name for entry in vocabulary.entries()]

        assert len(names) == len(set(names))


# %% what follows a dot


class TestVocabularyMembers:
    """
    Which members a type offers once a query writes its dot.
    """

    def test_a_dataclass_field_is_a_member_of_its_type(self, vocabulary):
        members = {
            member.name: member for member in vocabulary.members_of("BenchObject")
        }

        assert set(members) >= {"name", "kind", "label", "height_metres", "position"}
        assert members["name"].kind is VocabularyKind.FIELD

    def test_a_method_of_a_workspace_class_is_a_member(self, vocabulary):
        members = {member.name: member for member in vocabulary.members_of("Body")}

        assert members["has_collision"].kind is VocabularyKind.METHOD

    def test_every_member_is_named_once(self, vocabulary):
        members = [member.name for member in vocabulary.members_of("BenchObject")]

        assert len(members) == len(set(members))

    def test_no_member_is_a_dunder(self, vocabulary):
        members = vocabulary.members_of("BenchObject")

        assert not [member for member in members if member.name.startswith("_")]

    def test_a_property_of_a_workspace_class_is_a_member(self, vocabulary):
        members = {member.name: member for member in vocabulary.members_of("Body")}

        assert members["global_pose"].kind is VocabularyKind.PROPERTY

    def test_a_members_request_for_an_unknown_type_is_refused(self, vocabulary):
        with pytest.raises(UnknownVocabularyName):
            vocabulary.members_of("NoSuchType")


# %% the payload the panel reads


class TestVocabularyPayload:
    """
    The JSON shape the query box is served.
    """

    def test_the_payload_carries_every_entry_with_its_kind_as_a_string(
        self, vocabulary
    ):
        payload = vocabulary.to_payload()

        assert payload["ok"] is True
        assert len(payload["entries"]) == len(vocabulary.entries())
        assert payload["entries"][0]["kind"] in [kind.value for kind in VocabularyKind]

    def test_the_payload_of_one_type_names_the_type_it_describes(self, vocabulary):
        payload = vocabulary.members_payload("BenchObject")

        assert payload["ok"] is True
        assert payload["name"] == "BenchObject"
        assert [member["name"] for member in payload["members"]] == [
            member.name for member in vocabulary.members_of("BenchObject")
        ]
