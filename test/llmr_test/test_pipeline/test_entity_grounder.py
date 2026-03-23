from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmr.pipeline.entity_grounder import (
    EntityGrounder,
    GroundingResult,
    _camel_to_tokens,
    ground_entity,
    resolve_annotation_class,
)
from llmr.workflows.schemas.common import EntityDescriptionSchema


# ── _camel_to_tokens ──────────────────────────────────────────────────────────


class TestCamelToTokensGrounder:
    def test_multi_word(self):
        assert _camel_to_tokens("DrinkingContainer") == "drinking container"

    def test_already_lower(self):
        assert _camel_to_tokens("milk") == "milk"


# ── resolve_annotation_class ─────────────────────────────────────────────────


def _make_ann_cls(name: str, synonyms: set | None = None):
    """Create a fake SemanticAnnotation subclass with a given __name__."""
    cls = type(name, (), {})
    if synonyms:
        cls._synonyms = synonyms
    return cls


class TestResolveAnnotationClass:
    def test_exact_class_name_match(self):
        FakeMilk = _make_ann_cls("Milk")
        with patch(
            "llmr.pipeline.entity_grounder._all_annotation_subclasses",
            return_value=[FakeMilk],
        ):
            result = resolve_annotation_class("Milk")
            assert result is FakeMilk

    def test_case_insensitive_match(self):
        FakeMilk = _make_ann_cls("Milk")
        with patch(
            "llmr.pipeline.entity_grounder._all_annotation_subclasses",
            return_value=[FakeMilk],
        ):
            result = resolve_annotation_class("milk")
            assert result is FakeMilk

    def test_camel_expand_match(self):
        FakeDC = _make_ann_cls("DrinkingContainer")
        with patch(
            "llmr.pipeline.entity_grounder._all_annotation_subclasses",
            return_value=[FakeDC],
        ):
            result = resolve_annotation_class("drinking container")
            assert result is FakeDC

    def test_synonym_match(self):
        FakeCup = _make_ann_cls("Cup", synonyms={"mug", "glass"})
        with patch(
            "llmr.pipeline.entity_grounder._all_annotation_subclasses",
            return_value=[FakeCup],
        ):
            result = resolve_annotation_class("mug")
            assert result is FakeCup

    def test_no_match_returns_none(self):
        FakeMilk = _make_ann_cls("Milk")
        with patch(
            "llmr.pipeline.entity_grounder._all_annotation_subclasses",
            return_value=[FakeMilk],
        ):
            result = resolve_annotation_class("xyz_unknown_type_99")
            assert result is None

    def test_empty_annotation_list_returns_none(self):
        with patch(
            "llmr.pipeline.entity_grounder._all_annotation_subclasses",
            return_value=[],
        ):
            assert resolve_annotation_class("Milk") is None


# ── EntityGrounder._body_name ─────────────────────────────────────────────────


class TestBodyName:
    def test_nested_name_attr(self):
        body = MagicMock()
        body.name.name = "test_object"
        assert EntityGrounder._body_name(body) == "test_object"

    def test_flat_name_attr(self):
        body = MagicMock(spec=["name"])
        body.name = "flat_name"
        assert EntityGrounder._body_name(body) == "flat_name"

    def test_no_name_attr_returns_empty(self):
        body = MagicMock(spec=[])
        # spec=[] means no attributes — getattr returns None
        result = EntityGrounder._body_name(body)
        assert isinstance(result, str)


# ── EntityGrounder._multi_match_warning ───────────────────────────────────────


class TestMultiMatchWarning:
    def test_single_match_returns_none(self):
        body = MagicMock()
        body.name.name = "milk"
        assert EntityGrounder._multi_match_warning([body], "milk") is None

    def test_multiple_matches_returns_string(self):
        b1 = MagicMock()
        b1.name.name = "milk_1"
        b2 = MagicMock()
        b2.name.name = "milk_2"
        warning = EntityGrounder._multi_match_warning([b1, b2], "milk")
        assert warning is not None
        assert "2" in warning
        assert "milk" in warning

    def test_empty_list_returns_none(self):
        assert EntityGrounder._multi_match_warning([], "milk") is None


# ── EntityGrounder._filter_by_attributes ─────────────────────────────────────


class TestFilterByAttributes:
    def _make_body(self, name: str) -> MagicMock:
        body = MagicMock(spec=["name", "_semantic_annotations"])
        body.name = MagicMock()
        body.name.name = name
        body._semantic_annotations = []
        return body

    def test_matching_attribute_keeps_body(self, mock_world):
        grounder = EntityGrounder(mock_world)
        red_cup = self._make_body("red_cup")
        blue_cup = self._make_body("blue_cup")
        result = grounder._filter_by_attributes([red_cup, blue_cup], {"color": "red"})
        assert red_cup in result
        assert blue_cup not in result

    def test_no_match_returns_all_candidates(self, mock_world):
        grounder = EntityGrounder(mock_world)
        b1 = self._make_body("cup1")
        b2 = self._make_body("cup2")
        result = grounder._filter_by_attributes([b1, b2], {"color": "purple"})
        assert b1 in result
        assert b2 in result


# ── EntityGrounder._name_ground ───────────────────────────────────────────────


class TestNameGround:
    def test_name_substring_match(self, mock_world):
        body = MagicMock()
        body.name.name = "milk_0"
        mock_world.bodies = [body]
        grounder = EntityGrounder(mock_world)
        desc = EntityDescriptionSchema(name="milk")
        result = grounder._name_ground(desc)
        assert body in result.bodies

    def test_no_match_returns_empty(self, mock_world):
        body = MagicMock()
        body.name.name = "apple"
        mock_world.bodies = [body]
        grounder = EntityGrounder(mock_world)
        desc = EntityDescriptionSchema(name="milk")
        result = grounder._name_ground(desc)
        assert result.bodies == []

    def test_empty_name_returns_empty(self, mock_world):
        grounder = EntityGrounder(mock_world)
        desc = EntityDescriptionSchema(name="")
        result = grounder._name_ground(desc)
        assert result.bodies == []


# ── EntityGrounder.ground ─────────────────────────────────────────────────────


class TestGrounderGround:
    def test_name_ground_single_result_no_warning(self, mock_world):
        body = MagicMock()
        body.name.name = "milk_0"
        mock_world.bodies = [body]
        grounder = EntityGrounder(mock_world)
        desc = EntityDescriptionSchema(name="milk")
        result = grounder.ground(desc)
        assert body in result.bodies
        assert result.warning is None

    def test_name_ground_two_results_warning_set(self, mock_world):
        b1 = MagicMock()
        b1.name.name = "milk_0"
        b2 = MagicMock()
        b2.name.name = "milk_1"
        mock_world.bodies = [b1, b2]
        grounder = EntityGrounder(mock_world)
        desc = EntityDescriptionSchema(name="milk")
        result = grounder.ground(desc)
        assert len(result.bodies) == 2
        assert result.warning is not None

    def test_nothing_found_warning_set(self, mock_world):
        mock_world.bodies = []
        grounder = EntityGrounder(mock_world)
        desc = EntityDescriptionSchema(name="unknown_xyz")
        result = grounder.ground(desc)
        assert result.bodies == []
        assert result.warning is not None
        assert "unknown_xyz" in result.warning

    def test_semantic_type_tier1_hit(self, mock_world):
        """When annotation grounding finds bodies, tier-1 result is used."""
        body = MagicMock()
        body.name.name = "milk_0"

        ann = MagicMock()
        ann.bodies = [body]

        mock_world.get_semantic_annotations_by_type.return_value = [ann]

        FakeMilk = _make_ann_cls("Milk")
        grounder = EntityGrounder(mock_world)
        desc = EntityDescriptionSchema(name="milk", semantic_type="Milk")

        with patch(
            "llmr.pipeline.entity_grounder._all_annotation_subclasses",
            return_value=[FakeMilk],
        ):
            result = grounder.ground(desc)

        assert body in result.bodies


# ── ground_entity convenience wrapper ────────────────────────────────────────


class TestGroundEntity:
    def test_delegates_to_grounder(self, mock_world):
        body = MagicMock()
        body.name.name = "cup_0"
        mock_world.bodies = [body]
        desc = EntityDescriptionSchema(name="cup")
        result = ground_entity(desc, mock_world)
        assert body in result.bodies
