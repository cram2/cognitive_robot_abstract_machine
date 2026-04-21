"""Tests for :mod:`llmr.resolution.grounder` — two-tier EntityGrounder (annotation → name)."""

from __future__ import annotations

from typing_extensions import Any, Dict

import pytest
from krrood.symbol_graph.symbol_graph import SymbolGraph

from llmr.resolution.grounder import EntityGrounder, GroundingResult
from llmr.schemas import EntityDescriptionSchema

from ._fixtures.symbols import (
    Manipulator,
    MilkAnnotation,
    ParallelGripperLike,
    WorldBody,
)
from ._fixtures.worlds import robot_world, symbol_world  # noqa: F401


class TestGroundingResultDefaults:
    """:class:`GroundingResult` — simple dataclass invariants."""

    def test_defaults_to_empty_bodies_and_no_warning(self) -> None:
        result = GroundingResult()
        assert result.bodies == []
        assert result.warning is None


class TestAnnotationGrounding:
    """Tier 1 — ``semantic_type`` resolves to a Symbol subclass and its ``.bodies``."""

    def test_synonym_resolves_to_annotation_bodies(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """``semantic_type='milk'`` hits ``MilkAnnotation._synonyms`` then yields its body."""
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="milk_on_table", semantic_type="milk")
        result = grounder.ground(desc)
        assert symbol_world["milk_on_table"] in result.bodies

    def test_unknown_semantic_type_falls_back_to_name(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """When ``semantic_type`` is unresolvable, Tier 2 name search takes over."""
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="table", semantic_type="UnknownThing")
        result = grounder.ground(desc)
        assert symbol_world["table"] in result.bodies

    def test_annotation_without_bodies_returns_annotation_itself(
        self, robot_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """Manipulator instances lack ``.bodies`` → the annotation is itself returned."""
        grounder = EntityGrounder(groundable_type=Manipulator)
        desc = EntityDescriptionSchema(name="left", semantic_type="Manipulator")
        result = grounder.ground(desc, expected_type=Manipulator)
        assert robot_world["left"] in result.bodies

    def test_expected_type_returns_subclass_instance_directly(
        self, robot_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """``ParallelGripperLike`` is a subclass of :class:`Manipulator`."""
        grounder = EntityGrounder(groundable_type=Manipulator)
        desc = EntityDescriptionSchema(
            name="right", semantic_type="ParallelGripperLike"
        )
        result = grounder.ground(desc, expected_type=Manipulator)
        assert robot_world["right"] in result.bodies

    def test_expected_type_without_resolvable_class_uses_expected(
        self, robot_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """If ``semantic_type`` cannot be resolved, ``expected_type`` seeds Tier 1."""
        grounder = EntityGrounder(groundable_type=Manipulator)
        desc = EntityDescriptionSchema(name="left", semantic_type="NotAClass")
        result = grounder.ground(desc, expected_type=Manipulator)
        assert robot_world["left"] in result.bodies

    def test_name_filter_narrows_annotation_candidates(
        self, robot_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """When multiple annotations are found, the ``name`` substring narrows them."""
        grounder = EntityGrounder(groundable_type=Manipulator)
        desc = EntityDescriptionSchema(name="right_hand", semantic_type="Manipulator")
        result = grounder.ground(desc, expected_type=Manipulator)
        assert result.bodies == [robot_world["right"]]


class TestNameGrounding:
    """Tier 2 — substring match on ``description.name`` across ``groundable_type`` instances."""

    def test_exact_name_match(self, symbol_world: Dict[str, Any]) -> None:  # noqa: F811
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="counter")
        result = grounder.ground(desc)
        assert symbol_world["counter"] in result.bodies

    def test_substring_match_case_insensitive(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="CUP")
        result = grounder.ground(desc)
        display_names = {b.name for b in result.bodies}
        assert {"red_cup", "blue_cup"}.issubset(display_names)

    def test_missing_name_returns_warning(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="nonexistent")
        result = grounder.ground(desc)
        assert result.bodies == []
        assert result.warning is not None
        assert "nonexistent" in result.warning

    def test_empty_name_returns_empty_result(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """An empty ``name`` skips Tier 2 and yields the standard not-found warning."""
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="")
        result = grounder.ground(desc)
        assert result.bodies == []
        assert result.warning is not None


class TestAttributeRefinement:
    """:meth:`_filter_by_attributes` narrows multiple candidates by attribute match."""

    def test_attribute_narrows_candidates(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="cup", attributes={"color": "red"})
        result = grounder.ground(desc)
        assert result.bodies == [symbol_world["red_cup"]]

    def test_attribute_on_single_candidate_skipped(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """With a single candidate the attribute filter is not applied."""
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="table", attributes={"color": "green"})
        result = grounder.ground(desc)
        assert symbol_world["table"] in result.bodies

    def test_attribute_no_match_returns_original_candidates(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """If no candidate matches the attribute, keep the original list."""
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="cup", attributes={"color": "purple"})
        result = grounder.ground(desc)
        # Both cups still come back because no filter hit.
        display_names = {b.name for b in result.bodies}
        assert {"red_cup", "blue_cup"}.issubset(display_names)


class TestMultiMatchWarning:
    """Warnings surface whenever grounding resolves to more than one instance."""

    def test_multi_match_produces_warning(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="cup")
        result = grounder.ground(desc)
        assert result.warning is not None
        assert "cup" in result.warning

    def test_single_match_has_no_warning(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        grounder = EntityGrounder(groundable_type=WorldBody)
        desc = EntityDescriptionSchema(name="blue_cup")
        result = grounder.ground(desc)
        assert result.bodies == [symbol_world["blue_cup"]]
        assert result.warning is None


class TestExplicitSymbolGraph:
    """An explicit ``symbol_graph`` bypasses the singleton."""

    def test_empty_graph_returns_no_bodies(self) -> None:
        graph = SymbolGraph()
        grounder = EntityGrounder(groundable_type=WorldBody, symbol_graph=graph)
        desc = EntityDescriptionSchema(name="milk")
        result = grounder.ground(desc)
        assert result.bodies == []
        assert result.warning is not None
