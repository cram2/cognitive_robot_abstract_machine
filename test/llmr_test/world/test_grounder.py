"""Tests for EntityGrounder — entity description to Symbol resolution.
"""
from __future__ import annotations

from typing_extensions import Dict, Any
from types import SimpleNamespace

from krrood.symbol_graph.symbol_graph import Symbol

from llmr.schemas.entities import EntityDescriptionSchema
from llmr.world.grounder import (
    EntityGrounder,
    GroundingResult,
    ground_expected_entity,
    grounder_can_return_type,
    resolve_symbol_class,
    _camel_to_tokens,
)
from llmr.world.serializer import body_display_name, body_xyz, body_bounding_box
from .conftest import WorldBody


class Manipulator(Symbol):
    def __init__(self, name: str = "manipulator"):
        self.name = name


class PrefixedNameLike:
    def __init__(self, name: str):
        self.name = name


class TestEntityGrounder:
    """EntityGrounder.ground() — entity resolution logic."""

    def test_ground_returns_grounding_result(self, simple_world: Dict[str, Any]) -> None:
        """ground() returns a GroundingResult object."""
        grounder = EntityGrounder()
        description = EntityDescriptionSchema(name="test")
        result = grounder.ground(description)
        assert isinstance(result, GroundingResult)
        assert hasattr(result, "bodies")
        assert hasattr(result, "warning")

    def test_ground_empty_description_returns_result(self) -> None:
        """ground() with empty description returns a GroundingResult."""
        grounder = EntityGrounder()
        description = EntityDescriptionSchema(name="")
        result = grounder.ground(description)
        assert isinstance(result, GroundingResult)

    def test_ground_nonexistent_name_has_warning(self) -> None:
        """ground() with nonexistent name produces a warning."""
        grounder = EntityGrounder()
        description = EntityDescriptionSchema(name="nonexistent_xyz_12345")
        result = grounder.ground(description)
        assert result.bodies == []
        assert result.warning is not None

    def test_ground_accepts_semantic_type(self) -> None:
        """ground() accepts semantic_type in description."""
        grounder = EntityGrounder()
        description = EntityDescriptionSchema(
            name="test", semantic_type="SomeType"
        )
        result = grounder.ground(description)
        assert isinstance(result, GroundingResult)

    def test_name_grounding_returns_exact_matching_symbol(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Name grounding returns matching SymbolGraph instances."""
        grounder = EntityGrounder(groundable_type=symbol_world["body_type"])
        result = grounder.ground(EntityDescriptionSchema(name="red_cup"))

        assert result.bodies == [symbol_world["red_cup"]]
        assert result.warning is None

    def test_semantic_type_grounding_returns_annotated_body(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Semantic type grounding follows annotation .bodies links."""
        result = EntityGrounder(groundable_type=symbol_world["body_type"]).ground(
            EntityDescriptionSchema(
                name="milk",
                semantic_type=symbol_world["annotation_type"].__name__,
            )
        )

        assert result.bodies == [symbol_world["milk_on_table"]]

    def test_multiple_name_matches_return_warning(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Ambiguous grounding keeps candidates and reports the ambiguity."""
        result = EntityGrounder(groundable_type=symbol_world["body_type"]).ground(
            EntityDescriptionSchema(name="cup")
        )

        assert result.bodies == [symbol_world["red_cup"], symbol_world["blue_cup"]]
        assert result.warning is not None
        assert "2 candidates" in result.warning

    def test_attribute_filter_narrows_candidates(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Attribute values narrow ambiguous name matches."""
        result = EntityGrounder(groundable_type=symbol_world["body_type"]).ground(
            EntityDescriptionSchema(name="cup", attributes={"color": "blue"})
        )

        assert result.bodies == [symbol_world["blue_cup"]]

    def test_spatial_context_filters_by_parent_subtree(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Spatial context can narrow candidates by parent subtree."""
        result = EntityGrounder(groundable_type=symbol_world["body_type"]).ground(
            EntityDescriptionSchema(name="milk", spatial_context="on the table")
        )

        assert result.bodies == [symbol_world["milk_on_table"]]

    def test_ground_accepts_spatial_context(self) -> None:
        """ground() accepts spatial_context in description."""
        grounder = EntityGrounder()
        description = EntityDescriptionSchema(
            name="test", spatial_context="on the table"
        )
        result = grounder.ground(description)
        assert isinstance(result, GroundingResult)

    def test_grounder_with_specific_groundable_type(self) -> None:
        """EntityGrounder accepts groundable_type parameter."""
        grounder = EntityGrounder(groundable_type=Symbol)
        description = EntityDescriptionSchema(name="test")
        result = grounder.ground(description)
        assert isinstance(result, GroundingResult)

    def test_ground_result_bodies_is_list(self) -> None:
        """GroundingResult.bodies is always a list."""
        grounder = EntityGrounder()
        description = EntityDescriptionSchema(name="test")
        result = grounder.ground(description)
        assert isinstance(result.bodies, list)

    def test_ground_result_warning_optional(self) -> None:
        """GroundingResult.warning can be None or string."""
        grounder = EntityGrounder()
        description = EntityDescriptionSchema(name="test")
        result = grounder.ground(description)
        assert result.warning is not None
        assert isinstance(result.warning, str)


class TestExpectedTypeGrounding:
    """Expected-type grounding helpers used when action fields need a Symbol subtype."""

    def test_grounder_can_return_type_checks_search_scope(self) -> None:
        assert grounder_can_return_type(
            EntityGrounder(groundable_type=Symbol),
            Manipulator,
        )
        assert not grounder_can_return_type(
            EntityGrounder(groundable_type=WorldBody),
            Manipulator,
        )

    def test_ground_expected_entity_handles_prefixed_arm_names(self) -> None:
        alpha = SimpleNamespace(name=PrefixedNameLike("alpha_manipulator"))
        beta = SimpleNamespace(name=PrefixedNameLike("beta_manipulator"))
        resolved_params = {"arm": SimpleNamespace(name=PrefixedNameLike("alpha"))}

        class FakeSymbolGraph:
            def get_instances_of_type(self, raw_type):
                assert raw_type is Manipulator
                return [beta, alpha]

        result = ground_expected_entity(
            Manipulator,
            EntityDescriptionSchema(name="manipulator"),
            resolved_params,
            symbol_graph=FakeSymbolGraph(),
        )

        assert result is alpha


class TestResolveSymbolClass:
    """resolve_symbol_class() — semantic type to Symbol subclass resolution."""

    def test_exact_class_name_match(self) -> None:
        """Exact class name match resolves a concrete Symbol subclass."""
        cls = resolve_symbol_class("WorldBody")
        assert cls is WorldBody

    def test_case_insensitive_match(self) -> None:
        """Class name matching is case-insensitive."""
        cls = resolve_symbol_class("worldbody")
        assert cls is WorldBody

    def test_returns_none_for_unknown_type(self) -> None:
        """Returns None for unknown semantic type."""
        cls = resolve_symbol_class("UnknownType12345XYZ")
        assert cls is None


class TestCamelToTokens:
    """_camel_to_tokens() — CamelCase to lowercase tokens."""

    def test_splits_camel_case(self) -> None:
        """CamelCase → lowercase tokens."""
        assert _camel_to_tokens("PickUpAction") == "pick up action"

    def test_handles_single_word(self) -> None:
        """Single word remains lowercase."""
        assert _camel_to_tokens("Body") == "body"

    def test_handles_acronyms(self) -> None:
        """Acronyms are handled (lowercased)."""
        assert _camel_to_tokens("XMLParser") == "xmlparser"

    def test_handles_underscores(self) -> None:
        """Underscores are preserved during conversion."""
        tokens = _camel_to_tokens("My_CamelCase")
        assert tokens == "my_camel case"


class TestBodyHelpers:
    """body_display_name, body_xyz, body_bounding_box — duck-typed accessors."""

    def test_body_display_name_unwraps_nested_name(self) -> None:
        """body_display_name handles nested .name.name."""
        # Simulate a body with nested name structure (PrefixedName chain)
        inner_name = SimpleNamespace(name="actual_name")
        body = SimpleNamespace(name=inner_name)
        result = body_display_name(body)
        assert result == "actual_name"

    def test_body_display_name_returns_str_name(self) -> None:
        """body_display_name returns string name directly."""
        body = SimpleNamespace(name="simple_name")
        result = body_display_name(body)
        assert result == "simple_name"

    def test_body_display_name_returns_empty_for_none(self) -> None:
        """body_display_name returns empty string when name is None."""
        body = SimpleNamespace(name=None)
        result = body_display_name(body)
        assert result == ""

    def test_body_display_name_handles_missing_name(self) -> None:
        """body_display_name handles objects without .name attribute."""
        body = SimpleNamespace()  # no .name
        result = body_display_name(body)
        assert result == ""

    def test_body_xyz_returns_tuple_when_pose_present(self) -> None:
        """body_xyz returns (x, y, z) tuple when pose available."""
        # Mock a pose-like object with to_position()
        mock_point = SimpleNamespace(x=1.0, y=2.0, z=3.0)
        mock_pose = SimpleNamespace(to_position=lambda: mock_point)
        body = SimpleNamespace(global_pose=mock_pose)
        result = body_xyz(body)
        assert result == (1.0, 2.0, 3.0)

    def test_body_xyz_returns_none_when_pose_missing(self) -> None:
        """body_xyz returns None when pose unavailable."""
        body = SimpleNamespace()  # no global_pose
        result = body_xyz(body)
        assert result is None

    def test_body_bounding_box_returns_none_when_unavailable(self) -> None:
        """body_bounding_box returns None when collision unavailable."""
        body = SimpleNamespace()  # no collision
        result = body_bounding_box(body)
        assert result is None
