"""Tests for serialize_world_from_symbol_graph — world state serialization.
"""
from __future__ import annotations

from typing_extensions import Dict, Any
from types import SimpleNamespace

from krrood.symbol_graph.symbol_graph import Symbol

from llmr.world.serializer import (
    WorldSerializationOptions,
    serialize_world_from_symbol_graph,
    body_display_name,
    body_xyz,
    body_bounding_box,
)


class TestSerializeWorld:
    """serialize_world_from_symbol_graph() — world serialization."""

    def test_output_is_string(self, simple_world: Dict[str, Any]) -> None:
        """Output is a string."""
        result = serialize_world_from_symbol_graph()
        assert isinstance(result, str)

    def test_output_contains_world_state_header(self) -> None:
        """Output includes 'World State Summary' header."""
        result = serialize_world_from_symbol_graph()
        assert "World State Summary" in result

    def test_output_contains_semantic_annotations_section(self) -> None:
        """Output includes semantic type section."""
        result = serialize_world_from_symbol_graph()
        assert "Available Semantic Types" in result

    def test_output_contains_grounding_instructions(self) -> None:
        """Output includes explicit grounding instructions for the LLM."""
        result = serialize_world_from_symbol_graph()
        assert "## Grounding Instructions" in result
        assert "Use exact body_name values" in result

    def test_extra_context_appended(self) -> None:
        """extra_context parameter is appended to output."""
        custom_context = "Extra info about the world."
        result = serialize_world_from_symbol_graph(extra_context=custom_context)
        assert custom_context in result

    def test_respects_groundable_type_parameter(self) -> None:
        """serialize respects groundable_type parameter."""
        result = serialize_world_from_symbol_graph(groundable_type=Symbol)
        assert isinstance(result, str)

    def test_handles_empty_extra_context(self) -> None:
        """Handles empty extra_context gracefully."""
        result = serialize_world_from_symbol_graph(extra_context="")
        assert isinstance(result, str)
        assert "World State Summary" in result

    def test_output_length_reasonable(self) -> None:
        """Output has reasonable length (not empty, not huge)."""
        result = serialize_world_from_symbol_graph()
        assert len(result) > 20  # has at least some header

    def test_includes_bodies_or_annotations(self) -> None:
        """Output contains bodies or annotations section."""
        result = serialize_world_from_symbol_graph()
        assert "## Scene Objects" in result
        assert "## Available Semantic Types" in result

    def test_serializes_registered_bodies(self, symbol_world: Dict[str, Any]) -> None:
        """Registered SymbolGraph bodies appear in the scene listing."""
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])

        assert "milk_on_table" in result
        assert "red_cup" in result
        assert "table" in result

    def test_filters_structural_links_from_scene_listing(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Structural link suffixes are excluded from scene objects."""
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])

        assert "base_link" not in result

    def test_filters_semantic_robot_bodies_without_name_suffix(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Robot-owned bodies are excluded via semantic annotations, not name suffixes."""
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])

        assert "robot_base" not in result

    def test_include_structural_keeps_robot_and_suffix_bodies(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """include_structural disables both semantic and fallback structural filters."""
        result = serialize_world_from_symbol_graph(
            symbol_world["body_type"],
            options=WorldSerializationOptions(include_structural=True),
        )

        assert "base_link" in result
        assert "robot_base" in result

    def test_serializes_semantic_annotations(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Annotation instances are grouped by their associated body."""
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])

        assert "## Available Semantic Types" in result
        assert "- MilkAnnotation: milk_on_table" in result
        assert "| milk_on_table | WorldBody | MilkAnnotation | table |" in result

    def test_serializes_parent_context(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """Parent links are rendered as spatial context."""
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])

        assert "## Spatial Context" in result
        assert "- milk_on_table is under/within parent table" in result

    def test_options_truncate_objects_deterministically(
        self, symbol_world: Dict[str, Any]
    ) -> None:
        """max_objects limits scene rows and reports truncation."""
        result = serialize_world_from_symbol_graph(
            symbol_world["body_type"],
            options=WorldSerializationOptions(max_objects=2),
        )

        assert "Objects: 6 visible (showing 2)" in result
        assert "Truncated 4 additional object(s)." in result

    def test_extra_context_has_own_section(self) -> None:
        """extra_context is appended under an explicit section."""
        result = serialize_world_from_symbol_graph(extra_context="Prefer the left cup.")

        assert "## Extra Context" in result
        assert "Prefer the left cup." in result


class TestBodyDisplayNameHelper:
    """body_display_name() duck-typed accessor."""

    def test_extracts_simple_name(self) -> None:
        """Extracts .name when it's a string."""
        body = SimpleNamespace(name="test_body")
        assert body_display_name(body) == "test_body"

    def test_unwraps_nested_name(self) -> None:
        """Unwraps nested .name.name (PrefixedName chain)."""
        inner = SimpleNamespace(name="real_name")
        body = SimpleNamespace(name=inner)
        assert body_display_name(body) == "real_name"

    def test_handles_none_name(self) -> None:
        """Returns empty string when .name is None."""
        body = SimpleNamespace(name=None)
        assert body_display_name(body) == ""

    def test_handles_missing_name(self) -> None:
        """Returns empty string when .name attribute missing."""
        body = SimpleNamespace()
        assert body_display_name(body) == ""


class TestBodyXyzHelper:
    """body_xyz() duck-typed accessor."""

    def test_extracts_xyz_coordinates(self) -> None:
        """Extracts (x, y, z) from .global_pose.to_position()."""
        point = SimpleNamespace(x=1.5, y=2.5, z=3.5)
        pose = SimpleNamespace(to_position=lambda: point)
        body = SimpleNamespace(global_pose=pose)
        result = body_xyz(body)
        assert result == (1.5, 2.5, 3.5)

    def test_returns_none_when_no_pose(self) -> None:
        """Returns None when .global_pose missing."""
        body = SimpleNamespace()
        assert body_xyz(body) is None

    def test_handles_pose_access_error(self) -> None:
        """Returns None if pose access fails."""
        pose = SimpleNamespace()  # missing to_position
        body = SimpleNamespace(global_pose=pose)
        assert body_xyz(body) is None


class TestBodyBoundingBoxHelper:
    """body_bounding_box() duck-typed accessor."""

    def test_returns_none_when_no_collision(self) -> None:
        """Returns None when collision unavailable."""
        body = SimpleNamespace()
        assert body_bounding_box(body) is None

    def test_handles_collision_access_error(self) -> None:
        """Returns None if collision access fails."""
        body = SimpleNamespace(collision=None)  # None, no method chain
        assert body_bounding_box(body) is None


class TestSerializeWorldDeterminism:
    """Deterministic and bounded serializer output."""

    def test_serialize_world_output_is_deterministic(self, symbol_world: Dict[str, Any]) -> None:
        """Two calls with the same world state produce identical output."""
        result1 = serialize_world_from_symbol_graph(symbol_world["body_type"])
        result2 = serialize_world_from_symbol_graph(symbol_world["body_type"])
        assert result1 == result2

    def test_max_objects_truncates_scene_table(self, symbol_world: Dict[str, Any]) -> None:
        """max_objects=1 limits the scene table to at most one data row."""
        result = serialize_world_from_symbol_graph(
            symbol_world["body_type"],
            options=WorldSerializationOptions(max_objects=1),
        )
        # The header row is always present; only 1 data row should appear
        # Count '| ' occurrences in Scene Objects section as a proxy for row count
        lines = result.splitlines()
        scene_rows = [
            line for line in lines
            if line.startswith("| ") and "body_name" not in line and "---" not in line
        ]
        assert len(scene_rows) <= 1
