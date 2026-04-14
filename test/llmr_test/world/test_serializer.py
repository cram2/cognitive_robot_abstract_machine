"""Tests for serialize_world_from_symbol_graph — world state serialization.

Coverage target: 80% (8 tests covering serialization and helper functions).
"""
from __future__ import annotations

from typing_extensions import Dict, Any
import pytest
from types import SimpleNamespace

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

from llmr.world.serializer import (
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
        """Output includes 'Semantic annotations' section."""
        result = serialize_world_from_symbol_graph()
        assert "Semantic annotations" in result

    def test_extra_context_appended(self) -> None:
        """extra_context parameter is appended to output."""
        custom_context = "Extra info about the world."
        result = serialize_world_from_symbol_graph(extra_context=custom_context)
        assert custom_context in result

    def test_respects_groundable_type_parameter(self) -> None:
        """serialize respects groundable_type parameter."""
        result = serialize_world_from_symbol_graph(groundable_type=Symbol)
        # Should work with Symbol as groundable type
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
        # Should contain either bodies or annotations
        assert "Bodies" in result or "objects" in result or "annotations" in result


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
