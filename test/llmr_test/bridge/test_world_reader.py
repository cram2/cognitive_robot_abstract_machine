"""Tests for :mod:`llmr.bridge.world_reader` — SymbolGraph gateway and duck-typed body helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing_extensions import Any, Dict

import pytest
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.bridge.world_reader import (
    WorldSerializationOptions,
    _camel_to_tokens,
    body_bounding_box,
    body_display_name,
    body_xyz,
    get_instances,
    resolve_symbol_class,
    serialize_world_from_symbol_graph,
)

from .._fixtures.symbols import (
    FakeRobotAnnotation,
    Manipulator,
    MilkAnnotation,
    ParallelGripperLike,
    WorldBody,
)
from .._fixtures.worlds import robot_world, simple_world, symbol_world  # noqa: F401


class TestBodyDisplayName:
    """:func:`body_display_name` — safe wrapper over ``.name`` with PrefixedName unwrapping."""

    def test_plain_string_name(self) -> None:
        body = SimpleNamespace(name="milk")
        assert body_display_name(body) == "milk"

    def test_prefixed_name_chain(self) -> None:
        """``body.name.name`` is collapsed to its leaf string."""
        inner = SimpleNamespace(name="leaf")
        body = SimpleNamespace(name=inner)
        assert body_display_name(body) == "leaf"

    def test_none_name_returns_empty(self) -> None:
        assert body_display_name(SimpleNamespace(name=None)) == ""

    def test_missing_name_returns_empty(self) -> None:
        assert body_display_name(SimpleNamespace()) == ""


class TestBodyXYZ:
    """:func:`body_xyz` — returns ``(x, y, z)`` or ``None`` on failure."""

    def test_successful_extraction(self) -> None:
        pose = SimpleNamespace(to_position=lambda: SimpleNamespace(x=1.0, y=2.0, z=3.0))
        body = SimpleNamespace(global_pose=pose)
        assert body_xyz(body) == (1.0, 2.0, 3.0)

    def test_missing_pose_returns_none(self) -> None:
        assert body_xyz(SimpleNamespace()) is None

    def test_raising_pose_returns_none(self) -> None:
        pose = SimpleNamespace()  # no to_position
        assert body_xyz(SimpleNamespace(global_pose=pose)) is None


class TestBodyBoundingBox:
    """:func:`body_bounding_box` — returns dims tuple or ``None`` on failure."""

    def test_missing_collision_returns_none(self) -> None:
        assert body_bounding_box(SimpleNamespace()) is None

    def test_none_collision_returns_none(self) -> None:
        assert body_bounding_box(SimpleNamespace(collision=None)) is None

    def test_successful_extraction(self) -> None:
        """A duck-typed collision chain returns the (d, w, h) tuple."""
        bb = SimpleNamespace(dimensions=(0.5, 1.0, 1.5))
        bbox = SimpleNamespace(bounding_box=lambda: bb)
        coll = SimpleNamespace(as_bounding_box_collection_in_frame=lambda _frame: bbox)
        body = SimpleNamespace(collision=coll)
        assert body_bounding_box(body) == (0.5, 1.0, 1.5)


class TestCamelToTokens:
    """:func:`_camel_to_tokens` drives resolve_symbol_class token matching."""

    @pytest.mark.parametrize(
        "cls_name, tokens",
        [
            ("PickUpAction", "pick up action"),
            ("Body", "body"),
            ("XMLParser", "xmlparser"),  # acronyms stay together
            ("My_CamelCase", "my_camel case"),
        ],
    )
    def test_conversion(self, cls_name: str, tokens: str) -> None:
        assert _camel_to_tokens(cls_name) == tokens


class TestResolveSymbolClass:
    """:func:`resolve_symbol_class` — semantic-type string → Symbol subclass."""

    def test_exact_case_insensitive_match(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        assert resolve_symbol_class("WorldBody") is WorldBody
        assert resolve_symbol_class("worldbody") is WorldBody

    def test_camel_tokens(self, symbol_world: Dict[str, Any]) -> None:  # noqa: F811
        """Space-separated tokens match the CamelCase class name."""
        assert resolve_symbol_class("world body") is WorldBody

    def test_synonym_match(self, symbol_world: Dict[str, Any]) -> None:  # noqa: F811
        """``_synonyms`` on a Symbol subclass enables a shortcut alias."""
        assert resolve_symbol_class("milk") is MilkAnnotation

    def test_returns_none_for_unknown(
        self, symbol_world: Dict[str, Any]
    ) -> None:  # noqa: F811
        assert resolve_symbol_class("UnknownType") is None


class TestGetInstances:
    """:func:`get_instances` wraps ``SymbolGraph.get_instances_of_type``."""

    def test_returns_registered_bodies(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        bodies = get_instances(WorldBody)
        names = {body_display_name(b) for b in bodies}
        # base_link is registered too — filter only tests presence of expected names.
        assert {"red_cup", "blue_cup", "milk_on_table"}.issubset(names)

    def test_defaults_to_symbol(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """With no class argument, every Symbol instance is returned."""
        instances = get_instances()
        assert len(instances) >= 8  # bodies + two annotations

    def test_returns_empty_on_unpopulated_graph(self) -> None:
        """When no matching instances exist, the wrapper returns an empty list."""
        assert get_instances(Manipulator) == []


class TestSerializeWorldHeaders:
    """Serialiser output contains the expected markdown sections."""

    def test_includes_world_state_summary(self) -> None:
        result = serialize_world_from_symbol_graph()
        assert result.startswith("## World State Summary")

    def test_includes_all_expected_sections(self) -> None:
        result = serialize_world_from_symbol_graph()
        for section in (
            "## Grounding Instructions",
            "## Scene Objects",
            "## Available Semantic Types",
            "## Spatial Context",
            "## Symbol Relations",
        ):
            assert section in result

    def test_extra_context_has_section(self) -> None:
        result = serialize_world_from_symbol_graph(extra_context="Prefer left cup.")
        assert "## Extra Context" in result
        assert "Prefer left cup." in result

    def test_empty_extra_context_omits_section(self) -> None:
        result = serialize_world_from_symbol_graph(extra_context="")
        assert "## Extra Context" not in result


class TestSerializeWorldContent:
    """Content-level assertions against :fixture:`symbol_world`."""

    def test_registered_bodies_listed(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])
        for name in ("milk_on_table", "red_cup", "blue_cup", "table", "counter"):
            assert name in result

    def test_structural_suffix_filtered_by_default(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])
        assert "base_link" not in result

    def test_robot_owned_bodies_excluded_via_annotation(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """Bodies under a robot annotation are dropped from the scene table."""
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])
        assert "robot_base" not in result

    def test_include_structural_restores_filtered_bodies(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        result = serialize_world_from_symbol_graph(
            symbol_world["body_type"],
            options=WorldSerializationOptions(include_structural=True),
        )
        assert "base_link" in result
        assert "robot_base" in result

    def test_semantic_annotation_table_entry(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])
        assert "- MilkAnnotation: milk_on_table" in result
        assert "| milk_on_table | WorldBody | MilkAnnotation | table |" in result

    def test_parent_context_section(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        result = serialize_world_from_symbol_graph(symbol_world["body_type"])
        assert "- milk_on_table is under/within parent table" in result

    def test_max_objects_truncates_and_reports(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        result = serialize_world_from_symbol_graph(
            symbol_world["body_type"],
            options=WorldSerializationOptions(max_objects=2),
        )
        assert "(showing 2)" in result
        assert "Truncated" in result

    def test_deterministic_output(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        a = serialize_world_from_symbol_graph(symbol_world["body_type"])
        b = serialize_world_from_symbol_graph(symbol_world["body_type"])
        assert a == b


class TestRobotAnnotationMRO:
    """Robot-component classes without ``.bodies`` are still listed under their abstract name."""

    def test_manipulator_appears_in_semantic_types(
        self, robot_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """A :class:`Manipulator` instance shows up under the ``Manipulator`` key."""
        result = serialize_world_from_symbol_graph(Manipulator)
        assert "- Manipulator:" in result

    def test_concrete_subclass_groups_under_parent_name(
        self, robot_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """Subclasses of Manipulator are still grouped under ``Manipulator`` via MRO."""
        result = serialize_world_from_symbol_graph(Manipulator)
        # Both instances appear under the abstract type name.
        assert "left_hand" in result
        assert "right_hand" in result


class TestSerializerNoGraphFallback:
    """If SymbolGraph access fails, the serialiser returns a stub rather than crashing."""

    def test_no_scene_objects_row_when_graph_unavailable(self, monkeypatch) -> None:
        import llmr.bridge.world_reader as reader

        class BrokenGraph:
            def __init__(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(reader, "SymbolGraph", BrokenGraph)
        result = reader.serialize_world_from_symbol_graph()
        assert "## Scene Objects" in result
        assert "No scene objects found in SymbolGraph." in result


class TestSerializerWithExternalGraph:
    """Passing an explicit graph bypasses the singleton."""

    def test_empty_graph_reports_zero_objects(self) -> None:
        from krrood.symbol_graph.symbol_graph import SymbolGraph

        graph = SymbolGraph()
        result = serialize_world_from_symbol_graph(symbol_graph=graph)
        # The bridge still prints "Objects: 0 visible" when everything is hidden.
        assert "Objects:" in result
