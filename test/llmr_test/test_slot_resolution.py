"""Tests for :mod:`llmr.resolution.slot_resolution` — LLM slot output → Python value dispatch.

Each :class:`FieldKind` path is exercised once with a free :class:`MatchSlot` whose
``field_kind`` is pre-classified, plus the ENUM coercion and primitive coercion helpers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing_extensions import Any, Dict, Optional

import pytest
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.bridge.introspect import FieldKind
from llmr.bridge.match_reader import MatchSlot
from llmr.resolution.grounder import EntityGrounder
from llmr.resolution.slot_resolution import (
    coerce_enum,
    coerce_primitive,
    resolve_binding_value,
    resolve_entity_slot,
)
from llmr.schemas import EntityDescriptionSchema, SlotValue

from ._fixtures.actions import GraspType, MockGraspDescription
from ._fixtures.symbols import Manipulator, MilkAnnotation, WorldBody
from ._fixtures.worlds import robot_world, symbol_world  # noqa: F401

_SENTINEL = object()


def _make_slot(
    attribute_name: str,
    field_type: Any,
    field_kind: FieldKind,
    *,
    prompt_name: Optional[str] = None,
) -> MatchSlot:
    """Construct a free :class:`MatchSlot` stub for resolver tests."""
    return MatchSlot(
        attribute_name=attribute_name,
        prompt_name=prompt_name or attribute_name,
        field_type=field_type,
        field_kind=field_kind,
        value=...,
        is_free=True,
        _variable=SimpleNamespace(_value_=..., _type_=field_type),
    )


class TestResolveBindingEntity:
    """ENTITY slots delegate to :func:`resolve_entity_slot` via the grounder."""

    def test_returns_grounded_body(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        slot = _make_slot("object_designator", WorldBody, FieldKind.ENTITY)
        sv = SlotValue(
            field_name="object_designator",
            entity_description=EntityDescriptionSchema(name="milk_on_table"),
        )
        grounder = EntityGrounder(groundable_type=WorldBody)
        out = resolve_binding_value(
            slot, {"object_designator": sv}, grounder, {}, _SENTINEL
        )
        assert out is symbol_world["milk_on_table"]

    def test_missing_slot_value_returns_unresolved(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        slot = _make_slot("object_designator", WorldBody, FieldKind.ENTITY)
        grounder = EntityGrounder(groundable_type=WorldBody)
        out = resolve_binding_value(slot, {}, grounder, {}, _SENTINEL)
        assert out is _SENTINEL

    def test_wrong_expected_type_returns_unresolved(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        """Grounder finds a body, but it is not an instance of the required type."""
        slot = _make_slot("manipulator", Manipulator, FieldKind.ENTITY)
        sv = SlotValue(
            field_name="manipulator",
            entity_description=EntityDescriptionSchema(name="milk_on_table"),
        )
        grounder = EntityGrounder(groundable_type=WorldBody)
        out = resolve_binding_value(slot, {"manipulator": sv}, grounder, {}, _SENTINEL)
        assert out is _SENTINEL


class TestResolveBindingPose:
    """POSE slots ground a body and then unwrap ``global_pose``."""

    def test_returns_global_pose(self) -> None:
        pose = SimpleNamespace(tag="pose-stub")
        body = SimpleNamespace(name="kitchen_origin", global_pose=pose)

        class FixedGrounder:
            def ground(self, description: EntityDescriptionSchema, expected_type=None):
                from llmr.resolution.grounder import GroundingResult

                return GroundingResult(bodies=[body])

        slot = _make_slot("target_pose", object, FieldKind.POSE)
        sv = SlotValue(
            field_name="target_pose",
            entity_description=EntityDescriptionSchema(name="kitchen_origin"),
        )
        out = resolve_binding_value(
            slot, {"target_pose": sv}, FixedGrounder(), {}, _SENTINEL
        )
        assert out is pose

    def test_missing_global_pose_returns_unresolved(self) -> None:
        body = SimpleNamespace(name="no_pose_body")

        class FixedGrounder:
            def ground(self, description, expected_type=None):
                from llmr.resolution.grounder import GroundingResult

                return GroundingResult(bodies=[body])

        slot = _make_slot("target_pose", object, FieldKind.POSE)
        sv = SlotValue(
            field_name="target_pose",
            entity_description=EntityDescriptionSchema(name="no_pose_body"),
        )
        out = resolve_binding_value(
            slot, {"target_pose": sv}, FixedGrounder(), {}, _SENTINEL
        )
        assert out is _SENTINEL


class TestResolveBindingTypeRef:
    """TYPE_REF slots prefer :func:`resolve_symbol_class` over the grounded body."""

    def test_returns_class_when_semantic_type_resolves(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        slot = _make_slot("annotation_type", Symbol, FieldKind.TYPE_REF)
        sv = SlotValue(
            field_name="annotation_type",
            entity_description=EntityDescriptionSchema(
                name="milk", semantic_type="MilkAnnotation"
            ),
        )
        grounder = EntityGrounder(groundable_type=Symbol)
        out = resolve_binding_value(
            slot, {"annotation_type": sv}, grounder, {}, _SENTINEL
        )
        assert out is MilkAnnotation

    def test_falls_back_to_grounded_body_without_semantic_type(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        slot = _make_slot("annotation_type", Symbol, FieldKind.TYPE_REF)
        sv = SlotValue(
            field_name="annotation_type",
            entity_description=EntityDescriptionSchema(name="milk_on_table"),
        )
        grounder = EntityGrounder(groundable_type=WorldBody)
        out = resolve_binding_value(
            slot, {"annotation_type": sv}, grounder, {}, _SENTINEL
        )
        assert out is symbol_world["milk_on_table"]


class TestResolveBindingEnum:
    """ENUM slots coerce a plain string value to an enum member."""

    def test_enum_coercion(self) -> None:
        slot = _make_slot("grasp_type", GraspType, FieldKind.ENUM)
        sv = SlotValue(field_name="grasp_type", value="FRONT")
        out = resolve_binding_value(
            slot, {"grasp_type": sv}, EntityGrounder(), {}, _SENTINEL
        )
        assert out is GraspType.FRONT

    def test_empty_enum_returns_unresolved(self) -> None:
        slot = _make_slot("grasp_type", GraspType, FieldKind.ENUM)
        sv = SlotValue(field_name="grasp_type", value=None)
        out = resolve_binding_value(
            slot, {"grasp_type": sv}, EntityGrounder(), {}, _SENTINEL
        )
        assert out is _SENTINEL


class TestResolveBindingPrimitive:
    """PRIMITIVE slots cast the LLM string to the field's scalar type."""

    def test_float_value_coerced(self) -> None:
        slot = _make_slot("timeout", float, FieldKind.PRIMITIVE)
        sv = SlotValue(field_name="timeout", value="1.5")
        out = resolve_binding_value(
            slot, {"timeout": sv}, EntityGrounder(), {}, _SENTINEL
        )
        assert out == 1.5

    def test_missing_primitive_returns_unresolved(self) -> None:
        slot = _make_slot("timeout", float, FieldKind.PRIMITIVE)
        out = resolve_binding_value(slot, {}, EntityGrounder(), {}, _SENTINEL)
        assert out is _SENTINEL


class TestResolveBindingComplex:
    """COMPLEX slots are owned by nested Match expressions, not the resolver."""

    def test_complex_returns_unresolved(self) -> None:
        slot = _make_slot("grasp_description", MockGraspDescription, FieldKind.COMPLEX)
        sv = SlotValue(
            field_name="grasp_description",
            value="doesn't matter",
        )
        out = resolve_binding_value(
            slot, {"grasp_description": sv}, EntityGrounder(), {}, _SENTINEL
        )
        assert out is _SENTINEL


class TestResolveEntitySlotRecovery:
    """When the LLM uses ``value`` instead of ``entity_description``."""

    def test_value_fallback_reconstructs_description(
        self, symbol_world: Dict[str, Any]  # noqa: F811
    ) -> None:
        sv = SlotValue(field_name="object_designator", value="milk_on_table")
        grounder = EntityGrounder(groundable_type=WorldBody)
        out = resolve_entity_slot(
            sv,
            grounder,
            FieldKind.ENTITY,
            field_name="object_designator",
            expected_type=WorldBody,
            unresolved=_SENTINEL,
        )
        assert out is symbol_world["milk_on_table"]

    def test_both_missing_returns_unresolved(self) -> None:
        sv = SlotValue(field_name="object_designator")
        grounder = EntityGrounder(groundable_type=WorldBody)
        out = resolve_entity_slot(
            sv,
            grounder,
            FieldKind.ENTITY,
            field_name="object_designator",
            unresolved=_SENTINEL,
        )
        assert out is _SENTINEL

    def test_no_bodies_returns_unresolved(self) -> None:
        sv = SlotValue(
            field_name="object_designator",
            entity_description=EntityDescriptionSchema(name="does_not_exist"),
        )
        grounder = EntityGrounder(groundable_type=WorldBody)
        out = resolve_entity_slot(
            sv,
            grounder,
            FieldKind.ENTITY,
            field_name="object_designator",
            unresolved=_SENTINEL,
        )
        assert out is _SENTINEL


class TestCoerceEnum:
    """:func:`coerce_enum` — case-insensitive name match, with first-member fallback."""

    def test_exact_member(self) -> None:
        assert coerce_enum("FRONT", GraspType) is GraspType.FRONT

    def test_case_insensitive_member(self) -> None:
        assert coerce_enum("front", GraspType) is GraspType.FRONT

    def test_unknown_falls_back_to_first(self) -> None:
        assert coerce_enum("UNKNOWN", GraspType) is next(iter(GraspType))


class TestCoercePrimitive:
    """:func:`coerce_primitive` — string → bool / int / float / str."""

    @pytest.mark.parametrize(
        "value, expected",
        [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False)],
    )
    def test_bool_parsing(self, value: str, expected: bool) -> None:
        assert coerce_primitive(value, bool) is expected

    def test_int_parsing(self) -> None:
        assert coerce_primitive("42", int) == 42

    def test_int_parsing_failure_returns_raw(self) -> None:
        assert coerce_primitive("not-a-number", int) == "not-a-number"

    def test_float_parsing(self) -> None:
        assert coerce_primitive("3.14", float) == pytest.approx(3.14)

    def test_float_parsing_failure_returns_raw(self) -> None:
        assert coerce_primitive("nope", float) == "nope"

    def test_optional_unwrapping(self) -> None:
        """``Optional[int]`` unwraps to ``int`` for coercion."""
        assert coerce_primitive("7", Optional[int]) == 7

    def test_plain_string_passthrough(self) -> None:
        assert coerce_primitive("hello", str) == "hello"
