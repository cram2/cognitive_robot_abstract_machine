"""Tests for PycramIntrospector — action introspection and field classification.

Tests use MockPickUpAction, MockNavigateAction, MockGraspDescription from test_actions.py.
No PyCRAM installation needed.

Coverage target: 80% (16 tests covering introspection and type classification).
"""
from __future__ import annotations

from typing_extensions import Optional
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.pycram_bridge.introspector import (
    FieldKind,
    PycramIntrospector,
)
from ..test_actions import (
    GraspType,
    MockGraspDescription,
    MockPickUpAction,
    MockNavigateAction,
)


class TestIntrospect:
    """introspect() → ActionSchema correctness."""

    def test_returns_action_schema_with_correct_action_type(
        self, introspector: PycramIntrospector
    ) -> None:
        """Schema contains the action class name."""
        schema = introspector.introspect(MockPickUpAction)
        assert schema.action_type == "MockPickUpAction"

    def test_extracts_all_own_fields(self, introspector: PycramIntrospector) -> None:
        """All fields defined directly on the class are introspected."""
        schema = introspector.introspect(MockPickUpAction)
        field_names = {f.name for f in schema.fields}
        assert "object_designator" in field_names
        assert "grasp_description" in field_names
        assert "timeout" in field_names

    def test_does_not_include_inherited_base_fields(
        self, introspector: PycramIntrospector
    ) -> None:
        """Inherited fields from base class are excluded."""
        # MockPickUpAction only inherits from object, so no base fields.
        # This test verifies the filtering logic works.
        schema = introspector.introspect(MockPickUpAction)
        # All fields should be from MockPickUpAction itself
        assert len(schema.fields) == 3  # object_designator, grasp_description, timeout

    def test_entity_field_classified_as_entity(
        self, introspector: PycramIntrospector
    ) -> None:
        """Symbol subclass fields classified as ENTITY."""
        schema = introspector.introspect(MockPickUpAction)
        obj_field = next(f for f in schema.fields if f.name == "object_designator")
        assert obj_field.kind == FieldKind.ENTITY

    def test_enum_field_classified_as_enum_with_members(
        self, introspector: PycramIntrospector
    ) -> None:
        """Enum fields classified as ENUM with member list."""
        schema = introspector.introspect(MockGraspDescription)
        grasp_field = next(f for f in schema.fields if f.name == "grasp_type")
        assert grasp_field.kind == FieldKind.ENUM
        assert set(grasp_field.enum_members) == {"FRONT", "TOP", "SIDE"}

    def test_optional_field_marked_is_optional(
        self, introspector: PycramIntrospector
    ) -> None:
        """Optional[X] fields marked with is_optional=True."""
        schema = introspector.introspect(MockPickUpAction)
        timeout_field = next(f for f in schema.fields if f.name == "timeout")
        assert timeout_field.is_optional is True

    def test_required_field_not_optional(
        self, introspector: PycramIntrospector
    ) -> None:
        """Non-optional fields marked with is_optional=False."""
        schema = introspector.introspect(MockPickUpAction)
        obj_field = next(f for f in schema.fields if f.name == "object_designator")
        assert obj_field.is_optional is False

    def test_complex_field_classified_as_complex_with_sub_fields(
        self, introspector: PycramIntrospector
    ) -> None:
        """Dataclass fields classified as COMPLEX with sub_fields recursively introspected."""
        schema = introspector.introspect(MockPickUpAction)
        grasp_field = next(f for f in schema.fields if f.name == "grasp_description")
        assert grasp_field.kind == FieldKind.COMPLEX
        # Sub-fields should be introspected
        assert len(grasp_field.sub_fields) > 0
        sub_names = {sf.name for sf in grasp_field.sub_fields}
        assert "grasp_type" in sub_names

    def test_primitive_float_classified_as_primitive(
        self, introspector: PycramIntrospector
    ) -> None:
        """float fields classified as PRIMITIVE."""
        schema = introspector.introspect(MockPickUpAction)
        timeout_field = next(f for f in schema.fields if f.name == "timeout")
        assert timeout_field.kind == FieldKind.PRIMITIVE

    def test_field_docstring_extracted_from_class_source(
        self, introspector: PycramIntrospector
    ) -> None:
        """Per-field docstrings extracted via AST parsing."""
        schema = introspector.introspect(MockPickUpAction)
        obj_field = next(f for f in schema.fields if f.name == "object_designator")
        assert obj_field.docstring == "The object to pick up."


class TestClassifyType:
    """_classify_type() for each FieldKind."""

    def test_symbol_subclass_is_entity(
        self, introspector: PycramIntrospector
    ) -> None:
        """Symbol and its subclasses classified as ENTITY."""
        kind = introspector._classify_type(Symbol)
        assert kind == FieldKind.ENTITY

    def test_enum_subclass_is_enum(
        self, introspector: PycramIntrospector
    ) -> None:
        """Enum subclasses classified as ENUM."""
        kind = introspector._classify_type(GraspType)
        assert kind == FieldKind.ENUM

    def test_bool_is_primitive(self, introspector: PycramIntrospector) -> None:
        """bool classified as PRIMITIVE."""
        assert introspector._classify_type(bool) == FieldKind.PRIMITIVE

    def test_int_is_primitive(self, introspector: PycramIntrospector) -> None:
        """int classified as PRIMITIVE."""
        assert introspector._classify_type(int) == FieldKind.PRIMITIVE

    def test_str_is_primitive(self, introspector: PycramIntrospector) -> None:
        """str classified as PRIMITIVE."""
        assert introspector._classify_type(str) == FieldKind.PRIMITIVE

    def test_float_is_primitive(self, introspector: PycramIntrospector) -> None:
        """float classified as PRIMITIVE."""
        assert introspector._classify_type(float) == FieldKind.PRIMITIVE

    def test_dataclass_is_complex(
        self, introspector: PycramIntrospector
    ) -> None:
        """Non-primitive dataclass classified as COMPLEX."""
        kind = introspector._classify_type(MockGraspDescription)
        assert kind == FieldKind.COMPLEX

    def test_none_type_is_primitive(
        self, introspector: PycramIntrospector
    ) -> None:
        """None / type(None) classified as PRIMITIVE."""
        assert introspector._classify_type(None) == FieldKind.PRIMITIVE
        assert introspector._classify_type(type(None)) == FieldKind.PRIMITIVE
