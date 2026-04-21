"""Tests for :mod:`llmr.bridge.introspect` — action dataclass field classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing_extensions import List, Optional, Type

import pytest
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.bridge.introspect import (
    NO_DEFAULT,
    ActionSchema,
    FieldKind,
    FieldSpec,
    OwnDataclassIntrospector,
    PycramIntrospector,
    introspect,
)

from .._fixtures.actions import (
    GraspType,
    MockContainerAction,
    MockGraspDescription,
    MockNavigateAction,
    MockPickUpAction,
    MockPose,
    MockPoseAction,
    MockRequiredManipulatorAction,
    MockTypeRefAction,
)
from .._fixtures.symbols import Manipulator, ParallelGripperLike


class TestIntrospectSchema:
    """:meth:`PycramIntrospector.introspect` returns a fully populated :class:`ActionSchema`."""

    def test_action_type_name_is_class_name(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockPickUpAction)
        assert schema.action_type == "MockPickUpAction"
        assert schema.action_cls is MockPickUpAction

    def test_class_docstring_is_captured(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockPickUpAction)
        assert schema.docstring == "Minimal stand-in for PyCRAM PickUpAction."

    def test_own_fields_extracted(self, introspector: PycramIntrospector) -> None:
        schema = introspector.introspect(MockPickUpAction)
        names = {f.name for f in schema.fields}
        assert names == {"object_designator", "grasp_description", "timeout"}

    def test_non_dataclass_raises(self, introspector: PycramIntrospector) -> None:
        class NotADataclass:
            pass

        with pytest.raises(TypeError):
            introspector.introspect(NotADataclass)

    def test_entity_field_spec(self, introspector: PycramIntrospector) -> None:
        schema = introspector.introspect(MockPickUpAction)
        field = next(f for f in schema.fields if f.name == "object_designator")
        assert field.kind == FieldKind.ENTITY
        assert field.is_optional is False
        assert field.default is NO_DEFAULT
        assert field.docstring == "The object to pick up."
        assert field.raw_type is Symbol

    def test_optional_primitive_field_spec(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockPickUpAction)
        field = next(f for f in schema.fields if f.name == "timeout")
        assert field.kind == FieldKind.PRIMITIVE
        assert field.is_optional is True

    def test_complex_field_expands_sub_fields(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockPickUpAction)
        field = next(f for f in schema.fields if f.name == "grasp_description")
        assert field.kind == FieldKind.COMPLEX
        sub_names = {sf.name for sf in field.sub_fields}
        assert sub_names == {"grasp_type", "manipulator"}

    def test_enum_field_lists_members(self, introspector: PycramIntrospector) -> None:
        schema = introspector.introspect(MockGraspDescription)
        field = next(f for f in schema.fields if f.name == "grasp_type")
        assert field.kind == FieldKind.ENUM
        assert set(field.enum_members) == {"FRONT", "TOP", "SIDE"}

    def test_type_ref_keeps_inner_type(self, introspector: PycramIntrospector) -> None:
        schema = introspector.introspect(MockTypeRefAction)
        field = next(f for f in schema.fields if f.name == "annotation_type")
        assert field.kind == FieldKind.TYPE_REF
        assert field.raw_type is Symbol

    def test_required_symbol_subclass_entity(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockRequiredManipulatorAction)
        field = next(f for f in schema.fields if f.name == "manipulator")
        assert field.kind == FieldKind.ENTITY
        assert field.is_optional is False

    def test_pose_field_classified_via_mro_name(
        self, introspector: PycramIntrospector
    ) -> None:
        """``target_pose: MockPose`` hits the POSE branch via class-name MRO match."""
        schema = introspector.introspect(MockPoseAction)
        field = next(f for f in schema.fields if f.name == "target_pose")
        assert field.kind == FieldKind.POSE

    def test_container_field_falls_back_to_primitive(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockContainerAction)
        field = next(f for f in schema.fields if f.name == "object_designators")
        assert field.kind == FieldKind.PRIMITIVE


class TestClassifyType:
    """Direct :meth:`classify_type` coverage for each :class:`FieldKind` branch."""

    @pytest.mark.parametrize("primitive", [bool, int, float, str, bytes])
    def test_scalars_are_primitive(
        self, introspector: PycramIntrospector, primitive: type
    ) -> None:
        assert introspector.classify_type(primitive) is FieldKind.PRIMITIVE

    def test_none_is_primitive(self, introspector: PycramIntrospector) -> None:
        assert introspector.classify_type(None) is FieldKind.PRIMITIVE
        assert introspector.classify_type(type(None)) is FieldKind.PRIMITIVE

    def test_non_type_is_primitive(self, introspector: PycramIntrospector) -> None:
        """Non-type inputs (e.g. string annotations) fall through to PRIMITIVE."""
        assert introspector.classify_type("not-a-type") is FieldKind.PRIMITIVE

    def test_enum_subclass(self, introspector: PycramIntrospector) -> None:
        assert introspector.classify_type(GraspType) is FieldKind.ENUM

    def test_symbol_subclass_is_entity(self, introspector: PycramIntrospector) -> None:
        assert introspector.classify_type(Symbol) is FieldKind.ENTITY
        assert introspector.classify_type(Manipulator) is FieldKind.ENTITY
        assert introspector.classify_type(ParallelGripperLike) is FieldKind.ENTITY

    def test_dataclass_is_complex(self, introspector: PycramIntrospector) -> None:
        assert introspector.classify_type(MockGraspDescription) is FieldKind.COMPLEX

    def test_type_annotation_is_type_ref(
        self, introspector: PycramIntrospector
    ) -> None:
        assert introspector.classify_type(Type[Symbol]) is FieldKind.TYPE_REF

    def test_pose_name_match(self, introspector: PycramIntrospector) -> None:
        """Any class whose MRO contains a name in POSE_TYPE_NAMES is classified POSE."""

        class Pose:
            pass

        class CustomPose(Pose):
            pass

        assert introspector.classify_type(Pose) is FieldKind.POSE
        assert introspector.classify_type(CustomPose) is FieldKind.POSE


class TestDocstringExtraction:
    """AST-based field docstring extraction covers multiple layouts."""

    def test_attribute_docstrings_collected(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockPickUpAction)
        docs = {f.name: f.docstring for f in schema.fields}
        assert docs["object_designator"] == "The object to pick up."
        assert docs["timeout"] == "Maximum seconds to attempt the action."

    def test_fields_without_docstring_default_to_empty(
        self, introspector: PycramIntrospector
    ) -> None:
        @dataclass
        class PlainAction:
            value: int

        schema = introspector.introspect(PlainAction)
        assert schema.fields[0].docstring == ""

    def test_source_unavailable_returns_empty_docs(self) -> None:
        """When ``inspect.getsource`` fails, docstrings come back empty, not broken."""
        local_cls = type(
            "DynamicAction",
            (object,),
            {"__annotations__": {"x": int}, "__module__": __name__},
        )
        # Force a dataclass with no accessible source (built from type()).
        from dataclasses import dataclass as _dc

        local_cls = _dc(local_cls)
        docs = PycramIntrospector._extract_field_docstrings(local_cls)
        assert docs == {}


class TestOwnDataclassIntrospector:
    """:class:`OwnDataclassIntrospector` filters inherited fields."""

    def test_inherited_fields_excluded(self) -> None:
        @dataclass
        class Parent:
            parent_field: int = 0

        @dataclass
        class Child(Parent):
            child_field: int = 0

        introspector = OwnDataclassIntrospector()
        names = {attr.public_name for attr in introspector.discover(Child)}
        assert names == {"child_field"}


class TestModuleIntrospectHelper:
    """The module-level ``introspect()`` convenience wrapper."""

    def test_returns_action_schema(self) -> None:
        schema = introspect(MockNavigateAction)
        assert isinstance(schema, ActionSchema)
        assert schema.action_cls is MockNavigateAction
        assert any(f.name == "target_location" for f in schema.fields)


class TestFieldSpecDefaults:
    """Field defaults and sentinel values."""

    def test_required_field_uses_no_default_sentinel(
        self, introspector: PycramIntrospector
    ) -> None:
        schema = introspector.introspect(MockPickUpAction)
        required = next(f for f in schema.fields if f.name == "object_designator")
        assert required.default is NO_DEFAULT

    def test_field_spec_instantiates_with_defaults(self) -> None:
        spec = FieldSpec(name="x", raw_type=int, kind=FieldKind.PRIMITIVE)
        assert spec.default is NO_DEFAULT
        assert spec.enum_members == []
        assert spec.sub_fields == []
