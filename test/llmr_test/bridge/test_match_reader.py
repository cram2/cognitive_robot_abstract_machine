"""Tests for :mod:`llmr.bridge.match_reader` — Match → MatchData snapshots."""

from __future__ import annotations

import pytest

from llmr.bridge.introspect import FieldKind, PycramIntrospector
from llmr.bridge.match_reader import (
    MatchData,
    MatchSlot,
    finalize_match,
    read_match,
    required_match,
    unresolved_required_fields,
    write_slot_value,
)

from .._fixtures.actions import (
    GraspType,
    MockGraspDescription,
    MockNavigateAction,
    MockNestedWithTimeoutAction,
    MockPickUpAction,
    MockRequiredManipulatorAction,
    MockRequiredNestedAction,
)
from .._fixtures.symbols import Manipulator, WorldBody


class TestRequiredMatch:
    """:func:`required_match` leaves only required public fields free."""

    def test_leaves_optional_fields_bound(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockPickUpAction, introspector)
        # Only `object_designator` is required on MockPickUpAction.
        assert set(match.kwargs) == {"object_designator"}
        assert match.kwargs["object_designator"] is Ellipsis

    def test_required_primitive_navigate_action(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockNavigateAction, introspector)
        assert set(match.kwargs) == {"target_location"}

    def test_complex_field_becomes_nested_match(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockRequiredNestedAction, introspector)
        grasp_value = match.kwargs["grasp"]
        from krrood.entity_query_language.query.match import Match as KMatch

        assert isinstance(grasp_value, KMatch)
        # Nested Match exposes its own required sub-field as Ellipsis.
        assert grasp_value.kwargs["grasp_type"] is Ellipsis

    def test_uses_default_introspector_when_none_provided(self) -> None:
        """Calling without an introspector still produces a usable Match."""
        match = required_match(MockPickUpAction)
        assert match.type is MockPickUpAction


class TestReadMatch:
    """:func:`read_match` snapshots Match expressions into :class:`MatchData`."""

    def test_returns_match_data_with_action_type(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        assert isinstance(data, MatchData)
        assert data.action_type is MockPickUpAction
        assert data.action_name == "MockPickUpAction"

    def test_slot_prompt_name_strips_root_prefix(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        slot = data.slots[0]
        assert slot.prompt_name == "object_designator"
        assert slot.attribute_name == "object_designator"

    def test_slot_prompt_name_preserves_nested_dot(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockRequiredNestedAction, introspector)
        data = read_match(match, introspector)
        nested = next(s for s in data.slots if s.attribute_name == "grasp_type")
        assert nested.prompt_name == "grasp.grasp_type"

    def test_field_kinds_are_pre_classified(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockNestedWithTimeoutAction, introspector)
        data = read_match(match, introspector)
        kinds = {s.attribute_name: s.field_kind for s in data.slots}
        assert kinds["priority"] is FieldKind.PRIMITIVE
        assert kinds["grasp_type"] is FieldKind.ENUM

    def test_free_slots_all_ellipsis(self, introspector: PycramIntrospector) -> None:
        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        assert data.free_slot_names == ["object_designator"]
        assert all(s.is_free for s in data.free_slots)

    def test_slot_value_is_ellipsis_for_unbound_variables(
        self, introspector: PycramIntrospector
    ) -> None:
        """Unbound variables surface as ``Ellipsis`` with ``is_free=True``."""
        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        slot = data.slots[0]
        assert slot.is_free is True
        assert slot.value is ...

    def test_fixed_bindings_report_non_free_slots(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockPickUpAction, introspector)
        body = WorldBody("milk")
        attr_match = next(iter(match.matches_with_variables))
        attr_match.assigned_variable._value_ = body

        data = read_match(match, introspector)
        assert data.free_slots == []
        assert data.fixed_bindings == {"object_designator": body}

    def test_defaults_to_built_in_introspector(self) -> None:
        match = required_match(MockPickUpAction)
        data = read_match(match)
        assert data.slots, "expected at least one slot"


class TestWriteSlotValue:
    """:func:`write_slot_value` updates both the variable and the snapshot."""

    def test_writes_and_flips_is_free(self, introspector: PycramIntrospector) -> None:
        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        slot = data.slots[0]
        body = WorldBody("milk")

        ok = write_slot_value(slot, body)
        assert ok is True
        assert slot.is_free is False
        assert slot.value is body
        assert slot._variable._value_ is body

    def test_returns_false_when_assignment_raises(
        self, introspector: PycramIntrospector
    ) -> None:
        """A variable whose ``_value_`` assignment fails returns False instead of raising."""

        class ImmutableVar:
            _type_ = str

            def __setattr__(self, name: str, value: object) -> None:
                if name == "_value_":
                    raise RuntimeError("read-only")
                object.__setattr__(self, name, value)

        slot = MatchSlot(
            attribute_name="x",
            prompt_name="x",
            field_type=str,
            field_kind=FieldKind.PRIMITIVE,
            value=...,
            is_free=True,
            _variable=ImmutableVar(),
        )
        assert write_slot_value(slot, "v") is False
        # is_free stays True because the write failed.
        assert slot.is_free is True


class TestFinalizeMatch:
    """:func:`finalize_match` produces the concrete action instance."""

    def test_fully_resolved_navigate_action(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockNavigateAction, introspector)
        data = read_match(match, introspector)
        write_slot_value(data.slots[0], WorldBody("kitchen"))

        result = finalize_match(data)
        assert isinstance(result, MockNavigateAction)
        assert isinstance(result.target_location, WorldBody)
        assert result.target_location.name == "kitchen"

    def test_nested_enum_resolution(self, introspector: PycramIntrospector) -> None:
        match = required_match(MockRequiredNestedAction, introspector)
        data = read_match(match, introspector)
        for slot in data.slots:
            if slot.attribute_name == "object_designator":
                write_slot_value(slot, WorldBody("milk"))
            elif slot.attribute_name == "grasp_type":
                write_slot_value(slot, GraspType.FRONT)

        result = finalize_match(data)
        assert isinstance(result, MockRequiredNestedAction)
        assert result.grasp.grasp_type is GraspType.FRONT


class TestUnresolvedRequiredFields:
    """:func:`unresolved_required_fields` identifies fields still set to Ellipsis."""

    def test_all_free_means_all_unresolved(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        assert unresolved_required_fields(data, introspector) == ["object_designator"]

    def test_resolved_slot_removed_from_unresolved(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        write_slot_value(data.slots[0], WorldBody("milk"))
        assert unresolved_required_fields(data, introspector) == []

    def test_nested_match_counts_as_unresolved_until_leaves_are_set(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockRequiredNestedAction, introspector)
        data = read_match(match, introspector)
        # object_designator not yet set; grasp has an unresolved leaf.
        unresolved = unresolved_required_fields(data, introspector)
        assert "object_designator" in unresolved
        assert "grasp" in unresolved

    def test_nested_resolution_clears_grasp(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockRequiredNestedAction, introspector)
        data = read_match(match, introspector)
        for slot in data.slots:
            if slot.attribute_name == "object_designator":
                write_slot_value(slot, WorldBody("milk"))
            elif slot.attribute_name == "grasp_type":
                write_slot_value(slot, GraspType.FRONT)

        assert unresolved_required_fields(data, introspector) == []

    def test_non_introspectable_returns_empty(
        self, introspector: PycramIntrospector
    ) -> None:
        """When introspection of the action type fails the helper degrades gracefully."""

        class _ExplodingIntrospector:
            def introspect(self, action_cls: type) -> None:
                raise RuntimeError("boom")

        match = required_match(MockPickUpAction, introspector)
        data = read_match(match, introspector)
        assert (
            unresolved_required_fields(data, _ExplodingIntrospector())  # type: ignore[arg-type]
            == []
        )


class TestMatchDataProperties:
    """Derived properties on :class:`MatchData`."""

    def test_free_slot_names_match_free_slots(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockNestedWithTimeoutAction, introspector)
        data = read_match(match, introspector)
        assert set(data.free_slot_names) == {s.prompt_name for s in data.free_slots}

    def test_mixed_fixed_and_free_bindings(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockNestedWithTimeoutAction, introspector)
        data = read_match(match, introspector)
        # Fix priority; grasp.grasp_type still free.
        priority_slot = next(s for s in data.slots if s.attribute_name == "priority")
        write_slot_value(priority_slot, 7)

        assert data.fixed_bindings == {"priority": 7}
        assert data.free_slot_names == ["grasp.grasp_type"]


class TestRequiredManipulatorMatch:
    """Symbol subclasses are still classified ENTITY and required when un-defaulted."""

    def test_manipulator_slot_is_entity(self, introspector: PycramIntrospector) -> None:
        match = required_match(MockRequiredManipulatorAction, introspector)
        data = read_match(match, introspector)
        slot = next(s for s in data.slots if s.attribute_name == "manipulator")
        assert slot.field_kind is FieldKind.ENTITY
        assert slot.is_free is True

    def test_manipulator_assignment_roundtrip(
        self, introspector: PycramIntrospector
    ) -> None:
        match = required_match(MockRequiredManipulatorAction, introspector)
        data = read_match(match, introspector)
        left = Manipulator("left_hand")
        write_slot_value(data.slots[0], left)
        result = finalize_match(data)
        assert isinstance(result, MockRequiredManipulatorAction)
        assert result.manipulator is left
