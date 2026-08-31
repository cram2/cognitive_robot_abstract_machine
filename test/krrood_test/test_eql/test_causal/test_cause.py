from dataclasses import dataclass

from krrood.entity_query_language.operators.causal import Cause
from krrood.entity_query_language.factories import a, cause

# %% construction


def test_cause_is_a_cause_instance_with_no_type_of_its_own():
    assert isinstance(cause, Cause)
    assert cause._type_ is None


# %% flowing through Match (backfilled with a fresh, per-attribute copy on resolution)


@dataclass
class Pick:
    arm: float
    grasped: bool


def _cause_attribute_match(match):
    [attribute_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path == "Pick.arm"
    ]
    return attribute_match


def test_cause_flows_through_match_as_the_assigned_variable():
    match = a(Pick)(arm=cause, grasped=True)
    assert isinstance(_cause_attribute_match(match).assigned_variable, Cause)


def test_cause_backfills_its_type_from_the_attribute_it_is_assigned_to():
    match = a(Pick)(arm=cause, grasped=True)
    assert _cause_attribute_match(match).assigned_variable._type_ is float


def test_match_marks_a_cause_attribute_as_present():
    match = a(Pick)(arm=cause, grasped=True)
    assert match.has_cause_attributes is True


def test_match_without_cause_reports_no_cause_attributes():
    match = a(Pick)(arm=0.3, grasped=True)
    assert match.has_cause_attributes is False


def test_two_cause_marked_attributes_resolve_to_distinct_objects():
    # Backfilling one attribute's type onto the shared `cause` instance in place
    # would silently overwrite it for every other attribute also marked `cause` --
    # each attribute must resolve to its own copy instead.
    match = a(Pick)(arm=cause, grasped=cause)
    [arm_match, grasped_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path
        in ("Pick.arm", "Pick.grasped")
    ]
    assert arm_match.assigned_variable is not grasped_match.assigned_variable
    assert arm_match.assigned_variable._type_ is float
    assert grasped_match.assigned_variable._type_ is bool


def test_marking_attributes_as_cause_does_not_mutate_the_shared_instance():
    a(Pick)(arm=cause, grasped=cause)
    assert cause._type_ is None
