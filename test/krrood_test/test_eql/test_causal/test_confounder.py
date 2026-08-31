from dataclasses import dataclass

from krrood.entity_query_language.operators.causal import Confounder
from krrood.entity_query_language.factories import a, confounder

# %% construction


def test_confounder_is_a_confounder_instance_with_no_type_of_its_own():
    assert isinstance(confounder, Confounder)
    assert confounder._type_ is None


# %% flowing through Match (backfilled with a fresh, per-attribute copy on resolution)


@dataclass
class Trial:
    treatment: float
    season: str


def _confounder_attribute_match(match):
    [attribute_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path == "Trial.season"
    ]
    return attribute_match


def test_confounder_flows_through_match_as_the_assigned_variable():
    match = a(Trial)(treatment=0.3, season=confounder)
    assert isinstance(_confounder_attribute_match(match).assigned_variable, Confounder)


def test_confounder_backfills_its_type_from_the_attribute_it_is_assigned_to():
    match = a(Trial)(treatment=0.3, season=confounder)
    assert _confounder_attribute_match(match).assigned_variable._type_ is str


def test_two_confounder_marked_attributes_resolve_to_distinct_objects():
    # Backfilling one attribute's type onto the shared `confounder` instance in
    # place would silently overwrite it for every other attribute also marked
    # `confounder` -- each attribute must resolve to its own copy instead.
    match = a(Trial)(treatment=confounder, season=confounder)
    [treatment_match, season_match] = [
        attribute_match
        for attribute_match in match.matches_with_variables
        if attribute_match.name_from_variable_access_path
        in ("Trial.treatment", "Trial.season")
    ]
    assert treatment_match.assigned_variable is not season_match.assigned_variable
    assert treatment_match.assigned_variable._type_ is float
    assert season_match.assigned_variable._type_ is str


def test_marking_attributes_as_confounder_does_not_mutate_the_shared_instance():
    a(Trial)(treatment=confounder, season=confounder)
    assert confounder._type_ is None
