"""Integration tests: ``EQLSingleClassRDR`` populates ``corner_cases`` during fitting."""

from __future__ import annotations

import dataclasses

import pytest

from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.interface import FunctionInterface
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.serialization import walk_rules_in_emission_order

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()

FEATURE_FIELDS = [
    f.name for f in dataclasses.fields(Animal) if f.name not in ("name", "species")
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_animal(
    name: str,
    *,
    milk: bool = False,
    feathers: bool = False,
    fins: bool = False,
    backbone: bool = True,
    venomous: bool = False,
) -> Animal:
    """Pattern: DistinctAnimalCase — minimal animal with a unique discriminating feature."""
    return Animal(
        name=name,
        hair=milk,
        feathers=feathers,
        eggs=not milk,
        milk=milk,
        airborne=False,
        aquatic=fins,
        predator=False,
        toothed=backbone,
        backbone=backbone,
        breathes=not fins,
        venomous=venomous,
        fins=fins,
        legs=0 if fins else 4,
        tail=backbone,
        domestic=False,
        catsize=milk,
    )


def _scripted_expert(rules: dict) -> Expert:
    """A programmatic expert that returns the condition lambda for ``context.target_conclusion``.

    ``rules`` maps a target ``Species`` value to a callable ``(case_variable) -> condition``.
    """

    def answer(context, requests):
        return {"conditions": rules[context.target_conclusion](context.case_variable)}

    return Expert(interface=FunctionInterface(answer_fn=answer))


def _first_recorded_node_id(rdr: EQLSingleClassRDR):
    """Return the ``_id_`` of the first (root) condition node in emission order."""
    nodes = walk_rules_in_emission_order(rdr.conditions_root)
    assert nodes, "RDR must have at least one rule"
    return nodes[0]._id_


# ---------------------------------------------------------------------------
# Test 1 — first rule inserts one corner case entry
# ---------------------------------------------------------------------------


def test_first_rule_corner_case_is_recorded():
    """After fitting the first rule, exactly one corner case is stored for the root node."""
    rdr = EQLSingleClassRDR(Animal, "species")
    mammal = _make_animal("mammal", milk=True)
    expert = _scripted_expert({Species.mammal: lambda v: v.milk == True})

    rdr.fit_case(mammal, Species.mammal, expert)

    assert len(rdr.corner_cases.cases) == 1
    root_node_id = _first_recorded_node_id(rdr)
    assert rdr.corner_cases.get(root_node_id) is mammal


# ---------------------------------------------------------------------------
# Test 2 — alternative rule adds a second corner case entry
# ---------------------------------------------------------------------------


def test_alternative_rule_corner_case_is_recorded():
    """Fitting a second, non-firing case grows an alternative; the store has 2 entries."""
    rdr = EQLSingleClassRDR(Animal, "species")
    mammal = _make_animal("mammal", milk=True)
    bird = _make_animal("bird", feathers=True)
    expert = _scripted_expert(
        {
            Species.mammal: lambda v: v.milk == True,
            Species.bird: lambda v: v.feathers == True,
        }
    )

    rdr.fit_case(mammal, Species.mammal, expert)
    rdr.fit_case(bird, Species.bird, expert)

    assert len(rdr.corner_cases.cases) == 2
    # The node whose corner case is the bird instance is the alternative node.
    nodes = walk_rules_in_emission_order(rdr.conditions_root)
    assert len(nodes) == 2
    alternative_node_id = nodes[1]._id_
    assert rdr.corner_cases.get(alternative_node_id) is bird


# ---------------------------------------------------------------------------
# Test 3 — refinement rule adds a third corner case entry for the trigger case
# ---------------------------------------------------------------------------


def test_refinement_rule_corner_case_is_recorded():
    """
    Fitting a case that triggers a wrong rule grows a refinement; the store grows to 3
    entries and the refinement node maps to the triggering case instance.
    """
    rdr = EQLSingleClassRDR(Animal, "species")
    # Over-general rule: backbone -> fish. A mammal (backbone + milk) misfires.
    fish = _make_animal("fish", fins=True, backbone=False)
    bird = _make_animal("bird", feathers=True, backbone=False)
    mammal = _make_animal("mammal", milk=True, backbone=True)
    expert = _scripted_expert(
        {
            Species.fish: lambda v: v.fins == True,
            Species.bird: lambda v: v.feathers == True,
            Species.mammal: lambda v: v.milk == True,
        }
    )

    rdr.fit_case(fish, Species.fish, expert)
    rdr.fit_case(bird, Species.bird, expert)
    rdr.fit_case(mammal, Species.mammal, expert)

    assert len(rdr.corner_cases.cases) == 3
    nodes = walk_rules_in_emission_order(rdr.conditions_root)
    # The refinement node is the third in emission order.
    refinement_node_id = nodes[2]._id_
    assert rdr.corner_cases.get(refinement_node_id) is mammal


# ---------------------------------------------------------------------------
# Test 4 — fitting an already-correct case does not add a new corner case
# ---------------------------------------------------------------------------


def test_no_new_rule_means_no_new_corner_case():
    """Fitting a case that already classifies correctly leaves ``len(corner_cases.cases)`` unchanged."""
    rdr = EQLSingleClassRDR(Animal, "species")
    mammal = _make_animal("mammal", milk=True)
    expert = _scripted_expert({Species.mammal: lambda v: v.milk == True})

    rdr.fit_case(mammal, Species.mammal, expert)
    count_before = len(rdr.corner_cases.cases)

    rdr.fit_case(mammal, Species.mammal, expert)  # already correct

    assert len(rdr.corner_cases.cases) == count_before


# ---------------------------------------------------------------------------
# Test 5 — corner cases do not affect classification results
# ---------------------------------------------------------------------------


def test_corner_cases_do_not_affect_classification():
    """
    The classification result of ``rdr.classify(case)`` is identical before and after corner
    cases accumulate — corner-case data is pure side-data, not classification logic.
    """
    rdr = EQLSingleClassRDR(Animal, "species")
    mammal = _make_animal("mammal", milk=True)
    bird = _make_animal("bird", feathers=True)
    expert = _scripted_expert(
        {
            Species.mammal: lambda v: v.milk == True,
            Species.bird: lambda v: v.feathers == True,
        }
    )

    rdr.fit_case(mammal, Species.mammal, expert)
    result_before = rdr.classify(mammal)

    rdr.fit_case(bird, Species.bird, expert)

    result_after = rdr.classify(mammal)
    assert result_before == result_after == Species.mammal


# ---------------------------------------------------------------------------
# Test 6 — fit() for 3 independent cases populates all 3 corner cases
# ---------------------------------------------------------------------------


def test_fit_bulk_populates_all_corner_cases():
    """``rdr.fit(cases, targets, expert)`` for 3 cases that each require a new rule
    results in exactly 3 corner case entries."""
    rdr = EQLSingleClassRDR(Animal, "species")
    mammal = _make_animal("mammal", milk=True)
    bird = _make_animal("bird", feathers=True)
    fish = _make_animal("fish", fins=True, backbone=False)
    cases = [mammal, bird, fish]
    case_targets = [Species.mammal, Species.bird, Species.fish]
    expert = _scripted_expert(
        {
            Species.mammal: lambda v: v.milk == True,
            Species.bird: lambda v: v.feathers == True,
            Species.fish: lambda v: v.fins == True,
        }
    )

    rdr.fit(cases, case_targets, expert)

    assert len(rdr.corner_cases.cases) == 3
