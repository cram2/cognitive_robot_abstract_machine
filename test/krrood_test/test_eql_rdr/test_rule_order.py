"""Tests for ``walk_rules_in_emission_order`` — the canonical traversal order."""

from __future__ import annotations

import re

import pytest

from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import add, entity, variable
from krrood.entity_query_language.rdr.rule_tree import (
    insert_alternative,
    insert_refinement,
)
from krrood.entity_query_language.rdr.rule_tree_view import walk_rules
from krrood.entity_query_language.rdr.serialization import (
    rdr_to_python,
    walk_rules_in_emission_order,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rules.conclusion import Add

from .animal import Animal, Species

# ---------------------------------------------------------------------------
# Helpers — build minimal rule trees without going through the full RDR fit loop
# ---------------------------------------------------------------------------


def _animal_var():
    """Return a fresh Animal variable (new query-scope each call)."""
    return variable(Animal, domain=[])


def _single_rule_query():
    """Build a query with exactly one rule: milk == True -> mammal."""
    animal = _animal_var()
    query = entity(animal).where(animal.milk == True)
    with query:
        add(animal.species, Species.mammal)
    query.build()
    return query, animal


def _all_alternatives_query():
    """Build a query with three rules all as alternatives.

    Insertion order: A (milk/mammal), B (feathers/bird), D (fins/fish).
    """
    animal = _animal_var()
    query = entity(animal).where(animal.milk == True)
    with query:
        add(animal.species, Species.mammal)  # A
    query.build()
    node_b = insert_alternative(
        query._conditions_root_,
        animal.feathers == True,
        animal.species,
        Species.bird,
    )  # B
    node_d = insert_alternative(
        query._conditions_root_,
        animal.fins == True,
        animal.species,
        Species.fish,
    )  # D
    return query, animal, node_b, node_d


def _all_refinements_query():
    """Build a chained refinement query: A (backbone/fish) -> C (milk/mammal) -> E (feathers/bird).

    Each new refinement is anchored on the previous rule's condition node.
    """
    animal = _animal_var()
    query = entity(animal).where(animal.backbone == True)
    with query:
        add(animal.species, Species.fish)  # A
    query.build()
    node_c = insert_refinement(
        query._conditions_root_,
        animal.milk == True,
        animal.species,
        Species.mammal,
    )  # C
    node_e = insert_refinement(
        node_c,
        animal.feathers == True,
        animal.species,
        Species.bird,
    )  # E
    return query, animal, node_c, node_e


def _mixed_alt_then_ref_query():
    """Build the key divergence tree: A (backbone/fish), B (feathers/bird alt of A),
    C (milk/mammal refinement of A).

    The resulting DAG is Refinement(Alternative(A, B), C) — or similar left-nested form.
    walk_rules (display) order: A, B, C.
    _emit_rule_body (emission) order: A, C, B.
    """
    animal = _animal_var()
    query = entity(animal).where(animal.backbone == True)
    with query:
        add(animal.species, Species.fish)  # A
    query.build()
    insert_alternative(
        query._conditions_root_,
        animal.feathers == True,
        animal.species,
        Species.bird,
    )  # B — alternative of A
    insert_refinement(
        query._conditions_root_,
        animal.milk == True,
        animal.species,
        Species.mammal,
    )  # C — refinement of A
    return query, animal


# ---------------------------------------------------------------------------
# Test 1: Single rule
# ---------------------------------------------------------------------------


def test_single_rule_returns_list_of_length_one():
    """``walk_rules_in_emission_order`` on a one-rule tree returns exactly one node."""
    query, _animal = _single_rule_query()
    result = walk_rules_in_emission_order(query._conditions_root_)
    assert len(result) == 1


def test_single_rule_contains_the_condition_node():
    """The single element returned is the root condition node itself."""
    query, _animal = _single_rule_query()
    root = query._conditions_root_
    result = walk_rules_in_emission_order(root)
    assert result[0]._id_ == root._id_


# ---------------------------------------------------------------------------
# Test 2: All alternatives — insertion order preserved
# ---------------------------------------------------------------------------


def test_all_alternatives_returns_three_nodes():
    """Three alternatives produce a list of exactly three condition nodes."""
    query, _animal, _b, _d = _all_alternatives_query()
    result = walk_rules_in_emission_order(query._conditions_root_)
    assert len(result) == 3


def test_all_alternatives_first_node_is_first_inserted():
    """The first element of the emission list is rule A — the first inserted rule."""
    query, animal, _b, _d = _all_alternatives_query()
    # The first rule's condition is milk == True.  Use walk_rules to find it (A is [0]).
    display = walk_rules(query._conditions_root_)
    result = walk_rules_in_emission_order(query._conditions_root_)
    assert result[0]._id_ == display[0].condition._id_


def test_all_alternatives_emission_order_matches_insertion_order():
    """Emission order for all-alternative tree is the same as walk_rules display order.

    When there are no refinements, Alternative chains are reversed by ``_orient_run``
    back to insertion order, which happens to coincide with display order.
    """
    query, _animal, _b, _d = _all_alternatives_query()
    display_ids = [r.condition._id_ for r in walk_rules(query._conditions_root_)]
    emission_ids = [
        n._id_ for n in walk_rules_in_emission_order(query._conditions_root_)
    ]
    assert emission_ids == display_ids


# ---------------------------------------------------------------------------
# Test 3: All refinements — insertion order preserved (no reversal)
# ---------------------------------------------------------------------------


def test_all_refinements_returns_three_nodes():
    """A three-rule refinement chain produces a list of exactly three condition nodes."""
    query, _animal, _c, _e = _all_refinements_query()
    result = walk_rules_in_emission_order(query._conditions_root_)
    assert len(result) == 3


def test_all_refinements_emission_order_matches_insertion_order():
    """Emission order for a pure refinement chain is A, C, E — insertion order.

    Refinement chains grow inward; ``_orient_run`` does not reverse them, so
    ``_decompose`` already yields them in insertion order.
    """
    query, _animal, node_c, node_e = _all_refinements_query()
    display = walk_rules(query._conditions_root_)
    result = walk_rules_in_emission_order(query._conditions_root_)
    # Display order for refinements is also A, C, E — both orderings agree here.
    display_ids = [r.condition._id_ for r in display]
    emission_ids = [n._id_ for n in result]
    assert emission_ids == display_ids


# ---------------------------------------------------------------------------
# Test 4: Mixed tree — the key divergence case
# ---------------------------------------------------------------------------


def test_mixed_tree_returns_three_nodes():
    """The mixed Alternative+Refinement tree contains exactly three condition nodes."""
    query, _animal = _mixed_alt_then_ref_query()
    result = walk_rules_in_emission_order(query._conditions_root_)
    assert len(result) == 3


def test_mixed_tree_emission_first_node_is_base_rule():
    """In the mixed tree, the first emission node is the base rule A (backbone/fish)."""
    query, _animal = _mixed_alt_then_ref_query()
    display = walk_rules(query._conditions_root_)
    result = walk_rules_in_emission_order(query._conditions_root_)
    # display[0] is A (base rule) — emission must also start with A
    assert result[0]._id_ == display[0].condition._id_


def test_mixed_tree_emission_order_is_A_C_B():
    """Emission order for Refinement(Alternative(A, B), C) is A, C, B.

    ``_emit_rule_body`` calls ``_decompose`` on the root: it peels off the
    Alternative-of-B then the Refinement-of-C branch.  ``_orient_run`` reverses
    the Alternative run, placing C (refinement) before B (alternative) in the branch
    list.  The serializer emits the base A first, then walks each branch — C then B.
    """
    query, _animal = _mixed_alt_then_ref_query()
    display = walk_rules(query._conditions_root_)
    # display order: A[0], B[1], C[2]
    A_id = display[0].condition._id_
    B_id = display[1].condition._id_
    C_id = display[2].condition._id_

    result = walk_rules_in_emission_order(query._conditions_root_)
    emission_ids = [n._id_ for n in result]

    assert emission_ids == [A_id, C_id, B_id]


def test_mixed_tree_emission_order_differs_from_display_order():
    """Emission order must NOT equal display (walk_rules) order for the mixed tree.

    This is the core invariant: walk_rules gives A, B, C; emission gives A, C, B.
    If these were equal it would mean the two walkers agree — which is the bug
    Phase 0 is designed to prevent.
    """
    query, _animal = _mixed_alt_then_ref_query()
    display_ids = [r.condition._id_ for r in walk_rules(query._conditions_root_)]
    emission_ids = [
        n._id_ for n in walk_rules_in_emission_order(query._conditions_root_)
    ]
    assert emission_ids != display_ids


# ---------------------------------------------------------------------------
# Test 5: Emission order == serialization order (integration guard)
# ---------------------------------------------------------------------------


def _build_mixed_rdr() -> EQLSingleClassRDR:
    """Return an RDR with at least one refinement and one alternative, for integration tests."""
    rdr = EQLSingleClassRDR(Animal, "species")
    animal = rdr.case_variable
    query = entity(animal).where(animal.backbone == True)
    with query:
        add(animal.species, Species.fish)
    query.build()
    insert_alternative(
        query._conditions_root_,
        animal.feathers == True,
        animal.species,
        Species.bird,
    )
    insert_refinement(
        query._conditions_root_,
        animal.milk == True,
        animal.species,
        Species.mammal,
    )
    rdr.query = query
    rdr.conclusion_variable = animal.species
    return rdr


def test_emission_order_count_matches_add_occurrences_in_source():
    """The number of nodes in ``walk_rules_in_emission_order`` equals the number of
    ``add(`` calls emitted by ``rdr_to_python``.

    This is the weakest integration guarantee: the walker and the emitter visit the
    same number of rules.  A mismatch would mean one of them skips or double-counts
    a node.
    """
    rdr = _build_mixed_rdr()
    src = rdr_to_python(rdr)
    add_count_in_source = len(re.findall(r"\badd\(", src))
    emission_nodes = walk_rules_in_emission_order(rdr.query._conditions_root_)
    assert len(emission_nodes) == add_count_in_source


def test_emission_order_i_th_node_matches_i_th_add_in_source():
    """The i-th node in ``walk_rules_in_emission_order`` corresponds to the i-th
    ``add(`` block in the emitted source, verified by checking that the conclusion
    value of the i-th emission node appears in the i-th add-block of the source.

    This pins the ordering contract: the walker and the emitter traverse rules
    in identical sequence.

    Note: we match conclusion *values* (enum names) rather than condition repr
    to avoid fragility from variable-name or attribute-path formatting differences.
    """
    rdr = _build_mixed_rdr()
    src = rdr_to_python(rdr)

    # Extract the Species.X argument from each add( call, in document order.
    # Each emitted add line looks like: add(animal.species, Species.mammal)
    species_pattern = re.compile(r"\badd\([^,]+,\s*(Species\.\w+)\)")
    emitted_values = species_pattern.findall(src)

    emission_nodes = walk_rules_in_emission_order(rdr.query._conditions_root_)

    assert len(emitted_values) == len(emission_nodes)

    for i, (node, emitted_value_str) in enumerate(zip(emission_nodes, emitted_values)):
        add_conclusions = [c for c in node._conclusions_ if isinstance(c, Add)]
        assert add_conclusions, f"Node at index {i} has no Add conclusion"
        conclusion_add = add_conclusions[0]
        target = conclusion_add.right
        node_value = target._value_ if isinstance(target, Literal) else target
        # node_value is a Species enum; emitted_value_str is e.g. "Species.mammal"
        expected_str = f"Species.{node_value.name}"
        assert expected_str == emitted_value_str, (
            f"Position {i}: emission walker says {expected_str!r} "
            f"but source has {emitted_value_str!r}"
        )
