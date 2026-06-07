"""Tests for EQL rule-tree DAG serialization: persist as Python module and load it back."""

import os
import tempfile
import unittest

from krrood.entity_query_language.factories import (
    add,
    entity,
    not_,
    refinement,
    variable,
)
from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.interface import FunctionInterface
from krrood.entity_query_language.rdr.corner_case import CornerCaseStore
from krrood.entity_query_language.rdr.serialization import (
    load_rdr,
    rdr_to_python,
    save_rdr,
    walk_rules_in_emission_order,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.utils import UNSET

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()


def first(sp: Species) -> Animal:
    return next(a for a, t in zip(animals, targets) if t is sp)


def scripted_expert(rules) -> Expert:
    def answer(context, requests):
        return {"conditions": rules[context.target_conclusion](context.case_variable)}

    return Expert(interface=FunctionInterface(answer_fn=answer))


def _build_alternative_tree() -> EQLSingleClassRDR:
    """milk->mammal ; (no milk) feathers->bird ; (no milk) fins->fish — alternatives."""
    rdr = EQLSingleClassRDR(Animal, "species")
    expert = scripted_expert(
        {
            Species.mammal: lambda v: v.milk == True,
            Species.bird: lambda v: v.feathers == True,
            Species.fish: lambda v: v.fins == True,
        }
    )
    rdr.fit_case(first(Species.mammal), Species.mammal, expert)
    rdr.fit_case(first(Species.bird), Species.bird, expert)
    rdr.fit_case(first(Species.fish), Species.fish, expert)
    return rdr


def _build_refinement_tree() -> EQLSingleClassRDR:
    """backbone->fish (over-general) ; refine milk->mammal."""
    rdr = EQLSingleClassRDR(Animal, "species")
    expert = scripted_expert(
        {
            Species.fish: lambda v: v.backbone == True,
            Species.mammal: lambda v: v.milk == True,
        }
    )
    rdr.fit_case(first(Species.fish), Species.fish, expert)
    rdr.fit_case(first(Species.mammal), Species.mammal, expert)
    return rdr


def _build_sibling_refinement_tree() -> EQLSingleClassRDR:
    """backbone->fish, then refine the *same* fish rule twice (milk->mammal, feathers->bird).

    Two refinements share one anchor, so the serializer's per-selector ordering is what is
    under test: a blanket reverse would swap these siblings on every round-trip.
    """
    rdr = EQLSingleClassRDR(Animal, "species")
    expert = scripted_expert(
        {
            Species.fish: lambda v: v.backbone == True,
            Species.mammal: lambda v: v.milk == True,
            Species.bird: lambda v: v.feathers == True,
        }
    )
    rdr.fit_case(first(Species.fish), Species.fish, expert)
    rdr.fit_case(first(Species.mammal), Species.mammal, expert)
    rdr.fit_case(first(Species.bird), Species.bird, expert)
    return rdr


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestSerialization(unittest.TestCase):
    def test_empty_rdr_cannot_be_serialized(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        with self.assertRaises(ValueError):
            rdr_to_python(rdr)

    def test_source_uses_alternative_for_non_firing_rules(self):
        rdr = _build_alternative_tree()
        src = rdr_to_python(rdr)
        self.assertIn("with alternative(", src)
        self.assertIn("animal.milk == True", src)
        self.assertIn("Species.mammal", src)
        # Imports both the case type and the conclusion enum.
        self.assertIn("import Animal", src)
        self.assertIn("Species", src)

    def test_source_uses_refinement_for_overriding_rule(self):
        rdr = _build_refinement_tree()
        src = rdr_to_python(rdr)
        self.assertIn("with refinement(", src)

    def test_roundtrip_alternative_tree_preserves_classifications(self):
        rdr = _build_alternative_tree()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)
        for a, t in zip(animals, targets):
            self.assertEqual(rdr.classify(a), loaded.classify(a), a.name)

    def test_roundtrip_refinement_tree_preserves_classifications(self):
        rdr = _build_refinement_tree()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)
        for a, t in zip(animals, targets):
            self.assertEqual(rdr.classify(a), loaded.classify(a), a.name)

    def test_loaded_rdr_can_be_grown_further(self):
        rdr = _build_alternative_tree()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)

        # A reptile (backbone, no milk/feathers/fins) does not fire yet.
        reptile = first(Species.reptile)
        self.assertTrue(loaded.classify(reptile) is UNSET)
        expert = scripted_expert(
            {
                Species.reptile: lambda v: (v.backbone == True),
            }
        )
        # Fit on the loaded tree; it should grow and classify the reptile.
        loaded.fit_case(reptile, Species.reptile, expert)
        self.assertEqual(loaded.classify(reptile), Species.reptile)

    def test_double_roundtrip_is_stable(self):
        rdr = _build_alternative_tree()
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "m1.py")
            save_rdr(rdr, p1)
            loaded1 = load_rdr(p1)
            p2 = os.path.join(d, "m2.py")
            src1 = save_rdr(loaded1, p2)
            loaded2 = load_rdr(p2)
            # Source emitted from the reloaded model matches the file it was loaded from.
            with open(p2) as f:
                self.assertEqual(f.read(), src1)
        for a in animals:
            self.assertEqual(loaded1.classify(a), loaded2.classify(a), a.name)

    def test_sibling_refinements_emit_two_blocks(self):
        rdr = _build_sibling_refinement_tree()
        src = rdr_to_python(rdr)
        self.assertEqual(src.count("with refinement("), 2)

    def test_sibling_refinement_roundtrip_preserves_classifications(self):
        rdr = _build_sibling_refinement_tree()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)
        for a in animals:
            self.assertEqual(rdr.classify(a), loaded.classify(a), a.name)

    def test_sibling_refinement_roundtrip_is_byte_stable(self):
        # Regression: sibling refinements must not flip order on a round-trip. A blanket
        # reverse made accuracy oscillate (101 -> 91 -> 101 ...) across save/load.
        rdr = _build_sibling_refinement_tree()
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "m1.py")
            save_rdr(rdr, p1)
            loaded1 = load_rdr(p1)
            src2 = rdr_to_python(loaded1)
            with open(p1) as f:
                self.assertEqual(f.read(), src2)


def _build_chained_refinement_tree() -> EQLSingleClassRDR:
    """Build an RDR with two levels of chained refinements: backbone → fish,
    except if not_(milk) → molusc, except if venomous → reptile.

    The outer refinement uses ``not_()`` as its condition, covering the
    path where a negated refinement condition must still propagate the
    parent rule's satisfaction through the result chain.
    """
    animal_var = variable(Animal, domain=[])
    query = entity(animal_var).where(animal_var.backbone == True)
    with query:
        add(animal_var.species, Species.fish)
        with refinement(not_(animal_var.milk)):
            add(animal_var.species, Species.molusc)
            with refinement(animal_var.venomous):
                add(animal_var.species, Species.reptile)
    query.build()
    rdr = EQLSingleClassRDR(Animal, "species")
    rdr.case_variable = animal_var
    rdr.conclusion_variable = animal_var.species
    rdr.query = query
    return rdr


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestChainedRefinement(unittest.TestCase):
    """
    Regression tests for chained (multi-level) refinements, especially those
    whose conditions use ``not_()`` — the pattern that caused a missing rule
    in the human-fitted zoo model (scorpion classified as reptile instead of molusc).
    """

    def test_chained_refinement_roundtrip_preserves_classifications(self):
        rdr = _build_chained_refinement_tree()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)
        for a in animals:
            self.assertEqual(rdr.classify(a), loaded.classify(a), a.name)

    def test_chained_refinement_roundtrip_is_byte_stable(self):
        rdr = _build_chained_refinement_tree()
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "m1.py")
            save_rdr(rdr, p1)
            loaded1 = load_rdr(p1)
            src2 = rdr_to_python(loaded1)
            with open(p1) as f:
                self.assertEqual(f.read(), src2)

    def test_chained_refinement_emits_nested_blocks(self):
        rdr = _build_chained_refinement_tree()
        src = rdr_to_python(rdr)
        self.assertGreaterEqual(src.count("with refinement("), 2)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# CornerCase serialization round-trip
# ---------------------------------------------------------------------------


import pytest


def _build_one_rule_rdr_for_cc() -> EQLSingleClassRDR:
    """Smallest fitted RDR that records a corner case: one rule (milk -> mammal)."""
    rdr = EQLSingleClassRDR(Animal, "species")
    expert = scripted_expert({Species.mammal: lambda v: v.milk == True})
    rdr.fit_case(first(Species.mammal), Species.mammal, expert)
    return rdr


def _build_three_rule_rdr_for_cc() -> EQLSingleClassRDR:
    """Three-rule RDR (mammal / bird / fish alternatives) — each has a corner case."""
    rdr = EQLSingleClassRDR(Animal, "species")
    expert = scripted_expert(
        {
            Species.mammal: lambda v: v.milk == True,
            Species.bird: lambda v: v.feathers == True,
            Species.fish: lambda v: v.fins == True,
        }
    )
    rdr.fit_case(first(Species.mammal), Species.mammal, expert)
    rdr.fit_case(first(Species.bird), Species.bird, expert)
    rdr.fit_case(first(Species.fish), Species.fish, expert)
    return rdr


@pytest.mark.skipif(len(animals) == 0, reason="Failed to load zoo dataset")
class TestCornerCaseSerialization:
    """Serialization round-trip tests for ``CornerCaseStore``."""

    def test_rdr_corner_cases_key_present_in_emitted_source(self):
        """``rdr_to_python`` must emit a ``RDR_CORNER_CASES`` assignment in the source.

        Guarantees that the serializer generates the module-level dict that the
        loader expects to read back.
        """
        rdr = _build_one_rule_rdr_for_cc()
        src = rdr_to_python(rdr)
        assert "RDR_CORNER_CASES" in src

    def test_rdr_corner_cases_contains_case_constructor_source(self):
        """The emitted source must contain a constructor call for the case type.

        Concretely, ``Animal(`` must appear inside the ``RDR_CORNER_CASES`` block so
        the loaded module can reconstruct the original ``Animal`` instance.
        """
        rdr = _build_one_rule_rdr_for_cc()
        src = rdr_to_python(rdr)
        # The case type is Animal; its constructor appears in the RDR_CORNER_CASES dict.
        assert "Animal(" in src

    def test_save_and_load_restores_corner_case_store_count(self):
        """After a save/load round-trip the loaded store has the same number of entries.

        Guarantees that no corner cases are silently dropped during serialization or
        deserialization.
        """
        rdr = _build_one_rule_rdr_for_cc()
        original_count = len(rdr.corner_cases.cases)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)
        assert len(loaded.corner_cases.cases) == original_count

    def test_save_and_load_corner_case_equals_original(self):
        """After a round-trip, ``corner_cases.get(node_id)`` equals the original case.

        Equality is by value (``==``), not identity (``is``), because the loaded
        instance is a freshly reconstructed dataclass.
        """
        rdr = _build_one_rule_rdr_for_cc()
        # Retrieve the original case stored for the single rule.
        ordered_before = walk_rules_in_emission_order(rdr.conditions_root)
        assert len(ordered_before) == 1
        original_case = rdr.corner_cases.get(ordered_before[0]._id_)
        assert original_case is not None, "fit_case must have recorded a corner case"

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)

        ordered_after = walk_rules_in_emission_order(loaded.conditions_root)
        assert len(ordered_after) == 1
        restored_case = loaded.corner_cases.get(ordered_after[0]._id_)
        assert restored_case is not None, "load_rdr must rebuild the corner case"
        assert restored_case == original_case

    def test_save_and_load_all_rules_have_corner_cases(self):
        """After fitting 3 rules and doing a round-trip, all 3 nodes have corner cases.

        Guarantees that the positional index mapping covers every rule in the tree,
        not only the first or the last.
        """
        rdr = _build_three_rule_rdr_for_cc()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr(rdr, path)
            loaded = load_rdr(path)

        ordered = walk_rules_in_emission_order(loaded.conditions_root)
        assert len(ordered) == 3, "Three rules must have been loaded"
        for node in ordered:
            assert (
                loaded.corner_cases.get(node._id_) is not None
            ), f"Node {node._id_} has no corner case in the loaded store"

    def test_load_old_file_without_corner_cases_gives_empty_store(self):
        """Loading a file that has no ``RDR_CORNER_CASES`` gives an empty store.

        Backward-compatibility contract: old files load without error; the store is
        empty when no corner-case block is present.
        """
        legacy_path = os.path.join(
            os.path.dirname(__file__),
            "fitted_models",
            "zoo_species_rdr_no_corner_cases.py",
        )
        loaded = load_rdr(legacy_path)
        assert isinstance(loaded.corner_cases, CornerCaseStore)
        assert len(loaded.corner_cases.cases) == 0
