"""
Phase 6 tests: persist the EQL rule-tree DAG as a Python module and load it back.

The serialized artifact is Python source (no JSON, no rule-as-string). Round-tripping
must preserve classifications exactly.
"""

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
from krrood.entity_query_language.rdr.serialization import (
    load_rdr,
    rdr_to_python,
    save_rdr,
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
