"""
Phase 7 tests: the RDR backend for underspecified (``...``) EQL queries.

An underspecified query marks the attribute to infer with ``...``, optionally filters with
concrete kwargs, and supplies a domain of existing instances. The backend infers the
``...`` attribute on each matching instance — lazily, yielding ``UnificationDict``s by
default (or filling instances in place) — mirroring ordinary EQL evaluation.
"""

import dataclasses
import os
import tempfile
import types
import unittest

from typing_extensions import List, Optional

from krrood.entity_query_language.factories import and_, underspecified
from krrood.entity_query_language.core.base_expressions import UnificationDict
from krrood.entity_query_language.rdr.backend import RDRBackend
from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.interface import FunctionInterface
from krrood.entity_query_language.rdr.serialization import load_rdr, save_rdr
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.underspecified import (
    MultipleInferenceTargets,
    NoInferenceTarget,
    UnderspecifiedMatch,
    UnsupportedInferenceTarget,
)

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()
target_by_name = {a.name: t for a, t in zip(animals, targets)}

FEATURE_FIELDS = [
    f.name for f in dataclasses.fields(Animal) if f.name not in ("name", "species")
]

# Small slice of milk-bearing animals reused by constant-ground-truth tests.
_SMALL_MILK_ANIMALS = [a for a in animals if a.milk][:5]

# Shared cache for the expensive full-zoo fit (filled once by setUpModule).
_ZOO_MODEL_CACHE: Optional[EQLSingleClassRDR] = None


def first(sp: Species) -> Animal:
    return next(a for a, t in zip(animals, targets) if t is sp)


@dataclasses.dataclass
class Bag:
    """A type whose inferred attribute would be an unbounded iterable (single-class can't)."""

    items: List[str]
    label: Optional[str] = None


def _full_feature_conditions(context):
    v, case = context.case_variable, context.case_instance
    return and_(*[getattr(v, f) == getattr(case, f) for f in FEATURE_FIELDS])


def maximally_specific_expert() -> Expert:
    """Conditions = the full feature vector; guarantees convergence on the training set."""

    def answer(context, requests):
        return {"conditions": _full_feature_conditions(context)}

    return Expert(interface=FunctionInterface(answer_fn=answer))


def labelling_expert() -> Expert:
    """Also supplies the conclusion (from ground truth) when no target is given."""

    def answer(context, requests):
        result = {"conditions": _full_feature_conditions(context)}
        if any(r.name == "conclusion" for r in requests):
            result["conclusion"] = target_by_name[context.case_instance.name]
        return result

    return Expert(interface=FunctionInterface(answer_fn=answer))


def setUpModule():
    """Fit the full-zoo species model once for the whole module."""
    global _ZOO_MODEL_CACHE
    if not animals:
        return
    backend = RDRBackend(expert=maximally_specific_expert())
    backend.fit(
        underspecified(Animal, domain=animals)(species=...),
        ground_truth=lambda a: target_by_name[a.name],
    )
    _ZOO_MODEL_CACHE = backend.models.get((Animal, "species"))


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestUnderspecifiedMatchAdapter(unittest.TestCase):
    def test_discovers_single_inference_target(self):
        stmt = UnderspecifiedMatch(underspecified(Animal, domain=animals)(species=...))
        self.assertEqual(stmt.target_attribute_name, "species")
        self.assertEqual(stmt.case_type, Animal)

    def test_concrete_kwargs_filter_ellipsis_stripped(self):
        # milk=True filters the domain; species=... must NOT filter (it is the target).
        stmt = UnderspecifiedMatch(
            underspecified(Animal, domain=animals)(milk=True, species=...)
        )
        cases = list(stmt.filtered_cases())
        self.assertEqual(len(cases), sum(1 for a in animals if a.milk))
        self.assertTrue(all(c.milk for c in cases))

    def test_no_filter_streams_whole_domain(self):
        stmt = UnderspecifiedMatch(underspecified(Animal, domain=animals)(species=...))
        self.assertEqual(len(list(stmt.filtered_cases())), len(animals))

    def test_no_inference_target_raises(self):
        stmt = UnderspecifiedMatch(underspecified(Animal, domain=animals)(milk=True))
        with self.assertRaises(NoInferenceTarget):
            stmt.single_target()

    def test_multiple_inference_targets_raises(self):
        stmt = UnderspecifiedMatch(
            underspecified(Animal, domain=animals)(species=..., legs=...)
        )
        with self.assertRaises(MultipleInferenceTargets):
            stmt.single_target()

    def test_unbounded_iterable_target_rejected(self):
        stmt = UnderspecifiedMatch(underspecified(Bag, domain=[])(items=...))
        with self.assertRaises(UnsupportedInferenceTarget):
            stmt.single_target()


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestFromUnderspecified(unittest.TestCase):
    def test_derives_case_type_and_attribute(self):
        rdr = EQLSingleClassRDR.from_underspecified(underspecified(Animal)(species=...))
        self.assertIs(rdr.case_type, Animal)
        self.assertEqual(rdr.conclusion_attribute_name, "species")


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestRDRBackendInfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._fitted_model = _ZOO_MODEL_CACHE

    def _fitted_backend(self):
        backend = RDRBackend()
        backend.models[(Animal, "species")] = self._fitted_model
        return backend

    def test_infer_is_lazy(self):
        backend = self._fitted_backend()
        gen = backend.infer(underspecified(Animal, domain=animals)(species=...))
        self.assertIsInstance(gen, types.GeneratorType)

    def test_infer_yields_unification_dicts_by_default(self):
        backend = self._fitted_backend()
        query = underspecified(Animal, domain=animals)(species=...)
        results = list(backend.infer(query))
        self.assertTrue(all(isinstance(r, UnificationDict) for r in results))
        # Inferred values readable via the query's attribute; instances unchanged.
        correct = sum(
            1 for r, t in zip(results, targets) if r[query.variable.species] == t
        )
        self.assertEqual(correct, len(animals))
        self.assertEqual(results[0][query.variable].name, animals[0].name)
        self.assertTrue(all(a.species is None for a in animals))

    def test_fill_in_place_mutates_and_yields_instances(self):
        backend = self._fitted_backend()
        subset = animals[:5]
        query = underspecified(Animal, domain=subset)(species=...)
        yielded = list(backend.infer(query, fill_in_place=True))
        self.assertEqual([a.name for a in yielded], [a.name for a in subset])
        for a in subset:
            self.assertEqual(a.species, target_by_name[a.name])
        # restore so other tests see pristine instances
        for a in subset:
            a.species = None

    def test_infer_respects_concrete_filter(self):
        backend = self._fitted_backend()
        query = underspecified(Animal, domain=animals)(milk=True, species=...)
        results = list(backend.infer(query))
        self.assertEqual(len(results), sum(1 for a in animals if a.milk))
        self.assertTrue(
            all(r[query.variable.species] == Species.mammal for r in results)
        )


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestRDRBackendFitModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _SMALL_MILK_ANIMALS:
            return
        backend = RDRBackend(expert=maximally_specific_expert())
        backend.fit(
            underspecified(Animal, domain=_SMALL_MILK_ANIMALS)(milk=True, species=...),
            ground_truth=Species.mammal,
        )
        cls._mammal_backend = backend

    def test_constant_ground_truth(self):
        # Every milk-bearing animal is a mammal: a single ground-truth value for the subset.
        out = underspecified(Animal, domain=_SMALL_MILK_ANIMALS)(milk=True, species=...)
        self.assertTrue(
            all(r[out.variable.species] == Species.mammal for r in self._mammal_backend.infer(out))
        )

    def test_auto_fit_mode_without_ground_truth_uses_ask_for_rule(self):
        # Fresh backend, no model -> infer triggers fit mode; expert labels via ask_for_rule.
        backend = RDRBackend(expert=labelling_expert())
        query = underspecified(Animal, domain=animals[:12])(species=...)
        results = list(backend.infer(query))
        for r, (a, t) in zip(results, list(zip(animals, targets))[:12]):
            self.assertEqual(r[query.variable.species], t, a.name)

    def test_missing_expert_without_ground_truth_errors(self):
        backend = RDRBackend(expert=None)
        with self.assertRaises((ValueError, NotImplementedError)):
            list(backend.infer(underspecified(Animal, domain=animals[:3])(species=...)))


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestUnderspecifiedEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._fitted_model = _ZOO_MODEL_CACHE

    def test_fit_save_load_infer(self):
        model = self._fitted_model
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "zoo_rdr.py")
            save_rdr(model, path)
            loaded = load_rdr(path)

        original_backend = RDRBackend()
        original_backend.models[(Animal, "species")] = model
        reloaded_backend = RDRBackend()
        reloaded_backend.models[(Animal, "species")] = loaded

        q1 = underspecified(Animal, domain=animals)(species=...)
        q2 = underspecified(Animal, domain=animals)(species=...)
        before = [r[q1.variable.species] for r in original_backend.infer(q1)]
        after = [r[q2.variable.species] for r in reloaded_backend.infer(q2)]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
