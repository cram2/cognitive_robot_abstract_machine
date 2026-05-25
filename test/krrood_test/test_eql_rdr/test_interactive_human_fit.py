"""
Human-in-the-loop fitting of the zoo dataset (SKIPPED by default).

``TestFitZooAsHumanExpert`` opens a **real embedded IPython shell** and asks *you* — the
human expert — to author each rule's conditions, then saves the rule tree it learns next
to these tests so it can be reloaded as a fitted model. It is skipped unless you opt in::

    EQL_RDR_INTERACTIVE=1 pytest -s \\
        test/krrood_test/test_eql_rdr/test_interactive_human_fit.py::TestFitZooAsHumanExpert

``-s`` is required so pytest does not capture stdin/stdout (the shell needs the terminal).

For each animal the RDR cannot yet classify correctly, a shell opens with the case (as a
table), the current (wrong/missing) conclusion, the target species, the ``case_variable``
EQL variable, the concrete ``case_instance`` and the EQL factories in scope. Write a
condition over ``case_variable`` and assign it to ``conditions``, e.g.::

    conditions = case_variable.milk == True

then exit the shell (Ctrl-D). Because the RDR only prompts on misclassification, good
general rules keep the number of prompts small. The learned tree is written to
``fitted_models/zoo_species_rdr.py``; commit it to make ``TestLoadHumanFittedModel`` run.

``TestLoadHumanFittedModel`` is NOT interactive: it runs automatically once that model
file exists, loading it and using it as a fit model (directly and via the underspecified
RDR backend).
"""

import os
import unittest

from krrood.entity_query_language.factories import underspecified
from krrood.entity_query_language.rdr.backend import RDRBackend
from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.interactive import IPythonInterface
from krrood.entity_query_language.rdr.serialization import (
    load_rdr,
    rdr_to_python,
    save_rdr,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()

#: Where the human-authored rule tree is saved, alongside these tests.
FITTED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "fitted_models")
SAVED_MODEL_PATH = os.path.join(FITTED_MODELS_DIR, "zoo_species_rdr.py")


def _ipython_available() -> bool:
    try:
        import IPython  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(
    False,
    "human-interactive: set to True and run with `pytest -s`",
)
@unittest.skipUnless(_ipython_available(), "IPython not installed")
@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestFitZooAsHumanExpert(unittest.TestCase):
    def test_fit_and_save(self):
        # Ground-truth fitting: you (the expert) supply only the conditions; the species is
        # the known target. The RDR prompts you only when it would misclassify a case.
        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(animals, targets, Expert(interface=IPythonInterface()))

        os.makedirs(FITTED_MODELS_DIR, exist_ok=True)
        save_rdr(rdr, SAVED_MODEL_PATH)

        correct = sum(rdr.classify(a) == t for a, t in zip(animals, targets))
        print(f"\n[interactive] accuracy on fitted set: {correct}/{len(animals)}")
        print(f"[interactive] saved learned rule tree to: {SAVED_MODEL_PATH}")
        print("[interactive] commit it to enable TestLoadHumanFittedModel.\n")
        print(rdr_to_python(rdr))

        self.assertTrue(os.path.exists(SAVED_MODEL_PATH))


@unittest.skipUnless(
    os.path.exists(SAVED_MODEL_PATH),
    "no human-fitted model saved yet (run TestFitZooAsHumanExpert first)",
)
@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestLoadHumanFittedModel(unittest.TestCase):
    def test_loaded_model_is_usable_as_a_fit_model(self):
        rdr = load_rdr(SAVED_MODEL_PATH)
        self.assertIs(rdr.case_type, Animal)
        self.assertEqual(rdr.conclusion_attribute_name, "species")

        # Usable through the underspecified backend with no further fitting.
        backend = RDRBackend()
        backend.models[(Animal, "species")] = rdr
        query = underspecified(Animal, domain=animals)(species=...)
        inferred = [r[query.variable.species] for r in backend.infer(query)]

        # The backend reproduces the model's direct classifications exactly.
        self.assertEqual(inferred, [rdr.classify(a) for a in animals])

        correct = sum(1 for value, t in zip(inferred, targets) if value == t)
        print(f"\n[loaded model] accuracy on zoo set: {correct}/{len(animals)}")
        # The saved model must reproduce its fitted accuracy exactly: a serialization
        # round-trip that reorders sibling rules silently degraded this (101 -> 91).
        self.assertEqual(correct, len(animals))


if __name__ == "__main__":
    unittest.main()
