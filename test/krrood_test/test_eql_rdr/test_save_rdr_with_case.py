"""
Phase 3 tests: ``save_path`` on ``EQLSingleClassRDR`` and ``save_rdr_with_case``.

These tests gate two implementation steps:

Step A — ``save_path`` field on ``EQLSingleClassRDR`` (``rdr/single_class.py``):
  * ``save_path=None`` (default): ``fit_case`` must NOT write any file.
  * ``save_path=<tmp>`` present: after ``fit_case`` a file exists at that path.
  * A second ``fit_case`` call triggers a second write (spy counter == 2).

Step B — ``save_rdr_with_case(rdr, path)`` in ``rdr/serialization.py``:
  * File produced by ``save_rdr_with_case`` round-trips via ``load_rdr``.
  * ``FunctionCase``-based RDR: file contains the dataclass class-header section.
  * Non-``FunctionCase`` RDR: function falls back to plain ``save_rdr`` (no header).
  * ``rdr_to_python(case_type_is_local=True)``: case type not re-imported in the
    rule-tree section.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

import pytest

from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.function_case import FunctionCase
from krrood.entity_query_language.rdr.interface import FunctionInterface
from krrood.entity_query_language.rdr.serialization import (
    load_rdr,
    rdr_to_python,
    save_rdr,
    save_rdr_with_case,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.utils import UNSET

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

# ---------------------------------------------------------------------------
# Shared test data (loaded once, reused by all tests in this module).
# ---------------------------------------------------------------------------

animals, targets = load_zoo_animals()


def first(sp: Species) -> Animal:
    """Return the first animal in the zoo dataset labelled with ``sp``."""
    return next(a for a, t in zip(animals, targets) if t is sp)


def _scripted_expert(rules) -> Expert:
    """An expert whose answer function maps conclusion -> condition callable."""

    def answer(context, requests):
        return {"conditions": rules[context.target_conclusion](context.case_variable)}

    return Expert(interface=FunctionInterface(answer_fn=answer))


def _build_one_rule_rdr() -> EQLSingleClassRDR:
    """Smallest possible fitted RDR: one rule (milk -> mammal)."""
    rdr = EQLSingleClassRDR(Animal, "species")
    expert = _scripted_expert({Species.mammal: lambda v: v.milk == True})
    rdr.fit_case(first(Species.mammal), Species.mammal, expert)
    return rdr


def _build_two_rule_rdr() -> EQLSingleClassRDR:
    """Two-rule RDR: mammal + bird (alternative chain)."""
    rdr = EQLSingleClassRDR(Animal, "species")
    expert = _scripted_expert(
        {
            Species.mammal: lambda v: v.milk == True,
            Species.bird: lambda v: v.feathers == True,
        }
    )
    rdr.fit_case(first(Species.mammal), Species.mammal, expert)
    rdr.fit_case(first(Species.bird), Species.bird, expert)
    return rdr


# ---------------------------------------------------------------------------
# Minimal synthetic FunctionCase subclass (does NOT need @rdr, not yet implemented).
# ---------------------------------------------------------------------------


def _score_func(x: int, y: int) -> float:
    """Toy function whose parameters become FunctionCase fields."""
    return float(x + y)


@dataclass
class SyntheticFunctionCase(FunctionCase):
    """A hand-written FunctionCase subclass that mimics what @rdr would generate."""

    x: int
    y: int
    _output: float


SyntheticFunctionCase.function = _score_func


# ---------------------------------------------------------------------------
# Step A — ``save_path`` field and auto-save on ``fit_case``
# ---------------------------------------------------------------------------


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestSavePathFieldDefault(unittest.TestCase):
    """save_path=None (the default): fit_case must not write any file."""

    def test_save_path_default_is_none(self):
        """EQLSingleClassRDR must default save_path to None."""
        rdr = EQLSingleClassRDR(Animal, "species")
        self.assertIsNone(rdr.save_path)

    def test_fit_case_does_not_write_file_when_save_path_is_none(self):
        """No file is created when save_path is None, even after fit_case."""
        with tempfile.TemporaryDirectory() as d:
            # Spy on save_rdr_with_case: it must never be called.
            with patch(
                "krrood.entity_query_language.rdr.single_class.save_rdr_with_case"
            ) as mock_save:
                rdr = EQLSingleClassRDR(Animal, "species")
                expert = _scripted_expert(
                    {Species.mammal: lambda v: v.milk == True}
                )
                rdr.fit_case(first(Species.mammal), Species.mammal, expert)
                mock_save.assert_not_called()


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestSavePathFieldPresent(unittest.TestCase):
    """save_path=<path>: fit_case writes a file at that path."""

    def test_file_is_created_after_first_fit_case(self):
        """After fit_case, a non-empty file must exist at save_path."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            rdr = EQLSingleClassRDR(Animal, "species", save_path=path)
            expert = _scripted_expert(
                {Species.mammal: lambda v: v.milk == True}
            )
            rdr.fit_case(first(Species.mammal), Species.mammal, expert)
            self.assertTrue(os.path.exists(path))

    def test_file_content_is_non_empty_after_first_fit_case(self):
        """The file written by save_path must not be empty."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            rdr = EQLSingleClassRDR(Animal, "species", save_path=path)
            expert = _scripted_expert(
                {Species.mammal: lambda v: v.milk == True}
            )
            rdr.fit_case(first(Species.mammal), Species.mammal, expert)
            with open(path) as f:
                content = f.read()
            self.assertGreater(len(content.strip()), 0)

    def test_save_called_twice_for_two_fit_case_calls(self):
        """save_rdr_with_case must be invoked once per _insert_rule, i.e. twice for two new rules."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            with patch(
                "krrood.entity_query_language.rdr.single_class.save_rdr_with_case"
            ) as mock_save:
                rdr = EQLSingleClassRDR(Animal, "species", save_path=path)
                expert = _scripted_expert(
                    {
                        Species.mammal: lambda v: v.milk == True,
                        Species.bird: lambda v: v.feathers == True,
                    }
                )
                rdr.fit_case(first(Species.mammal), Species.mammal, expert)
                rdr.fit_case(first(Species.bird), Species.bird, expert)
                self.assertEqual(mock_save.call_count, 2)

    def test_save_not_called_when_case_already_correct(self):
        """fit_case on an already-correct case must not trigger a second write."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            with patch(
                "krrood.entity_query_language.rdr.single_class.save_rdr_with_case"
            ) as mock_save:
                rdr = EQLSingleClassRDR(Animal, "species", save_path=path)
                expert = _scripted_expert(
                    {Species.mammal: lambda v: v.milk == True}
                )
                rdr.fit_case(first(Species.mammal), Species.mammal, expert)
                count_after_first = mock_save.call_count
                # Fit the same case again — it is already correct, no new rule inserted.
                rdr.fit_case(first(Species.mammal), Species.mammal, expert)
                self.assertEqual(mock_save.call_count, count_after_first)


# ---------------------------------------------------------------------------
# Step B — ``save_rdr_with_case`` function
# ---------------------------------------------------------------------------


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestSaveRdrWithCaseRoundtrip(unittest.TestCase):
    """Files written by save_rdr_with_case must load correctly via load_rdr."""

    def test_loaded_rdr_classifies_trained_cases(self):
        """load_rdr of a save_rdr_with_case file must reproduce the original classifications."""
        rdr = _build_two_rule_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            loaded = load_rdr(path)
        self.assertEqual(loaded.classify(first(Species.mammal)), Species.mammal)
        self.assertEqual(loaded.classify(first(Species.bird)), Species.bird)

    def test_loaded_rdr_unclassified_case_returns_unset(self):
        """load_rdr of save_rdr_with_case returns UNSET for a case not covered by the rules."""
        rdr = _build_one_rule_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            loaded = load_rdr(path)
        self.assertIs(loaded.classify(first(Species.bird)), UNSET)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestSaveRdrWithCaseFunctionCaseHeader(unittest.TestCase):
    """FunctionCase-based RDR: the saved file must contain the dataclass class-header section."""

    def _build_function_case_rdr(self) -> EQLSingleClassRDR:
        """Build a minimal RDR whose case_type is a FunctionCase subclass."""
        rdr = EQLSingleClassRDR(SyntheticFunctionCase, "_output")
        case = SyntheticFunctionCase(x=1, y=2, _output=None)
        expert = _scripted_expert({3.0: lambda v: (v.x == 1)})
        rdr.fit_case(case, 3.0, expert)
        return rdr

    def test_saved_file_contains_dataclass_decorator(self):
        """The class-header section must include @dataclass."""
        rdr = self._build_function_case_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            with open(path) as f:
                content = f.read()
        self.assertIn("@dataclass", content)

    def test_saved_file_contains_class_definition_inheriting_function_case(self):
        """The class-header section must declare a class that inherits FunctionCase."""
        rdr = self._build_function_case_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            with open(path) as f:
                content = f.read()
        self.assertIn("FunctionCase", content)
        self.assertIn("class SyntheticFunctionCase", content)

    def test_saved_file_contains_function_classvar_assignment(self):
        """The class-header must assign the function ClassVar outside the class body."""
        rdr = self._build_function_case_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            with open(path) as f:
                content = f.read()
        # e.g. ``SyntheticFunctionCase.function = _score_func``
        self.assertIn("SyntheticFunctionCase.function", content)

    def test_saved_file_contains_all_dataclass_fields(self):
        """The class-header must list every non-self/cls annotated parameter as a field."""
        rdr = self._build_function_case_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            with open(path) as f:
                content = f.read()
        self.assertIn("x:", content)
        self.assertIn("y:", content)
        self.assertIn("_output:", content)

    def test_class_header_appears_before_rule_tree(self):
        """The dataclass header must precede the rule-tree section (variable + query lines)."""
        rdr = self._build_function_case_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            with open(path) as f:
                content = f.read()
        header_pos = content.find("@dataclass")
        query_pos = content.find("query = entity(")
        self.assertGreater(
            query_pos,
            header_pos,
            "Class header must appear before the rule-tree variable/query definitions.",
        )


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestSaveRdrWithCaseNonFunctionCaseFallback(unittest.TestCase):
    """Non-FunctionCase RDR: save_rdr_with_case falls back to plain save_rdr (no class header)."""

    def test_no_class_header_in_fallback_file(self):
        """Output for a plain (non-FunctionCase) RDR must not contain a @dataclass header."""
        rdr = _build_one_rule_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            with open(path) as f:
                content = f.read()
        self.assertNotIn("@dataclass\nclass", content)

    def test_fallback_file_matches_save_rdr_output(self):
        """Fallback output must be identical to what plain save_rdr would produce."""
        rdr = _build_one_rule_rdr()
        with tempfile.TemporaryDirectory() as d:
            path_with_case = os.path.join(d, "with_case.py")
            path_plain = os.path.join(d, "plain.py")
            save_rdr_with_case(rdr, path_with_case)
            save_rdr(rdr, path_plain)
            with open(path_with_case) as f:
                content_with_case = f.read()
            with open(path_plain) as f:
                content_plain = f.read()
        self.assertEqual(content_with_case, content_plain)

    def test_fallback_file_loads_correctly(self):
        """A fallback file must still be loadable via load_rdr and classify correctly."""
        rdr = _build_one_rule_rdr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            loaded = load_rdr(path)
        self.assertEqual(loaded.classify(first(Species.mammal)), Species.mammal)


# ---------------------------------------------------------------------------
# Step B — ``rdr_to_python`` with ``case_type_is_local=True``
# ---------------------------------------------------------------------------


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestRdrToPythonCaseTypeIsLocal(unittest.TestCase):
    """rdr_to_python(case_type_is_local=True): case type must not appear in an import line."""

    def test_case_type_not_imported_when_local(self):
        """When case_type_is_local=True, no import statement should import the case type."""
        rdr = _build_one_rule_rdr()
        src = rdr_to_python(rdr, case_type_is_local=True)
        import_lines = [
            line for line in src.splitlines() if line.strip().startswith(("import ", "from "))
        ]
        case_type_name = rdr.case_type.__name__  # "Animal"
        for line in import_lines:
            self.assertNotIn(
                case_type_name,
                line,
                f"Case type '{case_type_name}' must not appear in import line: {line!r}",
            )

    def test_conclusion_types_still_imported_when_local(self):
        """When case_type_is_local=True, conclusion types (enums) must still be imported."""
        rdr = _build_one_rule_rdr()
        src = rdr_to_python(rdr, case_type_is_local=True)
        # Species.mammal is referenced, so 'Species' must appear in an import line.
        import_lines = [
            line for line in src.splitlines() if line.strip().startswith(("import ", "from "))
        ]
        self.assertTrue(
            any("Species" in line for line in import_lines),
            "Conclusion enum 'Species' must still be imported even with case_type_is_local=True.",
        )

    def test_case_type_is_local_false_imports_case_type(self):
        """The default (case_type_is_local=False) must import the case type as usual."""
        rdr = _build_one_rule_rdr()
        src = rdr_to_python(rdr, case_type_is_local=False)
        import_lines = [
            line for line in src.splitlines() if line.strip().startswith(("import ", "from "))
        ]
        case_type_name = rdr.case_type.__name__
        self.assertTrue(
            any(case_type_name in line for line in import_lines),
            f"Case type '{case_type_name}' must appear in an import with case_type_is_local=False.",
        )

    def test_case_type_is_local_default_is_false(self):
        """rdr_to_python without the kwarg must behave the same as case_type_is_local=False."""
        rdr = _build_one_rule_rdr()
        src_default = rdr_to_python(rdr)
        src_explicit = rdr_to_python(rdr, case_type_is_local=False)
        self.assertEqual(src_default, src_explicit)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Phase 3 — save_rdr_with_case preserves the CornerCaseStore
# ---------------------------------------------------------------------------


import os as _os
import tempfile as _tempfile

from krrood.entity_query_language.rdr.serialization import walk_rules_in_emission_order


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestSaveRdrWithCasePreservesCornerCases(unittest.TestCase):
    """``save_rdr_with_case`` must persist and restore corner cases just like ``save_rdr``.

    The round-trip path goes through ``save_rdr_with_case`` (which prepends a
    ``FunctionCase`` class header for function-case RDRs and falls back to ``save_rdr``
    for plain RDRs).  In both cases the ``RDR_CORNER_CASES`` block must survive.
    """

    def test_save_rdr_with_case_preserves_corner_cases(self):
        """After ``save_rdr_with_case`` + ``load_rdr``, the loaded store has the
        same number of corner-case entries as the original.

        This tests both the plain-RDR fallback path (Animal case type is not a
        FunctionCase subclass) and validates that ``load_rdr`` restores the store
        from the ``RDR_CORNER_CASES`` block embedded by ``rdr_to_python``.
        """
        rdr = _build_two_rule_rdr()
        original_count = len(rdr.corner_cases.cases)
        # Sanity: two fit_case calls must have recorded two corner cases.
        self.assertEqual(original_count, 2)

        with _tempfile.TemporaryDirectory() as d:
            path = _os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            loaded = load_rdr(path)

        self.assertEqual(len(loaded.corner_cases.cases), original_count)

    def test_save_rdr_with_case_corner_case_values_equal_originals(self):
        """Each restored corner case must equal (``==``) the case used during fitting.

        Verifies that ``AsdictCaseSerializer`` faithfully reconstructs the ``Animal``
        dataclass through the constructor-source round-trip path.
        """
        rdr = _build_two_rule_rdr()
        ordered_before = walk_rules_in_emission_order(rdr.conditions_root)

        with _tempfile.TemporaryDirectory() as d:
            path = _os.path.join(d, "model.py")
            save_rdr_with_case(rdr, path)
            loaded = load_rdr(path)

        ordered_after = walk_rules_in_emission_order(loaded.conditions_root)
        self.assertEqual(len(ordered_before), len(ordered_after))
        for node_before, node_after in zip(ordered_before, ordered_after):
            original_case = rdr.corner_cases.get(node_before._id_)
            restored_case = loaded.corner_cases.get(node_after._id_)
            self.assertIsNotNone(
                original_case,
                "Each fitted rule must have a recorded corner case",
            )
            self.assertIsNotNone(
                restored_case,
                "Each loaded rule must have a restored corner case",
            )
            self.assertEqual(restored_case, original_case)
