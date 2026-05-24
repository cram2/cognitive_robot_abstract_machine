"""
Tests for the interactive expert: ``Expert`` (policy) over ``IPythonInterface`` (mechanism).

The real interface opens an embedded IPython shell; here we inject a stub ``shell_runner``
that plays the expert's part — building a live EQL condition expression from the namespace
the expert is given. This exercises namespace construction, scope capture, the live-object
answer contract, the validate/re-prompt loop, abort handling, and integration with fit_case.
"""

import dataclasses
import unittest

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.rdr.expert import (
    ANSWER_NAME,
    Expert,
    NoConditionsProvided,
)
from krrood.entity_query_language.rdr.interactive import IPythonInterface
from krrood.entity_query_language.rdr.interface import (
    CASE_INSTANCE_NAME,
    CASE_VARIABLE_NAME,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()

FEATURE_FIELDS = [
    f.name for f in dataclasses.fields(Animal) if f.name not in ("name", "species")
]

USER_SCOPE_SENTINEL = "interactive_sentinel"


def first(sp: Species) -> Animal:
    return next(a for a, t in zip(animals, targets) if t is sp)


def maximally_specific_runner(captured=None):
    """A stub shell runner that assigns a full-feature-vector condition.

    Builds the condition with the EQL ``and_`` taken *from the namespace* (proving the
    factories were injected) over the case variable, matching the case's features.
    """

    def run(namespace, header):
        if captured is not None:
            captured["namespace"] = namespace
            captured["header"] = header
        case = namespace[CASE_INSTANCE_NAME]
        case_variable = namespace[CASE_VARIABLE_NAME]
        and_ = namespace["and_"]
        namespace[ANSWER_NAME] = and_(
            *[getattr(case_variable, f) == getattr(case, f) for f in FEATURE_FIELDS]
        )

    return run


def expert_with(runner) -> Expert:
    return Expert(interface=IPythonInterface(shell_runner=runner))


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestInteractiveExpert(unittest.TestCase):
    def test_namespace_has_factories_case_instance_and_variable(self):
        captured = {}
        expert = expert_with(maximally_specific_runner(captured))
        rdr = EQLSingleClassRDR(Animal, "species")
        case = first(Species.mammal)
        expert.ask_for_conditions(case, None, Species.mammal, rdr.case_variable)

        ns = captured["namespace"]
        for verb in ("entity", "variable", "and_", "refinement", "alternative", "add"):
            self.assertIn(verb, ns)
        self.assertIn(CASE_VARIABLE_NAME, ns)
        self.assertIs(ns[CASE_VARIABLE_NAME], rdr.case_variable)
        self.assertIs(ns[CASE_INSTANCE_NAME], case)

    def test_header_mentions_case_target_and_answer_name(self):
        captured = {}
        expert = expert_with(maximally_specific_runner(captured))
        rdr = EQLSingleClassRDR(Animal, "species")
        expert.ask_for_conditions(
            first(Species.bird), Species.mammal, Species.bird, rdr.case_variable
        )
        header = captured["header"]
        self.assertIn(ANSWER_NAME, header)
        self.assertIn("bird", header.lower())
        self.assertIn(CASE_VARIABLE_NAME, header)

    def test_returns_live_eql_expression(self):
        expert = expert_with(maximally_specific_runner())
        rdr = EQLSingleClassRDR(Animal, "species")
        cond = expert.ask_for_conditions(
            first(Species.mammal), None, Species.mammal, rdr.case_variable
        )
        self.assertIsInstance(cond, SymbolicExpression)
        self.assertNotIsInstance(cond, str)

    def test_abort_raises_no_conditions(self):
        def run_and_abort(namespace, header):
            namespace["exit"]()  # expert gives up without answering

        expert = expert_with(run_and_abort)
        rdr = EQLSingleClassRDR(Animal, "species")
        with self.assertRaises(NoConditionsProvided):
            expert.ask_for_conditions(
                first(Species.mammal), None, Species.mammal, rdr.case_variable
            )

    def test_invalid_answer_is_reprompted_then_accepted(self):
        # First attempt builds the condition over the *concrete* case (a bool — invalid);
        # the loop must re-open with an error and accept the second, valid attempt.
        calls = {"n": 0}

        def run(namespace, header):
            calls["n"] += 1
            case = namespace[CASE_INSTANCE_NAME]
            case_variable = namespace[CASE_VARIABLE_NAME]
            if calls["n"] == 1:
                namespace[ANSWER_NAME] = case.milk == True  # plain bool — rejected
            else:
                self.assertIn("[error]", header)  # error surfaced on re-prompt
                namespace[ANSWER_NAME] = case_variable.milk == True

        expert = expert_with(run)
        rdr = EQLSingleClassRDR(Animal, "species")
        cond = expert.ask_for_conditions(
            first(Species.mammal), None, Species.mammal, rdr.case_variable
        )
        self.assertEqual(calls["n"], 2)
        self.assertIsInstance(cond, SymbolicExpression)

    def test_captures_user_definition_scope(self):
        interactive_sentinel = USER_SCOPE_SENTINEL  # noqa: F841
        rdr = EQLSingleClassRDR(Animal, "species")
        captured = {}
        expert = expert_with(maximally_specific_runner(captured))
        expert.ask_for_conditions(
            first(Species.mammal), None, Species.mammal, rdr.case_variable
        )
        self.assertEqual(
            captured["namespace"].get("interactive_sentinel"), USER_SCOPE_SENTINEL
        )

    def test_fit_through_interactive_expert(self):
        expert = expert_with(maximally_specific_runner())
        rdr = EQLSingleClassRDR(Animal, "species")
        subset = list(zip(animals, targets))[:15]
        for case, target in subset:
            rdr.fit_case(case, target, expert)
        for case, target in subset:
            self.assertEqual(rdr.classify(case), target, case.name)


if __name__ == "__main__":
    unittest.main()
