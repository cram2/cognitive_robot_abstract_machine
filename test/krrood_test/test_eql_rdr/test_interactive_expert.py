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
from krrood.entity_query_language.rdr.backward_inference import ConclusionKnowledge
from krrood.entity_query_language.rdr.expert import (
    ANSWER_NAME,
    Expert,
    NoConditionsProvided,
)
from krrood.entity_query_language.rdr.interactive import IPythonInterface
from krrood.entity_query_language.rdr.interface import (
    CASE_INSTANCE_NAME,
    CASE_VARIABLE_NAME, )
from krrood.entity_query_language.rdr.magics import (
    _KNOWLEDGE_KEY,
    _make_knowledge_magic,
)
from krrood.entity_query_language.rdr.utils import UNSET
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
        expert.ask_for_conditions(case, rdr.case_variable, Species.mammal)

        ns = captured["namespace"]
        for verb in ("entity", "variable", "and_", "refinement", "alternative", "add"):
            self.assertIn(verb, ns)
        self.assertIn(CASE_VARIABLE_NAME, ns)
        self.assertIs(ns[CASE_VARIABLE_NAME], rdr.case_variable)
        self.assertIs(ns[CASE_INSTANCE_NAME], case)

    def test_header_mentions_case_target(self):
        captured = {}
        expert = expert_with(maximally_specific_runner(captured))
        rdr = EQLSingleClassRDR(Animal, "species")
        expert.ask_for_conditions(first(Species.bird), rdr.case_variable, Species.bird, Species.mammal)
        header = captured["header"]
        self.assertIn("bird", header.lower())

    def test_returns_live_eql_expression(self):
        expert = expert_with(maximally_specific_runner())
        rdr = EQLSingleClassRDR(Animal, "species")
        cond = expert.ask_for_conditions(first(Species.mammal), rdr.case_variable, Species.mammal)
        self.assertIsInstance(cond, SymbolicExpression)
        self.assertNotIsInstance(cond, str)

    def test_abort_raises_no_conditions(self):
        def run_and_abort(namespace, header):
            namespace["exit"]()  # expert gives up without answering

        expert = expert_with(run_and_abort)
        rdr = EQLSingleClassRDR(Animal, "species")
        with self.assertRaises(NoConditionsProvided):
            expert.ask_for_conditions(first(Species.mammal), rdr.case_variable, Species.mammal)

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
        cond = expert.ask_for_conditions(first(Species.mammal), rdr.case_variable, Species.mammal)
        self.assertEqual(calls["n"], 2)
        self.assertIsInstance(cond, SymbolicExpression)

    def test_captures_user_definition_scope(self):
        interactive_sentinel = USER_SCOPE_SENTINEL  # noqa: F841
        rdr = EQLSingleClassRDR(Animal, "species")
        captured = {}
        expert = expert_with(maximally_specific_runner(captured))
        expert.ask_for_conditions(first(Species.mammal), rdr.case_variable, Species.mammal)
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


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestKnowsMagic(unittest.TestCase):
    """Tests for the ``%knows`` backward-inference magic."""

    def test_knowledge_key_in_namespace_when_rdr_set(self):
        """The RDR reference appears in the namespace under _KNOWLEDGE_KEY."""
        captured = {}

        def runner(ns, header):
            captured["key"] = ns.get(_KNOWLEDGE_KEY)
            cv = ns[CASE_VARIABLE_NAME]
            case = ns[CASE_INSTANCE_NAME]
            ns[ANSWER_NAME] = cv.milk == True

        rdr = EQLSingleClassRDR(Animal, "species")
        interface = IPythonInterface(shell_runner=runner, rdr=rdr)
        expert = Expert(interface=interface)
        expert.ask_for_conditions(first(Species.mammal), rdr.case_variable, Species.mammal)
        self.assertIs(captured["key"], rdr)

    def test_knowledge_key_absent_when_no_rdr(self):
        """Without ``rdr`` set, _KNOWLEDGE_KEY is not in the namespace."""
        captured = {}

        def runner(ns, header):
            captured["key"] = ns.get(_KNOWLEDGE_KEY)
            cv = ns[CASE_VARIABLE_NAME]
            case = ns[CASE_INSTANCE_NAME]
            ns[ANSWER_NAME] = cv.milk == True

        rdr = EQLSingleClassRDR(Animal, "species")
        interface = IPythonInterface(shell_runner=runner)
        expert = Expert(interface=interface)
        expert.ask_for_conditions(first(Species.mammal), rdr.case_variable, Species.mammal)
        self.assertIsNone(captured["key"])

    def test_knows_queries_rdr_directly(self):
        """what_do_we_know_about returns correct results after fitting through interactive."""
        rdr = EQLSingleClassRDR(Animal, "species")

        def runner(ns, header):
            cv = ns[CASE_VARIABLE_NAME]
            case = ns[CASE_INSTANCE_NAME]
            ns[ANSWER_NAME] = cv.milk == True

        interface = IPythonInterface(shell_runner=runner, rdr=rdr)
        expert = Expert(interface=interface)
        rdr.fit_case(first(Species.mammal), Species.mammal, expert)

        knowledge = rdr.what_do_we_know_about(Species.mammal)
        self.assertIsInstance(knowledge, ConclusionKnowledge)
        self.assertTrue(knowledge.is_satisfiable())
        self.assertEqual(len(knowledge.sufficient_condition_sets), 1)

    def test_knows_empty_rdr(self):
        """Empty RDR returns no paths for any value."""
        rdr = EQLSingleClassRDR(Animal, "species")
        knowledge = rdr.what_do_we_know_about(Species.molusc)
        self.assertIsInstance(knowledge, ConclusionKnowledge)
        self.assertFalse(knowledge.is_satisfiable())

    def test_knows_magic_function_evals_in_namespace(self):
        """The magic closure evals its argument and queries the RDR."""
        rdr = EQLSingleClassRDR(Animal, "species")

        def runner(ns, header):
            cv = ns[CASE_VARIABLE_NAME]
            case = ns[CASE_INSTANCE_NAME]
            ns[ANSWER_NAME] = cv.milk == True

        interface = IPythonInterface(shell_runner=runner, rdr=rdr)
        expert = Expert(interface=interface)
        rdr.fit_case(first(Species.mammal), Species.mammal, expert)

        # Build a namespace as the shell would see it
        ns = {_KNOWLEDGE_KEY: rdr, "Species": Species, "True": True, "False": False}
        magic = _make_knowledge_magic(ns, IPythonInterface().palette)

        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            magic("Species.mammal")

        output = f.getvalue()
        self.assertIn("mammal", output.lower())
        self.assertIn("milk", output.lower())

    def test_knows_magic_bad_argument(self):
        """Invalid magic argument prints an error."""
        rdr = EQLSingleClassRDR(Animal, "species")
        ns = {_KNOWLEDGE_KEY: rdr}
        magic = _make_knowledge_magic(ns, IPythonInterface().palette)

        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            magic("Species.NonExistentValue")

        output = f.getvalue()
        self.assertIn("error", output.lower())

    def test_knows_magic_empty_line(self):
        """Empty magic line prints usage hint."""
        rdr = EQLSingleClassRDR(Animal, "species")
        ns = {_KNOWLEDGE_KEY: rdr}
        magic = _make_knowledge_magic(ns, IPythonInterface().palette)

        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            magic("")

        output = f.getvalue()
        self.assertIn("usage", output.lower())


if __name__ == "__main__":
    unittest.main()
