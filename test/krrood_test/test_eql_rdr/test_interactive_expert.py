"""
Tests for the interactive expert: ``Expert`` (policy) over ``IPythonInterface`` (mechanism).

The real interface opens an embedded IPython shell; here we inject a stub ``shell_runner``
that plays the expert's part — building a live EQL condition expression from the namespace
the expert is given. This exercises namespace construction, scope capture, the live-object
answer contract, the validate/re-prompt loop, abort handling, and integration with fit_case.
"""

import contextlib
import dataclasses
import io
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
    CASE_VARIABLE_NAME,
)
from krrood.entity_query_language.rdr.magics import (
    _KNOWLEDGE_KEY,
    _make_knowledge_magic,
)
from krrood.entity_query_language.rdr.utils import UNSET
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.interface import CaseContext, FunctionInterface

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
        expert.ask_for_conditions(
            first(Species.bird), rdr.case_variable, Species.bird, Species.mammal
        )
        header = captured["header"]
        self.assertIn("bird", header.lower())

    def test_returns_live_eql_expression(self):
        expert = expert_with(maximally_specific_runner())
        rdr = EQLSingleClassRDR(Animal, "species")
        cond = expert.ask_for_conditions(
            first(Species.mammal), rdr.case_variable, Species.mammal
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
                first(Species.mammal), rdr.case_variable, Species.mammal
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
            first(Species.mammal), rdr.case_variable, Species.mammal
        )
        self.assertEqual(calls["n"], 2)
        self.assertIsInstance(cond, SymbolicExpression)

    def test_captures_user_definition_scope(self):
        interactive_sentinel = USER_SCOPE_SENTINEL  # noqa: F841
        rdr = EQLSingleClassRDR(Animal, "species")
        captured = {}
        expert = expert_with(maximally_specific_runner(captured))
        expert.ask_for_conditions(
            first(Species.mammal), rdr.case_variable, Species.mammal
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
        expert.ask_for_conditions(
            first(Species.mammal), rdr.case_variable, Species.mammal
        )
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
        expert.ask_for_conditions(
            first(Species.mammal), rdr.case_variable, Species.mammal
        )
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

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            magic("")

        output = f.getvalue()
        self.assertIn("usage", output.lower())


def _make_animal(
    name: str,
    *,
    milk: bool = False,
    feathers: bool = False,
    fins: bool = False,
    backbone: bool = True,
    venomous: bool = False,
) -> Animal:
    """Return a minimal animal with one discriminating feature set."""
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


def _scripted_function_expert(rules: dict) -> Expert:
    """A ``FunctionInterface``-backed expert that records every ``CaseContext`` it sees."""
    recorded: list = []

    def answer(context, requests):
        recorded.append(context)
        return {"conditions": rules[context.target_conclusion](context.case_variable)}

    expert = Expert(interface=FunctionInterface(answer_fn=answer))
    expert.recorded_contexts = recorded  # type: ignore[attr-defined]
    return expert


class TestCaseContextCornerCase(unittest.TestCase):
    """Tests for ``CaseContext.corner_case`` field and ``fit_case`` provenance wiring."""

    def test_case_context_has_corner_case_field(self):
        """``CaseContext`` exposes a ``corner_case`` attribute that defaults to ``None``."""
        rdr = EQLSingleClassRDR(Animal, "species")
        case = _make_animal("mammal", milk=True)
        ctx = CaseContext(case_instance=case, case_variable=rdr.case_variable)
        self.assertIsNone(ctx.corner_case)

    def test_fit_case_first_rule_corner_case_is_none(self):
        """When the very first rule is fitted (empty RDR, no prior firing) the
        ``CaseContext`` passed to the expert has ``corner_case == None``.

        No firing anchor exists for the first case, so there is no corner case to show.
        """
        rdr = EQLSingleClassRDR(Animal, "species")
        mammal = _make_animal("mammal", milk=True)
        expert = _scripted_function_expert({Species.mammal: lambda v: v.milk == True})

        rdr.fit_case(mammal, Species.mammal, expert)

        self.assertEqual(len(expert.recorded_contexts), 1)
        ctx = expert.recorded_contexts[0]
        self.assertIsNone(ctx.corner_case)

    def test_fit_case_refinement_populates_corner_case_in_context(self):
        """When a second case triggers a refinement (wrong rule fired) the ``CaseContext``
        passed to the expert for that second case has ``corner_case`` equal to the first
        case's Animal instance — i.e., the corner case of the firing rule.

        This is the core Phase 4 contract: the expert can see *why the original rule
        exists* by inspecting the corner case shown alongside the new case.
        """
        rdr = EQLSingleClassRDR(Animal, "species")
        mammal = _make_animal("mammal", milk=True)
        # fish has backbone=False so the mammal rule (milk==True) does NOT fire for it;
        # use an over-general first rule that WILL misfire for the second case.
        # Strategy: first rule is "backbone == True" -> mammal; second case is also
        # backbone==True but should be classified as bird (feathers==True).
        backbone_animal = _make_animal("backbone_mammal", milk=True, backbone=True)
        feathered_backbone = Animal(
            name="owl",
            hair=False,
            feathers=True,
            eggs=True,
            milk=False,
            airborne=True,
            aquatic=False,
            predator=True,
            toothed=False,
            backbone=True,
            breathes=True,
            venomous=False,
            fins=False,
            legs=2,
            tail=False,
            domestic=False,
            catsize=False,
        )

        expert = _scripted_function_expert(
            {
                Species.mammal: lambda v: v.backbone == True,
                Species.bird: lambda v: v.feathers == True,
            }
        )

        # First fit: backbone rule -> mammal (no prior firing; corner_case must be None).
        rdr.fit_case(backbone_animal, Species.mammal, expert)
        self.assertIsNone(expert.recorded_contexts[0].corner_case)

        # Second fit: feathered bird with backbone fires the mammal rule (wrong).
        # The refinement expert call must see corner_case == backbone_animal.
        rdr.fit_case(feathered_backbone, Species.bird, expert)

        self.assertEqual(len(expert.recorded_contexts), 2)
        ctx_refinement = expert.recorded_contexts[1]
        self.assertIs(ctx_refinement.corner_case, backbone_animal)

    def test_fit_case_alternative_corner_case_is_none(self):
        """When a second case does NOT fire any existing rule (alternative path) the
        ``CaseContext`` for that second case has ``corner_case == None``.

        No rule fired means no firing anchor, so no corner case to display.
        """
        rdr = EQLSingleClassRDR(Animal, "species")
        mammal = _make_animal("mammal", milk=True)
        # fish: no backbone, no milk, no feathers — will not fire the mammal rule.
        fish = _make_animal("fish", fins=True, backbone=False)

        expert = _scripted_function_expert(
            {
                Species.mammal: lambda v: v.milk == True,
                Species.fish: lambda v: v.fins == True,
            }
        )

        rdr.fit_case(mammal, Species.mammal, expert)
        rdr.fit_case(fish, Species.fish, expert)

        self.assertEqual(len(expert.recorded_contexts), 2)
        ctx_alternative = expert.recorded_contexts[1]
        self.assertIsNone(ctx_alternative.corner_case)


if __name__ == "__main__":
    unittest.main()
