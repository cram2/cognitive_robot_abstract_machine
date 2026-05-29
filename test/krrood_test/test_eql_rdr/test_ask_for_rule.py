"""
Tests for sequential conclusion-asking in EQL-based Ripple Down Rules.

:class:`Expert.ask_for_rule` is the path taken when ``fit_case`` receives no ground-truth
target (``target=UNSET``).  The expert is asked **two** sequential questions:

1. A focused *conclusion-only* interaction (``requests == [conclusion_request]``).
2. If the chosen conclusion differs from the current one, a *conditions-only* interaction
   via ``ask_for_conditions`` (``requests == [conditions_request]``).

Each test verifies exactly one behavioural guarantee of this sequential flow.
"""

from __future__ import annotations

import dataclasses
import unittest

from krrood.entity_query_language.factories import and_
from krrood.entity_query_language.rdr.aid import ConclusionAid
from krrood.entity_query_language.rdr.expert import (
    ANSWER_NAME,
    CONCLUSION_NAME,
    Expert,
    NoConclusionProvided,
    NoConditionsProvided,
    make_conclusion_validator,
)
from krrood.entity_query_language.rdr.interface import (
    AnswerRequest,
    CaseContext,
    ExpertAbort,
    FunctionInterface,
)
from krrood.entity_query_language.rdr.rule_tree_view import walk_rules
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.utils import UNSET

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()

# Feature fields: every trait column except the identifier and the predicted attribute.
FEATURE_FIELDS = [
    f.name for f in dataclasses.fields(Animal) if f.name not in ("name", "species")
]


def first(sp: Species) -> Animal:
    """Return the first animal in the dataset whose ground-truth label is *sp*."""
    return next(a for a, t in zip(animals, targets) if t is sp)


def _full_feature_conditions(case_variable, case):
    """EQL expression matching a case's complete feature vector."""
    return and_(
        *[getattr(case_variable, f) == getattr(case, f) for f in FEATURE_FIELDS]
    )


def labelling_expert(target_by_name: dict) -> Expert:
    """
    An expert for the ask-for-rule path: supplies *both* conclusion and conditions.

    The answer_fn is invoked twice per ``ask_for_rule``:
    - First with ``requests`` containing only the conclusion request.
    - Second (via ``ask_for_conditions``) with ``requests`` containing only the conditions request.
    """

    def answer(context, requests):
        result = {
            ANSWER_NAME: _full_feature_conditions(
                context.case_variable, context.case_instance
            )
        }
        if any(r.name == CONCLUSION_NAME for r in requests):
            result[CONCLUSION_NAME] = target_by_name[context.case_instance.name]
        return result

    return Expert(interface=FunctionInterface(answer_fn=answer))


def _rule_count(rdr: EQLSingleClassRDR) -> int:
    """Count the number of rules currently in *rdr*'s rule tree."""
    if rdr.conditions_root is None:
        return 0
    return len(walk_rules(rdr.conditions_root))


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestNoTargetSingleFitLabelsAndClassifies(unittest.TestCase):
    def test_no_target_single_fit_classifies_correctly(self):
        """fit_case with no target accepts the expert's conclusion and classifies the case."""
        target_by_name = {a.name: t for a, t in zip(animals, targets)}
        rdr = EQLSingleClassRDR(Animal, "species")
        animal = first(Species.mammal)
        rdr.fit_case(animal, expert=labelling_expert(target_by_name))
        self.assertEqual(rdr.classify(animal), Species.mammal)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestAskForRuleCallCount(unittest.TestCase):
    def test_answer_fn_invoked_twice_per_no_target_fit(self):
        """ask_for_rule calls answer_fn exactly twice: once for conclusion, once for conditions."""
        target_by_name = {a.name: t for a, t in zip(animals, targets)}
        call_log = []

        def tracking_answer(context, requests):
            call_log.append([r.name for r in requests])
            result = {
                ANSWER_NAME: _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            }
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = target_by_name[context.case_instance.name]
            return result

        expert = Expert(interface=FunctionInterface(answer_fn=tracking_answer))
        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit_case(first(Species.mammal), expert=expert)

        self.assertEqual(len(call_log), 2)

    def test_first_call_has_only_conclusion_request(self):
        """The first answer_fn invocation carries only the conclusion request."""
        target_by_name = {a.name: t for a, t in zip(animals, targets)}
        call_log = []

        def tracking_answer(context, requests):
            call_log.append([r.name for r in requests])
            result = {
                ANSWER_NAME: _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            }
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = target_by_name[context.case_instance.name]
            return result

        expert = Expert(interface=FunctionInterface(answer_fn=tracking_answer))
        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit_case(first(Species.mammal), expert=expert)

        self.assertIn(CONCLUSION_NAME, call_log[0])
        self.assertNotIn(ANSWER_NAME, call_log[0])

    def test_second_call_has_only_conditions_request(self):
        """The second answer_fn invocation carries only the conditions request."""
        target_by_name = {a.name: t for a, t in zip(animals, targets)}
        call_log = []

        def tracking_answer(context, requests):
            call_log.append([r.name for r in requests])
            result = {
                ANSWER_NAME: _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            }
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = target_by_name[context.case_instance.name]
            return result

        expert = Expert(interface=FunctionInterface(answer_fn=tracking_answer))
        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit_case(first(Species.mammal), expert=expert)

        self.assertIn(ANSWER_NAME, call_log[1])
        self.assertNotIn(CONCLUSION_NAME, call_log[1])


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestNoTargetBulkFit(unittest.TestCase):
    def test_bulk_no_target_fit_reproduces_ground_truth(self):
        """Bulk ask-for-rule fitting over ~12 animals reproduces each ground-truth label."""
        target_by_name = {a.name: t for a, t in zip(animals, targets)}
        subset_animals = animals[:12]
        subset_targets = targets[:12]

        rdr = EQLSingleClassRDR(Animal, "species")
        expert = labelling_expert(target_by_name)

        # Pass UNSET per case so each fit_case takes the ask_for_rule path.
        rdr.fit(subset_animals, targets=[UNSET] * len(subset_animals), expert=expert)

        for animal, expected in zip(subset_animals, subset_targets):
            with self.subTest(animal=animal.name):
                self.assertEqual(rdr.classify(animal), expected)

    def test_fit_without_targets_uses_ask_for_rule(self):
        """fit(cases) with no targets labels each case via ask_for_rule (UNSET, not None)."""
        target_by_name = {a.name: t for a, t in zip(animals, targets)}
        subset_animals = animals[:12]
        subset_targets = targets[:12]

        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(subset_animals, expert=labelling_expert(target_by_name))

        for animal, expected in zip(subset_animals, subset_targets):
            with self.subTest(animal=animal.name):
                self.assertEqual(rdr.classify(animal), expected)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestRefinementUnderNoTarget(unittest.TestCase):
    def test_no_target_refines_wrong_rule_for_mammal(self):
        """A no-target fit of a mammal corrects an over-general fish rule via refinement."""
        rdr = EQLSingleClassRDR(Animal, "species")

        # Seed an over-general rule: backbone → fish.
        def fish_conditions_answer(context, requests):
            return {ANSWER_NAME: context.case_variable.backbone == True}

        fish_expert = Expert(
            interface=FunctionInterface(answer_fn=fish_conditions_answer)
        )
        fish = first(Species.fish)
        rdr.fit_case(fish, Species.fish, fish_expert)

        # A mammal (also has backbone) mis-classifies as fish before the refinement.
        mammal = first(Species.mammal)
        self.assertEqual(rdr.classify(mammal), Species.fish)

        # Now label the mammal with no ground-truth target.
        def mammal_labelling_answer(context, requests):
            result = {
                ANSWER_NAME: _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            }
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = Species.mammal
            return result

        mammal_expert = Expert(
            interface=FunctionInterface(answer_fn=mammal_labelling_answer)
        )
        rdr.fit_case(mammal, expert=mammal_expert)

        # Refinement: the mammal now classifies as mammal; the fish still classifies as fish.
        self.assertEqual(rdr.classify(mammal), Species.mammal)
        self.assertEqual(rdr.classify(fish), Species.fish)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestKeepCurrentConclusion(unittest.TestCase):
    def test_fit_returns_current_when_expert_reaffirms_conclusion(self):
        """fit_case returns the current conclusion without inserting a rule when expert re-affirms it."""
        rdr = EQLSingleClassRDR(Animal, "species")
        mammal = first(Species.mammal)

        # Seed a rule so the mammal already classifies correctly.
        def seed_answer(context, requests):
            return {ANSWER_NAME: context.case_variable.milk == True}

        rdr.fit_case(
            mammal,
            Species.mammal,
            Expert(interface=FunctionInterface(answer_fn=seed_answer)),
        )
        self.assertEqual(rdr.classify(mammal), Species.mammal)

        rule_count_before = _rule_count(rdr)

        # Now no-target-fit the same case; expert returns conclusion == current.
        call_log = []

        def reaffirming_answer(context, requests):
            call_log.append([r.name for r in requests])
            result = {}
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = Species.mammal  # same as current
            return result

        reaffirm_expert = Expert(
            interface=FunctionInterface(answer_fn=reaffirming_answer)
        )
        returned = rdr.fit_case(mammal, expert=reaffirm_expert)

        self.assertEqual(returned, Species.mammal)
        # Rule count must be unchanged (no new rule was inserted).
        self.assertEqual(_rule_count(rdr), rule_count_before)

    def test_expert_not_asked_for_conditions_when_reaffirming(self):
        """When the expert re-affirms the current conclusion, answer_fn is called only once (conclusion only)."""
        rdr = EQLSingleClassRDR(Animal, "species")
        mammal = first(Species.mammal)

        def seed_answer(context, requests):
            return {ANSWER_NAME: context.case_variable.milk == True}

        rdr.fit_case(
            mammal,
            Species.mammal,
            Expert(interface=FunctionInterface(answer_fn=seed_answer)),
        )

        call_log = []

        def reaffirming_answer(context, requests):
            call_log.append([r.name for r in requests])
            result = {}
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = Species.mammal
            return result

        rdr.fit_case(
            mammal,
            expert=Expert(interface=FunctionInterface(answer_fn=reaffirming_answer)),
        )

        # Only the conclusion request was made — the conditions step was skipped.
        self.assertEqual(len(call_log), 1)
        self.assertIn(CONCLUSION_NAME, call_log[0])


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestAbortOnConclusion(unittest.TestCase):
    def test_abort_during_conclusion_raises_no_conclusion_provided(self):
        """ExpertAbort during the conclusion step raises NoConclusionProvided."""
        rdr = EQLSingleClassRDR(Animal, "species")

        def aborting_answer(context, requests):
            raise ExpertAbort([CONCLUSION_NAME])

        expert = Expert(interface=FunctionInterface(answer_fn=aborting_answer))
        with self.assertRaises(NoConclusionProvided):
            rdr.fit_case(first(Species.mammal), expert=expert)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestAbortOnConditions(unittest.TestCase):
    def test_abort_during_conditions_raises_no_conditions_provided(self):
        """ExpertAbort during the conditions step raises NoConditionsProvided."""
        rdr = EQLSingleClassRDR(Animal, "species")
        call_count = {"n": 0}

        def answer_fn(context, requests):
            call_count["n"] += 1
            if any(r.name == CONCLUSION_NAME for r in requests):
                # First call: supply a valid conclusion to proceed to the conditions step.
                return {CONCLUSION_NAME: Species.mammal}
            # Second call (conditions): abort.
            raise ExpertAbort([ANSWER_NAME])

        expert = Expert(interface=FunctionInterface(answer_fn=answer_fn))
        with self.assertRaises(NoConditionsProvided):
            rdr.fit_case(first(Species.mammal), expert=expert)

        # Confirm the conclusion step DID run (call_count at least 1) before the abort.
        self.assertGreaterEqual(call_count["n"], 1)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestAidSuggestionAccepted(unittest.TestCase):
    def test_aid_suggestion_pre_seeds_conclusion_and_is_used(self):
        """A valid ConclusionAid suggestion pre-seeds the conclusion default and is accepted."""
        mammal = first(Species.mammal)
        rdr = EQLSingleClassRDR(Animal, "species")

        class SpeciesAid(ConclusionAid):
            """Always suggests Species.mammal regardless of the case."""

            def suggest(self, context):
                return Species.mammal

        # The answer_fn returns ONLY conditions (no conclusion key).
        # This simulates the expert accepting the pre-seeded default without typing it.
        def conditions_only_answer(context, requests):
            result = {}
            if any(r.name == ANSWER_NAME for r in requests):
                result[ANSWER_NAME] = _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            # Deliberately omit CONCLUSION_NAME so the default (aid suggestion) stands.
            return result

        expert = Expert(
            interface=FunctionInterface(answer_fn=conditions_only_answer),
            aids=[SpeciesAid()],
        )
        rdr.fit_case(mammal, expert=expert)
        self.assertEqual(rdr.classify(mammal), Species.mammal)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestAidSuggestionOverridden(unittest.TestCase):
    def test_expert_can_override_aid_suggestion(self):
        """The expert can override the aid suggestion by supplying a different conclusion."""
        mammal = first(Species.mammal)
        bird = first(Species.bird)
        rdr = EQLSingleClassRDR(Animal, "species")

        class WrongAid(ConclusionAid):
            """Suggests the wrong species (bird) for every case."""

            def suggest(self, context):
                return Species.bird

        def overriding_answer(context, requests):
            result = {}
            if any(r.name == CONCLUSION_NAME for r in requests):
                # Expert consciously picks mammal even though the aid suggested bird.
                result[CONCLUSION_NAME] = Species.mammal
            if any(r.name == ANSWER_NAME for r in requests):
                result[ANSWER_NAME] = _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            return result

        expert = Expert(
            interface=FunctionInterface(answer_fn=overriding_answer),
            aids=[WrongAid()],
        )
        rdr.fit_case(mammal, expert=expert)
        self.assertEqual(rdr.classify(mammal), Species.mammal)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestInvalidAidSuggestionIgnored(unittest.TestCase):
    def test_invalid_aid_suggestion_does_not_pre_seed_conclusion(self):
        """An aid suggestion that fails domain validation is silently ignored (default stays UNSET)."""
        rdr = EQLSingleClassRDR(Animal, "species")

        class BadAid(ConclusionAid):
            """Returns a non-domain value (a plain string) as suggestion."""

            def suggest(self, context):
                return "nonsense"

        defaults_seen = []

        def tracking_answer(context, requests):
            for r in requests:
                if r.name == CONCLUSION_NAME:
                    defaults_seen.append(r.default)
            result = {}
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = Species.mammal
            if any(r.name == ANSWER_NAME for r in requests):
                result[ANSWER_NAME] = _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            return result

        expert = Expert(
            interface=FunctionInterface(answer_fn=tracking_answer),
            aids=[BadAid()],
        )
        rdr.fit_case(first(Species.mammal), expert=expert)

        # The bad suggestion must not have pre-seeded the answer; default stays UNSET.
        self.assertTrue(defaults_seen, "Conclusion request was never seen by answer_fn")
        self.assertIs(defaults_seen[0], UNSET)

    def test_invalid_aid_suggestion_does_not_prevent_correct_classification(self):
        """Classification still works correctly even when an aid suggestion is invalid."""
        rdr = EQLSingleClassRDR(Animal, "species")

        class BadAid(ConclusionAid):
            def suggest(self, context):
                return "nonsense"

        def answer(context, requests):
            result = {}
            if any(r.name == CONCLUSION_NAME for r in requests):
                result[CONCLUSION_NAME] = Species.mammal
            if any(r.name == ANSWER_NAME for r in requests):
                result[ANSWER_NAME] = _full_feature_conditions(
                    context.case_variable, context.case_instance
                )
            return result

        expert = Expert(
            interface=FunctionInterface(answer_fn=answer),
            aids=[BadAid()],
        )
        mammal = first(Species.mammal)
        rdr.fit_case(mammal, expert=expert)
        self.assertEqual(rdr.classify(mammal), Species.mammal)


class TestSuggestedConclusionHelper(unittest.TestCase):
    """Unit-tests for Expert._suggested_conclusion (independent of the zoo dataset)."""

    def _make_context_and_domain(self, animal, rdr):
        """Build a minimal CaseContext and conclusion domain for *animal* using *rdr*."""
        from krrood.entity_query_language.rdr.interface import CaseContext

        return CaseContext(
            case_instance=animal,
            case_variable=rdr.case_variable,
            current_conclusion=UNSET,
            conclusion_domain=rdr.conclusion_domain,
            aids=[],
        )

    @unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
    def test_valid_aid_suggestion_is_returned(self):
        """_suggested_conclusion returns the suggestion when it passes domain validation."""
        rdr = EQLSingleClassRDR(Animal, "species")
        animal = first(Species.bird)
        context = self._make_context_and_domain(animal, rdr)

        class GoodAid(ConclusionAid):
            def suggest(self, ctx):
                return Species.bird

        validator = make_conclusion_validator(rdr.conclusion_domain, allow_unset=False)
        expert = Expert(
            interface=FunctionInterface(answer_fn=lambda c, r: {}), aids=[GoodAid()]
        )
        result = expert._suggested_conclusion(context, validator)
        self.assertIs(result, Species.bird)

    @unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
    def test_invalid_aid_suggestion_returns_unset(self):
        """_suggested_conclusion returns UNSET when the suggestion fails domain validation."""
        rdr = EQLSingleClassRDR(Animal, "species")
        animal = first(Species.bird)
        context = self._make_context_and_domain(animal, rdr)

        class BadAid(ConclusionAid):
            def suggest(self, ctx):
                return "not_a_species"

        validator = make_conclusion_validator(rdr.conclusion_domain, allow_unset=False)
        expert = Expert(
            interface=FunctionInterface(answer_fn=lambda c, r: {}), aids=[BadAid()]
        )
        result = expert._suggested_conclusion(context, validator)
        self.assertIs(result, UNSET)

    @unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
    def test_none_suggestion_is_skipped(self):
        """_suggested_conclusion treats an aid returning None as "no suggestion" (skipped)."""
        rdr = EQLSingleClassRDR(Animal, "species")
        animal = first(Species.bird)
        context = self._make_context_and_domain(animal, rdr)

        validator = make_conclusion_validator(rdr.conclusion_domain, allow_unset=False)
        # Base ConclusionAid.suggest always returns None.
        expert = Expert(
            interface=FunctionInterface(answer_fn=lambda c, r: {}),
            aids=[ConclusionAid()],
        )
        result = expert._suggested_conclusion(context, validator)
        self.assertIs(result, UNSET)


if __name__ == "__main__":
    unittest.main()
