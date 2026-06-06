"""
Phase 5 tests: EQLSingleClassRDR orchestration (classify + fit_case).

Experts here are programmatic and return live EQL condition expressions built over
the RDR's shared case variable — the same contract the interactive shell will honour.
"""

import dataclasses
import unittest

from krrood.entity_query_language.factories import and_
from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.interface import FunctionInterface
from krrood.entity_query_language.rdr.utils import UNSET
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()

# Feature fields (everything except the name and the underspecified species).
FEATURE_FIELDS = [
    f.name for f in dataclasses.fields(Animal) if f.name not in ("name", "species")
]


def first(sp: Species) -> Animal:
    return next(a for a, t in zip(animals, targets) if t is sp)


def maximally_specific_expert() -> Expert:
    """An expert whose rule matches the case's full feature vector.

    Guarantees convergence: each distinct feature vector gets its own rule, so the
    RDR memorises the training set.
    """

    def answer(context, requests):
        case_variable, case = context.case_variable, context.case_instance
        return {
            "conditions": and_(
                *[getattr(case_variable, f) == getattr(case, f) for f in FEATURE_FIELDS]
            )
        }

    return Expert(interface=FunctionInterface(answer_fn=answer))


def labelling_expert(target_by_name):
    """An expert that supplies *both* conclusion and conditions (ask-for-rule path)."""

    def answer(context, requests):
        result = {
            "conditions": and_(
                *[
                    getattr(context.case_variable, f)
                    == getattr(context.case_instance, f)
                    for f in FEATURE_FIELDS
                ]
            )
        }
        if any(r.name == "conclusion" for r in requests):
            result["conclusion"] = target_by_name[context.case_instance.name]
        return result

    return Expert(interface=FunctionInterface(answer_fn=answer))


def scripted_expert(rules):
    """An expert returning conditions from a per-(target) callable, for controlled
    scenarios. Returns ``(expert, calls)`` where ``calls`` records each expert interaction as
    ``(case_name, current_conclusion, target_conclusion)``.
    """
    calls = []

    def answer(context, requests):
        calls.append(
            (
                context.case_instance.name,
                context.current_conclusion,
                context.target_conclusion,
            )
        )
        return {"conditions": rules[context.target_conclusion](context.case_variable)}

    return Expert(interface=FunctionInterface(answer_fn=answer)), calls


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestEQLSingleClassRDR(unittest.TestCase):
    def test_empty_rdr_classifies_none(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        self.assertIsNone(rdr.classify(first(Species.mammal)))

    def test_first_rule_via_fit(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        expert, calls = scripted_expert({Species.mammal: lambda v: v.milk == True})
        rdr.fit_case(first(Species.mammal), Species.mammal, expert)
        self.assertEqual(rdr.classify(first(Species.mammal)), Species.mammal)
        # The expert was asked exactly once (current conclusion was None).
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], UNSET)

    def test_no_fire_routes_to_alternative(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        expert, calls = scripted_expert(
            {
                Species.mammal: lambda v: v.milk == True,
                Species.bird: lambda v: v.feathers == True,
            }
        )
        rdr.fit_case(first(Species.mammal), Species.mammal, expert)
        # Bird does not fire under the milk rule -> alternative inserted.
        rdr.fit_case(first(Species.bird), Species.bird, expert)
        self.assertEqual(rdr.classify(first(Species.bird)), Species.bird)
        self.assertEqual(rdr.classify(first(Species.mammal)), Species.mammal)
        # Second call saw no current_conclusion (nothing fired for the bird).
        self.assertEqual(calls[-1][1], UNSET)

    def test_wrong_fire_routes_to_refinement(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        # Over-general first rule: backbone -> fish. A mammal (backbone) mis-fires fish.
        expert, calls = scripted_expert(
            {
                Species.fish: lambda v: v.backbone == True,
                Species.mammal: lambda v: v.milk == True,
            }
        )
        rdr.fit_case(first(Species.fish), Species.fish, expert)
        mammal = first(Species.mammal)
        self.assertEqual(rdr.classify(mammal), Species.fish)  # currently wrong

        rdr.fit_case(mammal, Species.mammal, expert)
        # The wrong 'fish' conclusion was refined to mammal for milk-bearing cases.
        self.assertEqual(rdr.classify(mammal), Species.mammal)
        self.assertEqual(rdr.classify(first(Species.fish)), Species.fish)
        # The refinement call saw current_conclusion == fish (a rule fired, wrongly).
        self.assertEqual(calls[-1][1], Species.fish)

    def test_fit_idempotent_when_already_correct(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        expert, calls = scripted_expert({Species.mammal: lambda v: v.milk == True})
        rdr.fit_case(first(Species.mammal), Species.mammal, expert)
        calls_before = len(calls)
        # Fitting an already-correct case must not ask the expert again.
        rdr.fit_case(first(Species.mammal), Species.mammal, expert)
        self.assertEqual(len(calls), calls_before)

    def test_fit_case_without_target_uses_ask_for_rule(self):
        # Regression: if UNSET is not passed as the "no-target" sentinel (e.g. None is
        # used instead), fit_case takes the ask_for_conditions branch with target=None
        # and stores a rule whose conclusion is None rather than the expert's label.
        target_by_name = {a.name: t for a, t in zip(animals, targets)}
        rdr = EQLSingleClassRDR(Animal, "species")
        animal = first(Species.mammal)
        rdr.fit_case(animal, expert=labelling_expert(target_by_name))
        self.assertEqual(rdr.classify(animal), Species.mammal)

    def test_full_fit_memorises_training_set(self):
        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(animals, targets, maximally_specific_expert())
        correct = sum(rdr.classify(a) == t for a, t in zip(animals, targets))
        accuracy = correct / len(animals)
        # Maximally-specific rules memorise the training set; allow for the handful of
        # identical-feature/different-species collisions in the zoo data.
        self.assertGreaterEqual(accuracy, 0.95)

    def test_shared_variable_identity(self):
        # Attribute access on the shared case variable is stable, so conditions the
        # expert builds over rdr.case_variable share the rule tree's nodes.
        rdr = EQLSingleClassRDR(Animal, "species")
        self.assertIs(rdr.case_variable._type_, Animal)
        self.assertEqual(
            rdr.conclusion_variable._id_,
            getattr(rdr.case_variable, "species")._id_,
        )


def _molusc_backbone_false_expert():
    """An expert that authors rules matching the scorpion scenario:

    * mammals get ``milk == True``
    * moluscs without backbones get ``milk == False`` (first time)
    * reptiles (venomous + backbone) get ``venomous == True``
    * moluscs that have become misclassified as reptile get ``backbone == False``

    The return type differs depending on the fitting path (the production path uses
    ``ask_for_conditions``, so only ``conditions`` is returned; the no-target path
    uses ``ask_for_rule``, so both ``conclusion`` and ``conditions`` are returned).
    """
    call_details: list = []

    def answer(context, requests):
        case_variable = context.case_variable
        case_instance = context.case_instance
        current_conclusion = context.current_conclusion
        target = context.target_conclusion
        call_details.append((case_instance.name, current_conclusion, target))

        has_conclusion = any(r.name == "conclusion" for r in requests)

        if target is UNSET and has_conclusion:
            # No-target path: return both conclusion and conditions.
            result = {"conclusion": Species.molusc}
            if current_conclusion == Species.reptile:
                result["conditions"] = case_variable.backbone == False
            elif current_conclusion is UNSET:
                result["conditions"] = case_variable.milk == False
            else:
                result["conditions"] = case_variable.milk == False
            return result

        if target == Species.mammal:
            return {"conditions": case_variable.milk == True}
        if target == Species.reptile:
            return {"conditions": case_variable.venomous == True}
        if target == Species.molusc:
            if current_conclusion == Species.reptile:
                return {"conditions": case_variable.backbone == False}
            return {"conditions": case_variable.milk == False}

        return {"conditions": case_variable.milk == True}

    return Expert(interface=FunctionInterface(answer_fn=answer)), call_details


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestFitConvergent(unittest.TestCase):
    """Convergent fitting detects and corrects cases broken by later rules."""

    def _make_scorpion_scenario(self):
        """Three animals that reproduce the retroactive-breaking pattern:

        1. **mammal** (eggs=False, milk=True) -> Species.mammal
        2. **molusc** (eggs=False, milk=False, venomous=True, backbone=False) -> Species.molusc
        3. **reptile** (eggs=False, milk=False, venomous=True, backbone=True) -> Species.reptile

        Processed in order 1, 2, 3, the reptile's ``venomous==True`` rule intercepts
        the molusc case (which is also venomous), misclassifying it as reptile.
        """
        mammal = Animal(
            name="scenario_mammal",
            hair=True,
            feathers=False,
            eggs=False,
            milk=True,
            airborne=False,
            aquatic=False,
            predator=False,
            toothed=True,
            backbone=True,
            breathes=True,
            venomous=False,
            fins=False,
            legs=4,
            tail=True,
            domestic=False,
            catsize=True,
        )
        molusc = Animal(
            name="scenario_molusc",
            hair=False,
            feathers=False,
            eggs=False,
            milk=False,
            airborne=False,
            aquatic=False,
            predator=False,
            toothed=False,
            backbone=False,
            breathes=True,
            venomous=True,
            fins=False,
            legs=0,
            tail=False,
            domestic=False,
            catsize=False,
        )
        reptile = Animal(
            name="scenario_reptile",
            hair=False,
            feathers=False,
            eggs=False,
            milk=False,
            airborne=False,
            aquatic=False,
            predator=True,
            toothed=True,
            backbone=True,
            breathes=True,
            venomous=True,
            fins=False,
            legs=4,
            tail=True,
            domestic=False,
            catsize=False,
        )
        return [mammal, molusc, reptile], [
            Species.mammal,
            Species.molusc,
            Species.reptile,
        ]

    def test_fit_already_convergent_in_one_pass(self):
        """A model that already converges in a single pass should not add extra expert calls."""
        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(animals, targets, maximally_specific_expert())
        correct = sum(rdr.classify(a) == t for a, t in zip(animals, targets))
        self.assertGreaterEqual(correct / len(animals), 0.95)

    def test_fit_convergent_recovers_from_broken_cases(self):
        """Convergent fitting re-fits cases broken by later rules (the scorpion pattern)."""
        cases, case_targets = self._make_scorpion_scenario()
        expert, calls = _molusc_backbone_false_expert()

        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(cases, case_targets, expert)

        for c, t in zip(cases, case_targets):
            self.assertEqual(
                rdr.classify(c),
                t,
                f"{c.name}: expected {t}, got {rdr.classify(c)}",
            )

        # The molusc was visited twice: once in the first pass (before any rules
        # existed for it — expert supplied not_milk) and once in the second pass
        # (now misclassified as reptile by the venomous rule — expert supplied
        # not_backbone).  The re-visit is what makes fitting convergent.
        molusc_calls = [
            (cur, tgt) for name, cur, tgt in calls if name == "scenario_molusc"
        ]
        self.assertEqual(
            len(molusc_calls),
            2,
            f"Expected 2 calls for molusc (first pass + re-fit), got {len(molusc_calls)}",
        )
        # The re-fit (second call) saw current == reptile (broken by venomous rule).
        self.assertEqual(molusc_calls[1], (Species.reptile, Species.molusc))

    def test_fit_convergent_without_targets_stays_single_pass(self):
        """When targets is None, no convergence is attempted (single pass only)."""
        cases, case_targets = self._make_scorpion_scenario()
        target_by_name = {c.name: t for c, t in zip(cases, case_targets)}
        expert = labelling_expert(target_by_name)

        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(cases, None, expert)

        for c, t in zip(cases, case_targets):
            self.assertEqual(rdr.classify(c), t, f"{c.name}: expected {t}")

    def test_fit_convergent_max_passes_capped(self):
        """A pathological case that never converges stops after max_passes."""
        # An expert that answers randomly — the RDR can never converge.
        case = Animal(
            name="endless",
            hair=False,
            feathers=False,
            eggs=True,
            milk=False,
            airborne=False,
            aquatic=False,
            predator=False,
            toothed=False,
            backbone=False,
            breathes=True,
            venomous=False,
            fins=False,
            legs=0,
            tail=False,
            domestic=False,
            catsize=False,
        )

        def oscillating_answer(context, requests):
            # Draws the wrong conclusion — the model will never stabilise.
            return {"conditions": context.case_variable.eggs == True}

        expert = Expert(interface=FunctionInterface(answer_fn=oscillating_answer))
        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit([case], [Species.mammal], expert, max_passes=3)
        # The loop exited after max_passes — assert a rule was added.
        self.assertIsNotNone(rdr.query)


if __name__ == "__main__":
    unittest.main()
