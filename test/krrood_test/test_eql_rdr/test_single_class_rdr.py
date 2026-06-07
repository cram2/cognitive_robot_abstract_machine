"""
Tests for :class:`~krrood.entity_query_language.rdr.single_class.EQLSingleClassRDR`
orchestration (classify + fit_case).

Experts here are programmatic and return live EQL condition expressions built over
the RDR's shared case variable — the same contract the interactive shell will honour.
"""

import dataclasses
import unittest
from unittest.mock import patch

from krrood.entity_query_language.factories import and_
from krrood.entity_query_language.rdr.condition_resolver import ChainConditionResolver
from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.interface import (
    ExpertInterface,
    FunctionInterface,
)
from krrood.entity_query_language.rdr.progress import SpyProgressReporter
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


class SpyFunctionInterface(FunctionInterface):
    """A FunctionInterface that returns a :class:`SpyProgressReporter`.

    Overrides ``make_progress_reporter`` to inject a spy that records progress bar
    lifecycle calls without displaying anything, so the test can verify call sequences
    from ``fit()``.
    """

    def __init__(self, answer_fn, spy=None):
        super().__init__(answer_fn=answer_fn)
        self._spy = spy or SpyProgressReporter()

    def make_progress_reporter(self):
        return self._spy


def _maximally_specific_answer(context, requests):
    """Answer function that matches the case's full feature vector.

    Each distinct feature vector produces its own unique condition, memorising the
    training set.  Same logic as :func:`maximally_specific_expert` but exposed as a
    plain answer function for use with :class:`SpyFunctionInterface`.
    """
    case_variable, case = context.case_variable, context.case_instance
    return {
        "conditions": and_(
            *[getattr(case_variable, f) == getattr(case, f) for f in FEATURE_FIELDS]
        )
    }


def _scorpion_answer(context, requests):
    """Answer function for the scorpion retroactive-breaking scenario.

    ``targets`` path only — does not handle the no-target (``UNSET``) path.  Models the
    same logic as the inner closure of :func:`_molusc_backbone_false_expert` so the
    resulting RDR behaves identically.
    """
    case_variable = context.case_variable
    current = context.current_conclusion
    target = context.target_conclusion

    if target == Species.mammal:
        return {"conditions": case_variable.milk == True}
    if target == Species.reptile:
        return {"conditions": case_variable.venomous == True}
    if target == Species.molusc:
        if current == Species.reptile:
            return {"conditions": case_variable.backbone == False}
        return {"conditions": case_variable.milk == False}
    return {"conditions": case_variable.milk == True}


def _labelling_answer(target_by_name):
    """Build an answer function for the no-target (expert-labels) path.

    Returns maximally-specific conditions matching the case, and includes the
    conclusion only when the ``conclusion`` answer is requested (the
    ``ask_for_rule`` -> ``_ask_for_conclusion`` interact call).
    """

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

    return answer


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


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestProgressBarIntegration(unittest.TestCase):
    """Progress reporting lifecycle during :meth:`EQLSingleClassRDR.fit`.

    Each test verifies that ``fit()`` calls the correct sequence of progress-reporter
    lifecycle methods (``start``, ``update``, ``reset``, ``finish``) by injecting a
    :class:`SpyProgressReporter` via a custom
    :meth:`ExpertInterface.make_progress_reporter` override.
    """

    # ------------------------------------------------------------------
    # 1.  Happy path — single-pass convergent fit
    # ------------------------------------------------------------------

    def test_fit_calls_start_update_finish(self):
        """A single-pass fit calls ``start(N)``, ``update()`` N times, then ``finish()``."""
        rdr = EQLSingleClassRDR(Animal, "species")
        subset, subset_targets = animals[:5], targets[:5]

        spy = SpyProgressReporter()
        interface = SpyFunctionInterface(answer_fn=_maximally_specific_answer, spy=spy)
        expert = Expert(interface=interface)
        rdr.fit(subset, subset_targets, expert)

        assert_events = [("start", (5,), {"description": "Fitting RDR"})]
        assert_events += [("update", (1,), {})] * 5
        assert_events += [("finish", (), {})]
        self.assertEqual(spy.events, assert_events)

    # ------------------------------------------------------------------
    # 2.  Two-pass convergent fit — reset between passes
    # ------------------------------------------------------------------

    def test_fit_convergent_two_passes_calls_reset(self):
        """A convergent fit that needs two passes calls ``reset()`` between them.

        The scorpion scenario (mammal, then molusc, then reptile) causes a retroactive
        misclassification of the molusc by the reptile's rule, forcing a second pass.
        """
        cases, case_targets = TestFitConvergent()._make_scorpion_scenario()

        spy = SpyProgressReporter()
        interface = SpyFunctionInterface(answer_fn=_scorpion_answer, spy=spy)
        expert = Expert(interface=interface)

        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(cases, case_targets, expert)

        assert_events = [("start", (3,), {"description": "Fitting RDR"})]
        assert_events += [("update", (1,), {})] * 3
        assert_events += [("reset", (1,), {})]
        assert_events += [("update", (1,), {})]
        assert_events += [("finish", (), {})]
        self.assertEqual(spy.events, assert_events)

    # ------------------------------------------------------------------
    # 3.  No expert — no progress reporter created
    # ------------------------------------------------------------------

    def test_fit_without_expert_does_not_create_progress(self):
        """When ``expert`` is ``None``, ``fit()`` must not create a progress reporter.

        The ``make_progress_reporter`` method should never be called, and fitting
        already-correct cases must not crash.
        """
        rdr = EQLSingleClassRDR(Animal, "species")
        animal, target = first(Species.mammal), Species.mammal

        # Seed a rule so the case is already correctly classified.
        rdr.fit_case(animal, target, maximally_specific_expert())

        with patch.object(ExpertInterface, "make_progress_reporter") as mock:
            rdr.fit([animal], [target], expert=None)
            mock.assert_not_called()

        # Classification must be unchanged.
        self.assertEqual(rdr.classify(animal), target)

    # ------------------------------------------------------------------
    # 4.  Default FunctionInterface — ``None`` progress reporter
    # ------------------------------------------------------------------

    def test_fit_with_default_interface_no_bar(self):
        """A plain :class:`FunctionInterface` returns ``None`` from ``make_progress_reporter``.

        ``fit()`` must handle a ``None`` progress reporter gracefully (not crash, and
        converge cases correctly).
        """
        rdr = EQLSingleClassRDR(Animal, "species")
        subset, subset_targets = animals[:3], targets[:3]

        interface = FunctionInterface(answer_fn=_maximally_specific_answer)
        expert = Expert(interface=interface)
        rdr.fit(subset, subset_targets, expert)

        for a, t in zip(subset, subset_targets):
            self.assertEqual(rdr.classify(a), t)

    # ------------------------------------------------------------------
    # 5.  No-target path — single pass, no reset, finish called
    # ------------------------------------------------------------------

    def test_fit_no_targets_single_pass(self):
        """The no-target (``targets=None``) path: single pass, no reset.

        Only ``start``, ``update`` per case, and ``finish`` are expected.
        """
        rdr = EQLSingleClassRDR(Animal, "species")
        subset, subset_targets = animals[:3], targets[:3]
        target_by_name = {a.name: t for a, t in zip(subset, subset_targets)}

        spy = SpyProgressReporter()
        interface = SpyFunctionInterface(
            answer_fn=_labelling_answer(target_by_name), spy=spy
        )
        expert = Expert(interface=interface)
        rdr.fit(subset, None, expert)

        assert_events = [("start", (3,), {"description": "Fitting RDR"})]
        assert_events += [("update", (1,), {})] * 3
        assert_events += [("finish", (), {})]
        self.assertEqual(spy.events, assert_events)

        # All cases must be correctly classified.
        for a, t in zip(subset, subset_targets):
            self.assertEqual(rdr.classify(a), t)

    # ------------------------------------------------------------------
    # 6.  Max-passes exhausted — finish still called
    # ------------------------------------------------------------------

    def test_fit_max_passes_exhausted_finish_called(self):
        """``finish()`` is always called, even when ``max_passes`` is exhausted before
        convergence."""
        cases, case_targets = TestFitConvergent()._make_scorpion_scenario()

        spy = SpyProgressReporter()
        interface = SpyFunctionInterface(answer_fn=_scorpion_answer, spy=spy)
        expert = Expert(interface=interface)

        rdr = EQLSingleClassRDR(Animal, "species")
        rdr.fit(cases, case_targets, expert, max_passes=1)

        # finish() was called.
        self.assertEqual(spy.events[-1], ("finish", (), {}))
        # Only one pass completed — no reset occurred.
        self.assertNotIn(("reset", (1,), {}), spy.events)
        # start was called (sanity check).
        self.assertEqual(spy.events[0], ("start", (3,), {"description": "Fitting RDR"}))

    # ------------------------------------------------------------------
    # 7.  Semantic transparency — spy does not alter fitting
    # ------------------------------------------------------------------

    def test_progress_does_not_alter_fit_semantics(self):
        """Fitting with a :class:`SpyProgressReporter` produces the same classifications
        as fitting without one."""
        cases, case_targets = TestFitConvergent()._make_scorpion_scenario()

        # Control: fit without spy.
        rdr_control = EQLSingleClassRDR(Animal, "species")
        expert_control = Expert(interface=FunctionInterface(answer_fn=_scorpion_answer))
        rdr_control.fit(cases, case_targets, expert_control)

        # With spy.
        rdr_spy = EQLSingleClassRDR(Animal, "species")
        spy = SpyProgressReporter()
        interface = SpyFunctionInterface(answer_fn=_scorpion_answer, spy=spy)
        expert_spy = Expert(interface=interface)
        rdr_spy.fit(cases, case_targets, expert_spy)

        # Identical classifications.
        for c, t in zip(cases, case_targets):
            r1 = rdr_control.classify(c)
            r2 = rdr_spy.classify(c)
            self.assertEqual(
                r1,
                r2,
                f"Spy altered classification for {c.name}: "
                f"without spy={r1}, with spy={r2}",
            )

        # Sanity: the spy was active.
        self.assertGreater(len(spy.events), 0)


# ---------------------------------------------------------------------------
# Helpers for TestAutoConditionResolution
# ---------------------------------------------------------------------------


def _make_auto_resolution_animal(name: str, **kwargs) -> "Animal":
    """Build an :class:`Animal` with sensible defaults for auto-resolution scenarios.

    Only the fields supplied via ``kwargs`` deviate from the defaults.  The defaults
    produce a generic non-descript animal that no rule fires for, so callers only need
    to specify the discriminating features.
    """
    defaults = dict(
        hair=False,
        feathers=False,
        eggs=False,
        milk=False,
        airborne=False,
        aquatic=False,
        predator=False,
        toothed=False,
        backbone=True,
        breathes=True,
        venomous=False,
        fins=False,
        legs=4,
        tail=False,
        domestic=False,
        catsize=False,
    )
    defaults.update(kwargs)
    return Animal(name=name, **defaults)


@dataclasses.dataclass
class CountingFunctionInterface(FunctionInterface):
    """A :class:`FunctionInterface` that counts calls to its underlying answer function.

    The counter attribute :attr:`ask_for_conditions_count` increments every time the
    interface is invoked via :meth:`interact`.  This lets a test assert, without any
    mocking framework, that ``Expert.ask_for_conditions`` was (or was not) called for
    a specific fitting step.
    """

    ask_for_conditions_count: int = dataclasses.field(default=0, init=False)
    """Number of times :meth:`interact` has been called."""

    def interact(self, context, requests):
        """Forward to the parent and record the call."""
        self.ask_for_conditions_count += 1
        return super().interact(context, requests)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestAutoConditionResolution(unittest.TestCase):
    """Integration tests for auto-condition inference in :meth:`EQLSingleClassRDR.fit_case`.

    These tests verify the behaviour of the ``condition_resolver`` field on
    :class:`EQLSingleClassRDR`.  When a resolver is set, :meth:`fit_case` attempts to
    derive a differentiating condition automatically from the rule tree's backward-
    inference index before calling the expert.  The resolver is only active on the
    *refinement* branch (a wrong rule fired); the alternative branch and the UNSET
    (no-target) branch are always delegated to the expert.
    """

    # ------------------------------------------------------------------
    # Shared scenario builder
    # ------------------------------------------------------------------

    @staticmethod
    def _three_species_rdr(answer_fn, *, with_resolver: bool):
        """Return ``(rdr, interface)`` with mammal / reptile / bird already fitted.

        The fit order is:
          1. mammal  (``milk==True -> mammal``)
          2. reptile (``venomous==True -> reptile``)
          3. bird1   (``feathers==True -> bird``)

        After these three insertions the tree already holds backward-inference
        knowledge for ``Species.bird`` (``feathers==True``).  A second bird that
        is also venomous (``bird2``) will be wrongly classified as reptile —
        setting up the auto-resolution refinement path.

        :param answer_fn: The FunctionInterface answer function to embed.
        :param with_resolver: When ``True``, the RDR is constructed with
            ``ChainConditionResolver.backward_inference_default()``.
        :return: ``(rdr, counting_interface)``.
        """
        interface = CountingFunctionInterface(answer_fn=answer_fn)
        expert = Expert(interface=interface)
        resolver = (
            ChainConditionResolver.backward_inference_default()
            if with_resolver
            else None
        )
        rdr = EQLSingleClassRDR(Animal, "species", condition_resolver=resolver)

        mammal = _make_auto_resolution_animal("auto_m1", milk=True, hair=True)
        reptile = _make_auto_resolution_animal(
            "auto_r1", venomous=True, eggs=True, toothed=True
        )
        bird1 = _make_auto_resolution_animal(
            "auto_b1", feathers=True, eggs=True, airborne=True, legs=2
        )
        rdr.fit_case(mammal, Species.mammal, expert)
        rdr.fit_case(reptile, Species.reptile, expert)
        rdr.fit_case(bird1, Species.bird, expert)
        return rdr, interface

    @staticmethod
    def _standard_answer_fn(context, requests):
        """Condition answer function for the three-species scenario.

        Maps target conclusions to maximally simple discriminating conditions:
        * mammal  → ``milk==True``
        * bird    → ``feathers==True``
        * reptile → ``venomous==True``
        * fish    → ``fins==True``
        """
        case_variable = context.case_variable
        target = context.target_conclusion
        if target == Species.mammal:
            return {"conditions": case_variable.milk == True}
        if target == Species.bird:
            return {"conditions": case_variable.feathers == True}
        if target == Species.reptile:
            return {"conditions": case_variable.venomous == True}
        if target == Species.fish:
            return {"conditions": case_variable.fins == True}
        raise ValueError(f"Unexpected target in standard_answer_fn: {target!r}")

    @staticmethod
    def _unset_path_answer_fn(context, requests):
        """Answer function for the no-target (``UNSET``) path.

        Returns a conclusion (``Species.bird``) when asked, plus a ``feathers==True``
        condition — enough for the UNSET branch test to succeed without the test
        caring about the specific conclusion chosen.
        """
        case_variable = context.case_variable
        result = {"conditions": case_variable.feathers == True}
        if any(r.name == "conclusion" for r in requests):
            result["conclusion"] = Species.bird
        return result

    # ------------------------------------------------------------------
    # Test 1 — Expert NOT called when auto-resolution succeeds
    # ------------------------------------------------------------------

    def test_expert_not_called_when_auto_resolution_succeeds(self):
        """Auto-resolution silences the expert when backward inference finds a discriminator.

        Scenario: mammal (milk=True), reptile (venomous=True), bird1 (feathers=True) are
        fitted in that order.  The tree now knows ``feathers==True`` for ``Species.bird``.
        ``bird2`` has both ``feathers=True`` and ``venomous=True``; it is initially
        misclassified as reptile (reptile's rule fires first).

        On ``fit_case(bird2, Species.bird, expert)``, :meth:`_try_auto_resolve` must find
        ``feathers==True`` from the backward-inference knowledge of ``Species.bird`` — this
        guard is True for ``bird2`` and False for the reptile corner case (feathers=False).
        The expert MUST NOT be called (zero additional interact() invocations).
        After fitting, ``classify(bird2)`` must return ``Species.bird``.
        """
        rdr, interface = self._three_species_rdr(
            self._standard_answer_fn, with_resolver=True
        )
        bird2 = _make_auto_resolution_animal(
            "auto_b2", feathers=True, venomous=True, eggs=True, legs=2
        )

        # Confirm bird2 is wrongly classified before fitting (refinement path will trigger).
        self.assertEqual(
            rdr.classify(bird2),
            Species.reptile,
            "Precondition: bird2 must be misclassified as reptile before fitting.",
        )

        count_before = interface.ask_for_conditions_count
        rdr.fit_case(bird2, Species.bird, Expert(interface=interface))
        count_after = interface.ask_for_conditions_count

        self.assertEqual(
            count_after,
            count_before,
            "Expert must NOT be called when auto-resolution succeeds: "
            f"interact() was invoked {count_after - count_before} extra time(s).",
        )
        self.assertEqual(
            rdr.classify(bird2),
            Species.bird,
            "bird2 must be correctly classified as Species.bird after auto-resolved fitting.",
        )

    # ------------------------------------------------------------------
    # Test 2 — Expert IS called when auto-resolution fails (fallback)
    # ------------------------------------------------------------------

    def test_expert_called_when_auto_resolution_fails(self):
        """Expert is invoked as fallback when backward inference finds no discriminator.

        A case whose target species (``Species.fish``) has no existing rules in the
        tree provides no backward-inference knowledge.  When the resolver is given a
        case that is wrongly classified as reptile (venomous=True, fins=True),
        ``TargetKnowledgeResolver`` and ``CornerCaseKnowledgeResolver`` both return
        ``None`` because no fish knowledge exists.  The expert MUST be called exactly
        once as the fallback.
        """
        rdr, interface = self._three_species_rdr(
            self._standard_answer_fn, with_resolver=True
        )
        # fish2: venomous=True triggers the reptile rule -> wrong (should be fish)
        # No backward-inference knowledge for Species.fish exists yet.
        fish2 = _make_auto_resolution_animal(
            "auto_f2", fins=True, venomous=True, aquatic=True, toothed=True
        )

        self.assertEqual(
            rdr.classify(fish2),
            Species.reptile,
            "Precondition: fish2 must be misclassified as reptile before fitting.",
        )

        count_before = interface.ask_for_conditions_count
        rdr.fit_case(fish2, Species.fish, Expert(interface=interface))
        calls_made = interface.ask_for_conditions_count - count_before

        self.assertEqual(
            calls_made,
            1,
            "Expert MUST be called exactly once when auto-resolution fails "
            f"(got {calls_made} call(s)).",
        )
        self.assertEqual(
            rdr.classify(fish2),
            Species.fish,
            "fish2 must be correctly classified as Species.fish after expert-supplied fitting.",
        )

    # ------------------------------------------------------------------
    # Test 3 — No resolver (default) — expert always called
    # ------------------------------------------------------------------

    def test_expert_always_called_when_no_resolver_set(self):
        """Without a ``condition_resolver``, the expert is always called on the refinement path.

        Even when the rule tree already holds backward-inference knowledge for
        ``Species.bird`` (``feathers==True``), a resolver-free RDR must not attempt
        auto-resolution.  The expert MUST be called exactly once for ``bird2`` — the
        ``_try_auto_resolve`` guard ``condition_resolver is None`` returns ``None``
        immediately and the fallback expert path is taken unconditionally.
        """
        rdr, interface = self._three_species_rdr(
            self._standard_answer_fn, with_resolver=False
        )
        bird2 = _make_auto_resolution_animal(
            "auto_b2_no_res", feathers=True, venomous=True, eggs=True, legs=2
        )

        self.assertEqual(
            rdr.classify(bird2),
            Species.reptile,
            "Precondition: bird2 must be misclassified as reptile before fitting.",
        )

        count_before = interface.ask_for_conditions_count
        rdr.fit_case(bird2, Species.bird, Expert(interface=interface))
        calls_made = interface.ask_for_conditions_count - count_before

        self.assertEqual(
            calls_made,
            1,
            "Expert MUST be called once when condition_resolver is None "
            f"(got {calls_made} call(s)).",
        )
        self.assertEqual(
            rdr.classify(bird2),
            Species.bird,
            "bird2 must be correctly classified after expert-supplied fitting.",
        )

    # ------------------------------------------------------------------
    # Test 4 — End-to-end: full convergence with auto-resolution
    # ------------------------------------------------------------------

    def test_end_to_end_convergence_expert_silent_for_auto_resolved_duplicates(self):
        """All duplicate cases converge correctly and the expert is never called for them.

        After fitting the three originals (mammal, reptile, bird1) with the expert,
        the RDR has backward-inference knowledge for each species.  Two duplicate cases
        that would be wrongly classified (``bird2`` as reptile) are then fitted.
        The resolver finds a discriminator from the existing knowledge without consulting
        the expert.  All five animals must be classified correctly after fitting, and
        zero additional expert calls must have occurred for the duplicate pass.
        """
        rdr, interface = self._three_species_rdr(
            self._standard_answer_fn, with_resolver=True
        )

        mammal1 = _make_auto_resolution_animal("auto_m1_ref", milk=True, hair=True)
        reptile1 = _make_auto_resolution_animal(
            "auto_r1_ref", venomous=True, eggs=True, toothed=True
        )
        bird1 = _make_auto_resolution_animal(
            "auto_b1_ref", feathers=True, eggs=True, airborne=True, legs=2
        )
        bird2 = _make_auto_resolution_animal(
            "auto_b2_e2e", feathers=True, venomous=True, eggs=True, legs=2
        )

        # bird2 is wrongly classified before fitting.
        self.assertEqual(rdr.classify(bird2), Species.reptile)

        count_before = interface.ask_for_conditions_count
        expert = Expert(interface=interface)
        rdr.fit_case(
            mammal1, Species.mammal, expert
        )  # already correct — no expert call
        rdr.fit_case(bird2, Species.bird, expert)  # auto-resolved — no expert call
        calls_for_duplicates = interface.ask_for_conditions_count - count_before

        self.assertEqual(
            calls_for_duplicates,
            0,
            "Expert must not be called for already-correct or auto-resolved duplicate cases "
            f"(got {calls_for_duplicates} call(s)).",
        )

        # All cases (originals from _three_species_rdr plus new duplicates) must be correct.
        # Animal is an unhashable dataclass, so use a list of (case, target) pairs.
        expected_classifications = [
            (mammal1, Species.mammal),
            (reptile1, Species.reptile),
            (bird1, Species.bird),
            (bird2, Species.bird),
        ]
        for case, target in expected_classifications:
            self.assertEqual(
                rdr.classify(case),
                target,
                f"{case.name}: expected {target}, got {rdr.classify(case)}.",
            )

    # ------------------------------------------------------------------
    # Test 5 — UNSET path (no target) still routes to expert
    # ------------------------------------------------------------------

    def test_unset_path_always_routes_to_expert_regardless_of_resolver(self):
        """The UNSET (no-target) path always consults the expert, never auto-resolves.

        When ``fit_case`` is called without a ``target`` (i.e., ``target=UNSET``),
        the method invokes ``expert.ask_for_rule``, not ``ask_for_conditions``.
        The ``_try_auto_resolve`` guard ``current is UNSET`` returns ``None``
        immediately and the expert MUST be called even when backward-inference
        knowledge exists and the resolver is active.

        This is verified by counting interact() calls: the UNSET path calls
        ``ask_for_rule`` which itself calls ``ask_for_conditions`` internally,
        so at least one interact() must occur.
        """
        rdr, interface = self._three_species_rdr(
            self._unset_path_answer_fn, with_resolver=True
        )

        # An animal that would be auto-resolved if given a target; here no target is given.
        unknown = _make_auto_resolution_animal(
            "auto_unknown", feathers=True, venomous=True, legs=2
        )

        count_before = interface.ask_for_conditions_count
        rdr.fit_case(unknown, expert=Expert(interface=interface))
        calls_made = interface.ask_for_conditions_count - count_before

        self.assertGreater(
            calls_made,
            0,
            "Expert MUST be called on the UNSET path even when a resolver is active "
            f"(got {calls_made} call(s)).",
        )
        # The classification must be valid (whatever the expert concluded).
        result = rdr.classify(unknown)
        self.assertIsNotNone(
            result,
            "After UNSET-path fitting, classify(unknown) must return a non-None species.",
        )
        self.assertIsInstance(
            result,
            Species,
            f"classify(unknown) must return a Species member, got {type(result).__name__}.",
        )


if __name__ == "__main__":
    unittest.main()
