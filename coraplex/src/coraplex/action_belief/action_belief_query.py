from __future__ import annotations

import dataclasses
import itertools
import math
from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

from krrood.entity_query_language.factories import (
    evaluate_condition,
    get_false_statements,
)
from coraplex.action_belief.action_belief_space import (
    ACTION_BELIEF_SPACES,
    ActionBeliefPoint,
)
from coraplex.action_belief.results import ActionBeliefResult
from coraplex.datastructures.dataclasses import Context
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.view_manager import ViewManager
from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% candidate keys


def _candidate_key(candidate: Dict[str, Any]) -> str:
    """
    :return: A deterministic string key for a candidate, for looking it up in an
        :class:`~coraplex.action_belief.action_belief_space.ActionBeliefSpace`'s prior.

    Only meaningful for bucket-A fields: it stringifies each value verbatim, so a
    bucket-B field's value (e.g. a ``Pose``) would need an exact ``repr()`` match to
    ever hit a ``prior`` entry, which is impractical for continuous values.
    """
    return str(
        sorted(candidate.items(), key=lambda field_and_value: field_and_value[0])
    )


_EAGER_EVALUATION_CEILING = 3
"""
Upper bound on how many candidates :class:`ActionBeliefQuery` ever evaluates eagerly
before ranking, regardless of a registered type's own domain size.

Each eager evaluation deep-copies the live world (:meth:`ActionBeliefQuery._fresh_context`)
to check one candidate in isolation, and that throwaway world is not reliably released
afterward. This value was found empirically, against the full ``coraplex`` test suite
(not just the one module that exercises it most): 3 is clean, 4 already reproduces the
world-leak failure in ``test/conftest.py`` (``count_worlds``) at
``test_transport_open_container[pr2]`` in
``test/coraplex_test/test_designator/test_multi_robot_action_designator.py`` -- a
composite ``TransportAction`` chains several registered choice points (two
``NavigateAction`` groundings, ``PickUpAction``, ``PlaceAction``) across four robot
parametrizations in that module, and the margin against the suite's leak-detection
threshold is this tight. Raising this ceiling is not safe without first addressing why
a throwaway world outlives the candidate check it was built for.
"""


@dataclass
class ActionBeliefQuery:
    """
    Enumerates the registered choice points of one action grounding and ranks them by
    whether ``pre_condition`` holds, evaluated on a throwaway copy of the live world.
    """

    action_type: Type[ActionDescription]
    """
    The action type being ranked.

    Must be a key in
    :data:`~coraplex.action_belief.action_belief_space.ACTION_BELIEF_SPACES`.
    """

    fixed_kwargs: Dict[str, Any]
    """
    Constructor kwargs held fixed across every candidate.

    For a bucket-B choice point, the entry under its field name must be the
    :class:`~coraplex.locations.base.Location` sampled for that field, not a concrete
    pose.
    """

    context: Context
    """
    Context of the live plan.

    Only ever read from; :meth:`evaluate_candidate` checks candidates against a deep
    copy, never this context's own world.
    """

    prior: Optional[Dict[str, float]] = None
    """
    Prior weight per candidate, keyed by :func:`_candidate_key`.

    ``None`` falls back to the registered
    :class:`~coraplex.action_belief.action_belief_space.ActionBeliefSpace`'s own
    prior (uniform if that is also unset).
    """

    max_location_candidates: Optional[int] = None
    """
    Cutoff for how many candidates to evaluate eagerly before ranking: how many poses
    :meth:`enumerate_candidates` draws from a bucket-B choice point's ``Location``, and
    how many already-grounded actions :meth:`rank_grounded_actions` evaluates up front.

    ``None`` (the default) resolves in :meth:`__post_init__` via
    :meth:`_default_max_location_candidates`, scaling with :attr:`action_type`'s own
    domain: a type with a small bucket-A domain (e.g. ``OpenAction``'s two arms) gets
    it fully ranked for free, while a larger bucket-A domain (e.g. ``ReachAction``'s 48
    arm/approach/alignment/reverse combinations) or any bucket-B choice point is capped
    at :data:`_EAGER_EVALUATION_CEILING` instead of paying for its full domain.

    Pass an explicit value to override this for one query -- e.g. in a test that wants
    a specific prefix length regardless of the registered domain.
    """

    def __post_init__(self) -> None:
        if self.max_location_candidates is None:
            self.max_location_candidates = self._default_max_location_candidates()

    def _default_max_location_candidates(self) -> int:
        """
        :return: :attr:`action_type`'s own bucket-A domain size (the product of every
            registered bucket-A choice point's domain length), capped at
            :data:`_EAGER_EVALUATION_CEILING` -- or the ceiling itself if
            :attr:`action_type` has a bucket-B (``Location``-backed) choice point,
            since a ``Location`` can produce arbitrarily many poses and has no fixed
            domain to size against.
        """
        if any(point.bucket == "B" for point in self.choice_points):
            return _EAGER_EVALUATION_CEILING
        domain_size = math.prod(
            (len(point.domain) for point in self.choice_points), start=1
        )
        return min(domain_size, _EAGER_EVALUATION_CEILING)

    # %% construction

    @classmethod
    def for_registered_type(
        cls, action_type: Type[ActionDescription], context: Context
    ) -> Optional["ActionBeliefQuery"]:
        """
        :return: An :class:`ActionBeliefQuery` for ``action_type`` with no fixed
            kwargs, or ``None`` if ``action_type`` isn't registered in
            :data:`~coraplex.action_belief.action_belief_space.ACTION_BELIEF_SPACES`.

        For callers that only rank or check already-grounded actions (e.g.
        :meth:`rank_grounded_actions`, :meth:`predict_feasible`), which don't need a
        real :attr:`fixed_kwargs` template.
        """
        if action_type not in ACTION_BELIEF_SPACES:
            return None
        return cls(action_type=action_type, fixed_kwargs={}, context=context)

    @property
    def choice_points(self) -> List[ActionBeliefPoint]:
        """
        :return: Every registered choice point for :attr:`action_type`.
        """
        return ACTION_BELIEF_SPACES[self.action_type].choice_points

    # %% candidate construction

    def enumerate_candidates(self) -> Iterator[Dict[str, Any]]:
        """
        :return: Every combination of choice-point field name to candidate value,
            drawn from bucket A's declared domain or bucket B's ``Location``, in the
            order ``itertools.product`` produces them.
        """
        field_names = [point.field_name for point in self.choice_points]
        value_options = [
            (
                point.domain
                if point.bucket == "A"
                else list(
                    itertools.islice(
                        self.fixed_kwargs[point.field_name],
                        self.max_location_candidates,
                    )
                )
            )
            for point in self.choice_points
        ]
        for combination in itertools.product(*value_options):
            yield dict(zip(field_names, combination))

    def _materialize_kwargs(
        self, candidate: Dict[str, Any], robot: AbstractRobot
    ) -> Dict[str, Any]:
        """
        :return: Constructor kwargs for :attr:`action_type` with ``candidate``
            applied on top of :attr:`fixed_kwargs`. A nested field (e.g.
            ``grasp_description.approach_direction``) is applied via
            :func:`dataclasses.replace` on its :attr:`fixed_kwargs` template object.
            Supports exactly one level of dotted nesting -- no registered choice
            point nests deeper than that today.

        If ``candidate`` sets ``arm``, ``grasp_description.end_effector`` is
        re-resolved from it, since the grasp orientation is computed relative to
        that end effector and it is not itself a registered choice point.
        """
        kwargs = dict(self.fixed_kwargs)
        nested_overrides: Dict[str, Dict[str, Any]] = {}
        for dotted_field_name, value in candidate.items():
            *nested_path, leaf_name = dotted_field_name.split(".")
            if not nested_path:
                kwargs[leaf_name] = value
                continue
            nested_overrides.setdefault(nested_path[0], {})[leaf_name] = value
        for top_level_name, overrides in nested_overrides.items():
            kwargs[top_level_name] = dataclasses.replace(
                kwargs[top_level_name], **overrides
            )
        if "arm" in candidate and "grasp_description" in kwargs:
            kwargs["grasp_description"] = dataclasses.replace(
                kwargs["grasp_description"],
                end_effector=ViewManager.get_end_effector_view(kwargs["arm"], robot),
            )
        return kwargs

    def extract_candidate(self, action: ActionDescription) -> Dict[str, Any]:
        """
        :return: The current value of each registered choice-point field, read
            directly off ``action`` -- the inverse of :meth:`_materialize_kwargs`.
        """
        candidate = {}
        for point in self.choice_points:
            value = action
            for segment in point.field_name.split("."):
                value = getattr(value, segment)
            candidate[point.field_name] = value
        return candidate

    def perturb(
        self, action: ActionDescription, changes: Dict[str, Any]
    ) -> ActionDescription:
        """
        :return: A copy of ``action`` with every field in ``changes`` overridden to
            its paired value; every other constructor kwarg is taken from ``action``
            itself.

        :attr:`fixed_kwargs` must equal ``action.designator_parameter`` for this to
        carry over the rest of ``action``'s own field values correctly -- this is a
        different use of that field than :meth:`enumerate_candidates`'s (a template
        held fixed across many candidates), one field template for one action here.
        """
        kwargs = self._materialize_kwargs(changes, robot=action.robot)
        return self.action_type(**kwargs)

    def perturbed_action(
        self, action: ActionDescription, field_name: str, value: Any
    ) -> ActionDescription:
        """
        :return: A copy of ``action`` with ``field_name`` overridden to ``value``;
            every other constructor kwarg is taken from ``action`` itself.

        A single-field convenience wrapper around :meth:`perturb`.
        """
        return self.perturb(action, {field_name: value})

    # %% feasibility evaluation

    def _fresh_context(self) -> Context:
        """
        :return: A :class:`~coraplex.datastructures.dataclasses.Context` wrapping a
            throwaway deep copy of the live world and robot, so a candidate can be
            checked without mutating the live plan state.
        """
        world = deepcopy(self.context.world)
        robot = world.get_semantic_annotation_by_id(self.context.robot.id)
        return Context(
            world=world,
            robot=robot,
            alternative_motion_mappings=self.context.alternative_motion_mappings,
        )

    @staticmethod
    def _evaluate_pre_condition(
        action: ActionDescription, fresh_context: Context
    ) -> Tuple[bool, Optional[str]]:
        """
        :return: Whether ``action``'s ``pre_condition`` holds under ``fresh_context``,
            and (if not) which statements were false.
        """
        condition = action.pre_condition(
            action.bound_variables, fresh_context, action.designator_parameter
        )
        if evaluate_condition(condition):
            return True, None
        false_statements = get_false_statements(condition)
        return False, ", ".join(statement._name_ for statement in false_statements)

    def evaluate_candidate(
        self, candidate: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        :return: Whether ``candidate``'s ``pre_condition`` holds against a throwaway
            copy of the live world, and (if not) which statements were false.
        """
        fresh_context = self._fresh_context()
        action = self.action_type(
            **self._materialize_kwargs(candidate, fresh_context.robot)
        )
        return self._evaluate_pre_condition(action, fresh_context)

    def predict_feasible(self, action: ActionDescription) -> bool:
        """
        :return: Whether ``action``'s ``pre_condition`` holds against a throwaway
            copy of the live world.

        Unlike :meth:`evaluate_candidate`, this takes an already fully-specified
        action rather than a candidate dict, for callers that only need a yes/no
        prediction for one concrete grounding.
        """
        passed, _ = self._evaluate_pre_condition(action, self._fresh_context())
        return passed

    # %% ranking and reporting

    def rank_grounded_actions(
        self, actions: Iterable[ActionDescription]
    ) -> Iterator[ActionDescription]:
        """
        Reorder already-grounded actions by predicted feasibility, without dropping
        any: this ranks a query backend's output for retry ordering, unlike
        :meth:`run`, which builds and evaluates its own candidates from scratch.

        Only the first :attr:`max_location_candidates` of ``actions`` are evaluated
        and ranked eagerly. Each evaluation re-checks ``pre_condition`` against a
        fresh deep copy of the world, so for a registered type with a bucket-B
        choice point (e.g. ``NavigateAction.target_location``) the query backend can
        supply far more candidates than are ever useful to check up front. Anything
        beyond that prefix is appended afterward, unranked, in its original
        streaming order -- still triable, just not evaluated ahead of time.

        :param actions: Already-grounded instances of :attr:`action_type`, e.g. from
            ``context.query_backend.evaluate(...)``.
        :return: The ranked prefix (``pre_condition``-holding candidates first, by
            prior weight, then the ones that failed), followed by the unranked
            remainder of ``actions``.
        """
        action_iterator = iter(actions)
        buffered = list(itertools.islice(action_iterator, self.max_location_candidates))
        prior = (
            self.prior
            if self.prior is not None
            else ACTION_BELIEF_SPACES[self.action_type].prior
        )

        feasible: List[Tuple[ActionDescription, float]] = []
        infeasible: List[ActionDescription] = []
        for action in buffered:
            if self.predict_feasible(action):
                weight = (
                    prior.get(_candidate_key(self.extract_candidate(action)), 1.0)
                    if prior
                    else 1.0
                )
                feasible.append((action, weight))
            else:
                infeasible.append(action)

        feasible.sort(key=lambda action_and_weight: action_and_weight[1], reverse=True)
        for action, _ in feasible:
            yield action
        yield from infeasible
        yield from action_iterator

    def run(self) -> ActionBeliefResult:
        """
        :return: The ranked result of evaluating every candidate
            :meth:`enumerate_candidates` produces.
        """
        candidates = list(self.enumerate_candidates())
        prior = (
            self.prior
            if self.prior is not None
            else ACTION_BELIEF_SPACES[self.action_type].prior
        )

        feasible: List[Dict[str, Any]] = []
        rejected_examples: List[Tuple[Dict[str, Any], str]] = []
        for candidate in candidates:
            passed, reason = self.evaluate_candidate(candidate)
            if passed:
                feasible.append(candidate)
            else:
                rejected_examples.append((candidate, reason))

        weights = [
            prior.get(_candidate_key(candidate), 1.0) if prior else 1.0
            for candidate in feasible
        ]
        total_weight = sum(weights) or 1.0
        ranked_candidates = sorted(
            zip(feasible, (weight / total_weight for weight in weights)),
            key=lambda candidate_and_posterior: candidate_and_posterior[1],
            reverse=True,
        )

        chosen, posterior = ranked_candidates[0] if ranked_candidates else ({}, 0.0)
        return ActionBeliefResult(
            action_type=self.action_type,
            chosen=chosen,
            posterior=posterior,
            candidates_enumerated=len(candidates),
            candidates_feasible=len(feasible),
            rejected_examples=rejected_examples,
            ranked_candidates=ranked_candidates,
        )
