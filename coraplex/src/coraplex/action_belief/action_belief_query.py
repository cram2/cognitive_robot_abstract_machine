from __future__ import annotations

import dataclasses
import itertools
from copy import deepcopy
from dataclasses import dataclass

from typing_extensions import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

from krrood.entity_query_language.factories import (
    evaluate_condition,
    get_false_statements,
)
from coraplex.action_belief.action_belief_space import ACTION_BELIEF_SPACES
from coraplex.action_belief.results import ActionBeliefResult
from coraplex.datastructures.dataclasses import Context
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.view_manager import ViewManager
from semantic_digital_twin.robots.robot_parts import AbstractRobot


def _candidate_key(candidate: Dict[str, Any]) -> str:
    """
    :return: A deterministic string key for a candidate, for looking it up in an
        :class:`~coraplex.action_belief.action_belief_space.ActionBeliefSpace`'s prior.
    """
    return str(
        sorted(candidate.items(), key=lambda field_and_value: field_and_value[0])
    )


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

    max_location_candidates: int = 10
    """
    Cutoff for how many candidates to evaluate eagerly before ranking: how many poses
    :meth:`enumerate_candidates` draws from a bucket-B choice point's ``Location``, and
    how many already-grounded actions :meth:`rank_grounded_actions` evaluates up front.

    Bounding an otherwise-lazy stream to a small prefix keeps both cheap.
    """

    @property
    def _choice_points(self):
        return ACTION_BELIEF_SPACES[self.action_type].choice_points

    def enumerate_candidates(self) -> Iterator[Dict[str, Any]]:
        """
        :return: Every combination of choice-point field name to candidate value,
            drawn from bucket A's declared domain or bucket B's ``Location``, in the
            order ``itertools.product`` produces them.
        """
        field_names = [point.field_name for point in self._choice_points]
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
            for point in self._choice_points
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
        for point in self._choice_points:
            value = action
            for segment in point.field_name.split("."):
                value = getattr(value, segment)
            candidate[point.field_name] = value
        return candidate

    def perturbed_action(
        self, action: ActionDescription, field_name: str, value: Any
    ) -> ActionDescription:
        """
        :return: A copy of ``action`` with ``field_name`` overridden to ``value``;
            every other constructor kwarg is taken from ``action`` itself.

        :attr:`fixed_kwargs` must equal ``action.designator_parameter`` for this to
        carry over the rest of ``action``'s own field values correctly -- this is a
        different use of that field than :meth:`enumerate_candidates`'s (a template
        held fixed across many candidates), one field template for one action here.
        """
        kwargs = self._materialize_kwargs({field_name: value}, robot=action.robot)
        return self.action_type(**kwargs)

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
