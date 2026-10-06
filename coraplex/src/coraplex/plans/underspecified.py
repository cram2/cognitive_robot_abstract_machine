from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import Optional, Tuple, Type, TYPE_CHECKING, Iterator

from coraplex.datastructures.enums import ActionTrialVisualization, ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.language import SequentialNode
from coraplex.plans.executables import (
    Executable,
    GiskardExecutable,
    UnderspecifiedExecutable,
)
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, ExecutionBoundaryNode
from coraplex.visualization import RvizVisualization
from krrood.entity_query_language.query.match import Match

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context
    from coraplex.robot_plans.actions.base import ActionDescription
    from semantic_digital_twin.world import World


# %% trying a grounded action out before it is executed for real


@dataclass
class ActionTrial:
    """
    Tries grounded actions against a copy of the world, to check that a candidate can
    succeed before it is attempted for real.

    One copy serves every candidate: after each attempt its model is rolled back and its
    state restored, and when `context.world` has changed since, the copy replays those
    model and state changes instead of being taken anew. Collision rules changed after
    the copy was taken are not carried over.

    Trials never publish to a synchronizer, always run as
    :attr:`~coraplex.datastructures.enums.ExecutionType.SIMULATED`, and always evaluate
    pre- and postconditions. While the context is debugging, the copy is shown in RViz
    under its own frame prefix and marker topic.
    """

    context: Context
    """
    The context the candidates were grounded in.

    Only ever read from: a trial never mutates it or the world it points at, and the
    candidates themselves are left untouched too, so they can still be attached and
    executed for real afterwards.
    """

    _copied_context: Optional[Context] = field(default=None, init=False, repr=False)
    """
    The context pointing at the copy candidates are tried against, kept until that copy
    no longer matches the world it was taken from.
    """

    _source_versions: Optional[Tuple[int, int]] = field(
        default=None, init=False, repr=False
    )
    """
    The model and state versions `context.world` had when the copy last matched it, used
    to notice that it has moved on and the copy has to be caught up.
    """

    _replayed_modification_blocks: int = field(default=0, init=False, repr=False)
    """
    How many of the modification blocks of `context.world` the copy already holds.
    """

    copy_marker_alpha: float = field(default=0.9, kw_only=True)
    """
    The opacity the copy is drawn with while debugging, so it can be told apart from the
    world it copies where the two overlap.
    """

    _visualization: Optional[RvizVisualization] = field(
        default=None, init=False, repr=False
    )
    """
    The RViz publishing of the current copy, while the context is debugging.
    """

    def succeeds(self, action: ActionDescription) -> bool:
        """
        Run `action` against the copy and restore the copy afterwards.

        The action is copied onto the copy first (see :meth:`_on_the_copy`): reading
        through a reference to the world it was grounded in would be harmless, but an
        action that modifies the model (attaching a grasped body, say) requires the
        entities it is given to belong to the world being modified.

        The version to roll back to is read here rather than when the copy is taken, so
        each attempt undoes only its own modifications. Reverting is itself recorded, so
        rolling every attempt back to where the copy started would mean undoing a longer
        and longer run of blocks, most of them already-undone ones.

        The action is tried inside a sequence of its own, the way it is executed for
        real, so that the nodes a plan transformation puts beside it are tried with it
        rather than failing on a candidate that has no siblings.

        :param action: The grounded action to try out.
        :return: True if `action` runs to completion without failing.
        """
        context = self._copy()
        world = context.world
        plan = Plan(context=context)
        candidate_sequence = SequentialNode()
        candidate = ActionNode(designator=self._on_the_copy(action, world))
        plan.add_node(candidate_sequence)
        candidate_sequence.add_child(candidate)
        version = world.get_world_model_manager().version

        with (
            world.reset_state_context(),
            ExecutionEnvironment(
                ExecutionType.SIMULATED,
                collision_avoidance=GiskardExecutable.collision_avoidance,
            ),
        ):
            try:
                candidate_sequence.perform()
                return True
            except PlanFailure:
                return False
            finally:
                if plan.action_trial is not None:
                    plan.action_trial.discard()
                # Undo the model changes before leaving the reset context restores the
                # state, which needs the degrees of freedom it was snapshotted with.
                world.rollback_to_version(version)

    @staticmethod
    def _on_the_copy(action: ActionDescription, world: World) -> ActionDescription:
        """
        :param action: The grounded action to try out.
        :param world: The copy to try it against.
        :return: A new action with the parameters of `action`, referring to `world`.
            Only the parameters are carried over: the plan node of `action` belongs to
            the plan it was grounded in, not to the trial.
        """
        return replace(
            action,
            **{
                parameter.name: world.rebind_world_entities(
                    getattr(action, parameter.name)
                )
                for parameter in fields(action)
                if parameter.init
            },
        )

    def _copy(self) -> Context:
        """
        :return: The context pointing at the copy to try candidates against, caught up
            with `context.world` if that has changed since the copy last matched it.
        """
        versions = (
            self.context.world.get_world_model_manager().version,
            self.context.world.state.version,
        )
        if self._copied_context is None:
            self._take_copy()
        elif self._source_versions != versions:
            self._catch_up()
        self._source_versions = versions
        return self._copied_context

    def _take_copy(self) -> None:
        """
        Copy `context.world` and, while the context is debugging, start publishing the
        copy.
        """
        world = deepcopy(self.context.world)
        self._replayed_modification_blocks = len(
            self.context.world.get_world_model_manager().model_modification_blocks
        )
        self._copied_context = replace(
            self.context,
            world=world,
            robot=world.get_semantic_annotation_by_id(self.context.robot.id),
            evaluate_conditions=True,
        )
        if self.context.debug:
            self._visualization = RvizVisualization(
                world,
                ros_node=self.context.ros_node,
                collision_visualization=True,
                frame_prefix=ActionTrialVisualization.FRAME_PREFIX,
                marker_topic=ActionTrialVisualization.MARKER_TOPIC,
                marker_alpha=self.copy_marker_alpha,
            ).start()

    def _catch_up(self) -> None:
        """
        Bring the copy up to date with `context.world`: replay the modifications made to
        it since, the way copying it replays all of them, and take over its state.

        The copy's own modifications are all rolled back by then, so it still matches
        the world as it was when it last caught up.
        """
        modification_blocks = (
            self.context.world.get_world_model_manager().model_modification_blocks
        )
        world = self._copied_context.world
        with world.modify_world():
            for block in modification_blocks[self._replayed_modification_blocks :]:
                block.update_references_for_world_and_apply(world=world)
            world.state.merge_state(self.context.world.state)
        self._replayed_modification_blocks = len(modification_blocks)

    def discard(self) -> None:
        """
        Release the copy, so the next trial takes a fresh one.
        """
        self._stop_visualization()
        self._copied_context = None
        self._source_versions = None

    def _stop_visualization(self) -> None:
        """
        Stop publishing the current copy, if it is being published.
        """
        if self._visualization is None:
            return
        self._visualization.stop()
        self._visualization = None


# %% resolving an underspecified action to a candidate that works


@dataclass(eq=False, repr=False)
class UnderspecifiedNode(ExecutionBoundaryNode):
    """
    An action or language expression that is described by an underspecified `an(...)`
    match statement.

    This node is used to generate fully specified actions  or language expressions.
    The semantics are: try until it succeeds or fails if the underspecified action is exhausted.
    It tries at most :attr:`candidate_limit` candidates.
    """

    underspecified_action: Match = field(kw_only=True)
    """
    The underspecified statement that can be used to generate actions.
    """

    _action_iterator: Optional[Iterator[ActionDescription]] = field(
        default=None, kw_only=True
    )
    """
    The iterator that is used to generate the actions.

    Only available after the first call to notify.
    """

    current_candidate: Optional[ActionNode] = field(
        default=None, init=False, repr=False
    )
    """
    The action candidate this node currently resolves to, set by `advance` at execution
    time.

    On failure, `advance` replaces it with the next candidate.
    """

    current_candidate_sequence: Optional[SequentialNode] = field(
        default=None, init=False, repr=False
    )
    """
    The sequence that is executed for the current candidate.

    It holds the candidate and everything a plan transformation put beside it, so that
    those nodes are part of what this node runs rather than being skipped.
    """

    _candidates_pulled: int = field(default=0, init=False, repr=False)
    """
    How many candidates the current run through the underspecified statement has
    produced.
    """

    _transformations_applied: bool = field(default=False, init=False, repr=False)
    """
    Whether the plan transformations matching this node have rewritten the plan around
    it already, so that expanding the plan again does not apply them a second time.
    """

    @property
    def designator_type(self) -> Type:
        return self.underspecified_action._type_

    @property
    def candidate_limit(self) -> int:
        """
        :return: How many candidates this node tries: the underspecified statement's own
            limit, or the context's if it has none.
        """
        return self.underspecified_action._limit_ or self.context.candidates_to_try

    @property
    def reached_candidate_limit(self) -> bool:
        """
        :return: Whether the last run through the underspecified statement stopped
            because it produced :attr:`candidate_limit` candidates.
        """
        return self._candidates_pulled == self.candidate_limit

    @property
    def trial(self) -> ActionTrial:
        """
        The trial every candidate of this node is tried against.

        It is the trial of this node's plan, shared with every other underspecified node
        of the plan, so they all try their candidates in one copy of the world rather
        than each paying for its own.
        """
        if self.plan.action_trial is None:
            self.plan.action_trial = ActionTrial(context=self.context)
        return self.plan.action_trial

    def _pull_next_action(self) -> Optional[ActionDescription]:
        """
        Pull the next grounded action from the iterator, without attaching it anywhere.

        :return: The next grounded action, or None if the iterator is exhausted or has
            produced :attr:`candidate_limit` candidates already.
        """
        if self._action_iterator is None:
            self._action_iterator = self.context.query_backend.evaluate(
                self.underspecified_action
            )
            self._candidates_pulled = 0

        if self.reached_candidate_limit:
            self.stop_grounding()
            return None
        action = next(self._action_iterator, None)
        if action is None:
            self._action_iterator = None
            return None
        self._candidates_pulled += 1
        return action

    def _attach(self, action: ActionDescription) -> ActionNode:
        """
        Wrap a grounded action in an `ActionNode` and add it below a fresh attempt
        sequence of this node.

        :param action: The grounded action to attach.
        :return: The new candidate node.
        """
        candidate_sequence = SequentialNode()
        candidate = ActionNode(designator=action)
        self.add_child(candidate_sequence)
        candidate_sequence.add_child(candidate)
        self.current_candidate_sequence = candidate_sequence
        self.current_candidate = candidate
        return candidate

    def stop_grounding(self) -> None:
        """
        Release the action iterator once no further candidate will be requested from it.

        Between candidates the iterator is left suspended (rather than exhausted) so a
        later retry can resume the search instead of restarting it; a suspended
        generator keeps every value its frame holds alive, including resources a
        candidate generator only builds to validate against (for example a location's
        deep-copied test world). Once a candidate is accepted and no retry will happen,
        closing the iterator here releases those resources immediately instead of
        retaining them for this node's whole lifetime. The trial's copy of the world is
        kept, since the plan's other underspecified nodes try their candidates in it
        too.
        """
        if self._action_iterator is not None:
            self._action_iterator.close()
            self._action_iterator = None

    def notify(self):
        # Resolution is deferred to execution time: the underspecified statement can
        # only be grounded once the preceding actions have run and mutated the world
        # (e.g. the torso is raised, the object is in the gripper). The grounding
        # happens in UnderspecifiedExecutable, so expansion only lets the plan
        # transformations matching this node rewrite the plan around it.
        if self._transformations_applied:
            return
        self._transformations_applied = True
        self.plan.apply_plan_transformations(self)

    def advance(self) -> bool:
        """
        Resolve the next candidate that survives a trial, and expand it against the
        current world state.

        Every grounded action is first tried against a disposable copy of the world
        (:class:`ActionTrial`), which is rolled back between candidates; a candidate that
        fails there is discarded without ever being attached to the plan or touching the
        real world, so a bad parameterization cannot poison a later attempt. Only a
        candidate that survives its trial is attached and returned.

        Driven by :class:`~pycram.plans.executables.UnderspecifiedExecutable` to ground the
        action at execution time, and reused by failure handling to retry with a freshly
        generated action.

        :return: True if a new candidate was generated, False if the iterator is
            exhausted without any candidate surviving its trial.
        """
        action = self._pull_next_action()
        while action is not None:
            if self.trial.succeeds(action):
                self._attach(action)
                self.current_candidate_sequence.notify()
                return True
            action = self._pull_next_action()
        return False

    def parse(self) -> Executable:
        # Defer resolution to execution: the returned executable grounds the action
        # when it is reached, against the world state produced by the preceding nodes.
        return UnderspecifiedExecutable(node=self, context=self.context)

    def __repr__(self):
        return f"{self.designator_type.__name__}"
