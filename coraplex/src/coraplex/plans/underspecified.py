from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Type, TYPE_CHECKING, Iterator

from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.executables import (
    Executable,
    GiskardExecutable,
    UnderspecifiedExecutable,
)
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionNode, ExecutionBoundaryNode
from krrood.entity_query_language.query.match import Match

if TYPE_CHECKING:
    from coraplex.datastructures.dataclasses import Context
    from coraplex.robot_plans.actions.base import ActionDescription
    from semantic_digital_twin.world import World


# %% trying a grounded action out before it is executed for real


@dataclass
class ActionTrial:
    """
    Tries grounded actions against a disposable copy of the world, to check that a
    candidate can succeed before it is attempted for real.

    One copy serves every candidate: after an attempt the copy is rolled back to the
    model version it was at and its state is restored, so the next candidate starts from
    the same point without another copy having to be made. A fresh copy is taken
    whenever `context.world` has itself moved on, so a trial always reflects the state
    and model changes actually in it.

    The copy is never connected to a synchronizer, so nothing a trial does is published,
    and a trial always runs under a forced
    :attr:`~coraplex.datastructures.enums.ExecutionType.SIMULATED` execution regardless
    of the execution type the real attempt will use. Conditions are always evaluated
    too: whether a candidate is worth attempting for real is exactly what its pre- and
    postconditions decide, so a plan that skips them elsewhere does not skip them here.
    """

    context: Context
    """
    The context the candidates were grounded in.

    Only ever read from: a trial never mutates it or the world it points at, and the
    candidates themselves are left untouched too, so they can still be attached and
    executed for real afterwards.
    """

    _world: Optional[World] = field(default=None, init=False, repr=False)
    """
    The copy candidates are tried against, kept until it no longer matches the world it
    was taken from.
    """

    _context: Optional[Context] = field(default=None, init=False, repr=False)
    """
    The context pointing at `_world`, rebuilt whenever a new copy is taken.
    """

    _source_versions: Optional[Tuple[int, int]] = field(
        default=None, init=False, repr=False
    )
    """
    The model and state versions `context.world` had when the copy was taken, used to
    notice that it has moved on and the copy has to be replaced.
    """

    def succeeds(self, action: ActionDescription) -> bool:
        """
        Run `action` against the copy and restore the copy afterwards.

        The action is copied onto the copy first: reading through a reference to the
        world it was grounded in would be harmless, but an action that modifies the
        model (attaching a grasped body, say) requires the entities it is given to
        belong to the world being modified.

        The version to roll back to is read here rather than when the copy is taken, so
        each attempt undoes only its own modifications. Reverting is itself recorded, so
        rolling every attempt back to where the copy started would mean undoing a longer
        and longer run of blocks, most of them already-undone ones.

        :param action: The grounded action to try out.
        :return: True if `action` runs to completion without raising a `PlanFailure`.
        """
        world = self._copy()
        plan = Plan(context=self._context)
        candidate = ActionNode(designator=world.rebind_world_entities(action))
        plan.add_node(candidate)
        version = world.get_world_model_manager().version

        with world.reset_state_context(), ExecutionEnvironment(
            ExecutionType.SIMULATED,
            collision_avoidance=GiskardExecutable.collision_avoidance,
        ):
            try:
                candidate.perform()
                return True
            except PlanFailure:
                return False
            finally:
                # Undo the model changes before leaving the reset context restores the
                # state, which needs the degrees of freedom it was snapshotted with.
                world.rollback_to_version(version)

    def _copy(self) -> World:
        """
        :return: The copy to try candidates against, taken again if `context.world` has
            changed since the current one was made.
        """
        versions = (
            self.context.world.get_world_model_manager().version,
            self.context.world.state.version,
        )
        if self._world is None or self._source_versions != versions:
            self._world = deepcopy(self.context.world)
            self._context = replace(
                self.context,
                world=self._world,
                robot=self._world.get_semantic_annotation_by_id(self.context.robot.id),
                evaluate_conditions=True,
            )
            self._source_versions = versions
        return self._world

    def discard(self) -> None:
        """
        Release the copy, so the next trial takes a fresh one.
        """
        self._world = None
        self._context = None
        self._source_versions = None


# %% resolving an underspecified action to a candidate that works


@dataclass(eq=False, repr=False)
class UnderspecifiedNode(ExecutionBoundaryNode):
    """
    An action or language expression that is described by an underspecified `an(...)`
    match statement.

    This node is used to generate fully specified actions  or language expressions.
    The semantics are: try until it succeeds or fails if the underspecified action is exhausted.
    If you want to limit the number of attempts, add a limit clause to the underspecified action.
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

    _trial: Optional[ActionTrial] = field(default=None, init=False, repr=False)
    """
    The trial every candidate of this node is tried against.

    Held across candidates so they share one copy of the world, rather than each paying
    for its own.
    """

    @property
    def designator_type(self) -> Type:
        return self.underspecified_action.type

    def _pull_next_action(self) -> Optional[ActionDescription]:
        """
        Pull the next grounded action from the iterator, without attaching it anywhere.

        :return: The next grounded action, or None if the iterator is exhausted.
        """
        if self._action_iterator is None:
            self._action_iterator = self.context.query_backend.evaluate(
                self.underspecified_action
            )

        action = next(self._action_iterator, None)
        if action is None:
            self._action_iterator = None
        return action

    def _attach(self, action: ActionDescription) -> ActionNode:
        """
        Wrap a grounded action in an `ActionNode` and add it as this node's child.

        :param action: The grounded action to attach.
        :return: The new candidate node.
        """
        candidate = ActionNode(designator=action)
        self.add_child(candidate)
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
        released for the same reason.
        """
        if self._action_iterator is not None:
            self._action_iterator.close()
            self._action_iterator = None
        if self._trial is not None:
            self._trial.discard()

    def notify(self):
        # Resolution is deferred to execution time: the underspecified statement can
        # only be grounded once the preceding actions have run and mutated the world
        # (e.g. the torso is raised, the object is in the gripper). The grounding
        # happens in UnderspecifiedExecutable, so expansion does nothing here.
        pass

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
        if self._trial is None:
            self._trial = ActionTrial(context=self.context)

        action = self._pull_next_action()
        while action is not None:
            if self._trial.succeeds(action):
                self._attach(action)
                self.current_candidate.notify()
                return True
            action = self._pull_next_action()
        return False

    def parse(self) -> Executable:
        # Defer resolution to execution: the returned executable grounds the action
        # when it is reached, against the world state produced by the preceding nodes.
        return UnderspecifiedExecutable(node=self, context=self.context)

    def __repr__(self):
        return f"{self.designator_type.__name__}"
