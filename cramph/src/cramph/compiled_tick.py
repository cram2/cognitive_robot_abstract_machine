from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    TYPE_CHECKING,
    Self,
)

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import VariableParameters
from cramph.data_types import (
    LifeCycleValues,
    LifeCyclePredicate,
)
from cramph.exceptions import (
    TickDoesNotSettleError,
)
from cramph.node import (
    DerivedConditionVariable,
    StatechartNode,
)

if TYPE_CHECKING:
    from cramph.statechart import Statechart

# %% settling one tick


class PassInputKind(StrEnum):
    """
    A value per node that a pass through the statechart reads on top of the life
    cycle, observation and last observation states.
    """

    LIFE_CYCLE_AT_CYCLE_START = "life_cycle_at_cycle_start"
    """
    The life cycle state the node entered the tick with.
    """

    TICK_OBSERVATION = "tick_observation"
    """
    What :meth:`~cramph.node.StatechartNode.on_tick`
    returned for the node this tick.
    """

    HAS_TICK_OBSERVATION = "has_tick_observation"
    """
    Whether :meth:`~cramph.node.StatechartNode.on_tick`
    returned an observation for the node this tick.
    """

    OWN_TRANSITION_TAKEN = "own_transition_taken"
    """
    Whether the node already took a transition triggered by its own conditions this
    tick.
    """


@dataclass
class PassInput:
    """
    One value per node of a statechart that a pass reads, together with the
    variables standing for it in the compiled pass.
    """

    variables: list[sm.FloatVariable]
    """
    The variable of every node, in node order.
    """

    data: NDArray[np.float64]
    """
    The value of every node, in node order.
    """

    @classmethod
    def create(cls, kind: PassInputKind, nodes: list[StatechartNode]) -> Self:
        """
        :param kind: What the values stand for.
        :param nodes: The nodes to hold a value for, in node order.
        :return: An input holding zero for every node.
        """
        return cls(
            variables=[
                sm.FloatVariable(name=f"{node.life_cycle_variable.name}/{kind}")
                for node in nodes
            ],
            data=np.zeros(len(nodes)),
        )


@dataclass
class LifeCycleChange:
    """
    One node changing its life cycle state during a pass.
    """

    node: StatechartNode
    """
    The node whose life cycle state changed.
    """

    previous_state: LifeCycleValues
    """
    The life cycle state before the change.
    """

    new_state: LifeCycleValues
    """
    The life cycle state after the change.
    """

    def run_callback(self) -> None:
        """
        Calls the callback of :attr:`node` that matches this change, e.g.
        :meth:`~StatechartNode.on_start`. A node that starts paused gets
        :meth:`~StatechartNode.on_start` and then
        :meth:`~StatechartNode.on_pause`. A change with no dedicated callback calls
        nothing.
        """
        match (self.previous_state, self.new_state):
            case (_, LifeCycleValues.NOT_STARTED):
                self.node.on_reset()
            case (LifeCycleValues.NOT_STARTED, LifeCycleValues.RUNNING):
                self.node.on_start()
            case (LifeCycleValues.NOT_STARTED, LifeCycleValues.PAUSED):
                self.node.on_start()
                self.node.on_pause()
            case (LifeCycleValues.RUNNING, LifeCycleValues.PAUSED):
                self.node.on_pause()
            case (LifeCycleValues.PAUSED, LifeCycleValues.RUNNING):
                self.node.on_unpause()
            case (
                (LifeCycleValues.RUNNING | LifeCycleValues.PAUSED),
                _,
            ) if self.new_state.is_terminal:
                self.node.on_end()


@dataclass
class CompiledTick:
    """
    Brings every node of a statechart to the state it reaches in one tick.

    One compiled pass updates every node at once, reading the states the previous pass
    left. Passes repeat until no state changes, so how deeply nodes are nested does not
    change when they react to each other. Within a pass:

    1. A node observes if it was running when the tick started and is still
       running or paused. Its observation expression reads the states of the previous
       pass. A node that was paused when the tick started and is still running
       or paused keeps its observation. A node that stopped running during this tick keeps
       the observation it stopped on until the next one. Every other node
       observes Unknown.
    2. A node that has neither ended nor started the tick ended takes over its
       observation as its last observation.
    3. Every node takes its next life cycle transition, reading the observations of this
       pass and the life cycle state its parent reaches in this pass. A node takes at
       most one transition triggered by its own conditions per tick;
       transitions its parent forces on it always happen.

    .. note:: Life cycle callbacks run afterwards, once per change and in the order the
        changes happened, so no Python code runs between passes.
    """

    statechart: Statechart
    """
    The statechart whose nodes are updated.
    """

    _pass_budget: int = field(init=False)
    """
    The most passes that may change :attr:`statechart` within one tick,
    :attr:`~cramph.statechart.Statechart.pass_limit` extended by
    :attr:`~cramph.statechart.Statechart.passes_per_nesting_level` for every nesting level it has.
    """

    _nodes: list[StatechartNode] = field(init=False)
    """
    Every node of :attr:`statechart`, in node order.
    """

    _life_cycle_at_cycle_start: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.LIFE_CYCLE_AT_CYCLE_START`.
    """

    _tick_observation: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.TICK_OBSERVATION`.
    """

    _has_tick_observation: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.HAS_TICK_OBSERVATION`.
    """

    _own_transition_taken: PassInput = field(init=False)
    """
    See :attr:`PassInputKind.OWN_TRANSITION_TAKEN`.
    """

    _compiled_pass: sm.CompiledFunction = field(init=False)
    """
    One pass, compiled into one function by :meth:`compile`.
    """

    _next_observation: NDArray[np.float64] = field(init=False)
    """
    The observation of every node after the latest pass, a view on the pass output.
    """

    _next_last_observation: NDArray[np.float64] = field(init=False)
    """
    The last observation of every node after the latest pass, a view on the pass output.
    """

    _next_life_cycle: NDArray[np.float64] = field(init=False)
    """
    The life cycle state of every node after the latest pass, a view on the pass output.
    """

    _next_own_transition_taken: NDArray[np.float64] = field(init=False)
    """
    Whether every node took a transition triggered by its own conditions this tick,
    after the latest pass, a view on the pass output.
    """

    def compile(self) -> None:
        """
        Builds one pass through the statechart, compiles it and binds its inputs
        to the state arrays it reads, and to the world and float variable data of the
        context of :attr:`statechart`.
        """
        self._nodes = self.statechart.nodes
        deepest_nesting = max(node.depth for node in self._nodes)
        self._pass_budget = (
            self.statechart.pass_limit
            + self.statechart.passes_per_nesting_level * deepest_nesting
        )
        self._life_cycle_at_cycle_start = PassInput.create(
            PassInputKind.LIFE_CYCLE_AT_CYCLE_START, self._nodes
        )
        self._tick_observation = PassInput.create(
            PassInputKind.TICK_OBSERVATION, self._nodes
        )
        self._has_tick_observation = PassInput.create(
            PassInputKind.HAS_TICK_OBSERVATION, self._nodes
        )
        self._own_transition_taken = PassInput.create(
            PassInputKind.OWN_TRANSITION_TAKEN, self._nodes
        )
        self._compile_pass()

    def _compile_pass(self) -> None:
        """
        Compiles :meth:`_create_pass` and binds every input and output.
        """
        context = self.statechart.context
        inputs = [
            (
                [node.life_cycle_variable for node in self._nodes],
                self.statechart.life_cycle_state.data,
            ),
            (
                [node.observation_variable for node in self._nodes],
                self.statechart.observation_state.data,
            ),
            (
                [node.last_observation for node in self._nodes],
                self.statechart.last_observation_state.data,
            ),
            *[
                (pass_input.variables, pass_input.data)
                for pass_input in [
                    self._life_cycle_at_cycle_start,
                    self._tick_observation,
                    self._has_tick_observation,
                    self._own_transition_taken,
                ]
            ],
            (context.world.state.get_variables(), context.world.state._data),
            (
                context.float_variable_data.variables,
                context.float_variable_data.data,
            ),
        ]
        self._compiled_pass = self._create_pass().compile(
            parameters=VariableParameters.from_lists(
                *[variables for variables, _ in inputs]
            ),
            sparse=False,
        )
        for argument_index, (_, data) in enumerate(inputs):
            self._compiled_pass.bind_args_to_memory_view(
                arg_idx=argument_index, numpy_array=data
            )
        (
            self._next_observation,
            self._next_last_observation,
            self._next_life_cycle,
            self._next_own_transition_taken,
        ) = np.split(self._compiled_pass.evaluate(), 4)

    def _create_pass(self) -> sm.Vector:
        """
        :return: The observation, last observation, life cycle state and whether an own
            transition was taken of every node after one pass, concatenated in that
            order.
        """
        observations = [
            self._create_observation(node, index)
            for index, node in enumerate(self._nodes)
        ]
        last_observations = [
            sm.if_else(
                condition=sm.logic_or(
                    node.life_cycle_predicate_expression(
                        LifeCyclePredicate.IS_TERMINATED
                    ),
                    LifeCyclePredicate.IS_TERMINATED.expression(
                        self._life_cycle_at_cycle_start.variables[index]
                    ),
                ),
                if_result=node.last_observation,
                else_result=observations[index],
            )
            for index, node in enumerate(self._nodes)
        ]
        next_life_cycles, forced_life_cycles = self._create_next_life_cycles()
        life_cycles = self._read_this_pass_observations(
            sm.Vector(next_life_cycles + forced_life_cycles),
            observations=observations,
            last_observations=last_observations,
        )
        next_life_cycles = list(life_cycles)[: len(self._nodes)]
        forced_life_cycles = list(life_cycles)[len(self._nodes) :]
        own_transitions_taken = [
            sm.logic_or(own_transition_taken, next_life_cycle != forced_life_cycle)
            for next_life_cycle, forced_life_cycle, own_transition_taken in zip(
                next_life_cycles,
                forced_life_cycles,
                self._own_transition_taken.variables,
            )
        ]
        return sm.Vector(
            observations + last_observations + next_life_cycles + own_transitions_taken
        )

    def _create_observation(self, node: StatechartNode, index: int) -> sm.Scalar:
        """
        :param node: The node to build the observation for.
        :param index: The index of `node`.
        :return: What `node` observes after a pass.
        """
        observed = sm.if_else(
            condition=self._has_tick_observation.variables[index],
            if_result=self._tick_observation.variables[index],
            else_result=DerivedConditionVariable.substitute_in(
                node._observation_expression
            ),
        )
        life_cycle_at_cycle_start = self._life_cycle_at_cycle_start.variables[index]
        ran_at_cycle_start = LifeCyclePredicate.IS_RUNNING.expression(
            life_cycle_at_cycle_start
        )
        paused_at_cycle_start = LifeCyclePredicate.IS_PAUSED.expression(
            life_cycle_at_cycle_start
        )
        still_running_or_paused = sm.logic_or(
            node.life_cycle_predicate_expression(LifeCyclePredicate.IS_RUNNING),
            node.life_cycle_predicate_expression(LifeCyclePredicate.IS_PAUSED),
        )
        return sm.if_cases(
            cases=[
                (sm.logic_and(ran_at_cycle_start, still_running_or_paused), observed),
                (
                    sm.logic_and(paused_at_cycle_start, still_running_or_paused),
                    node.observation_variable,
                ),
                (
                    sm.logic_or(ran_at_cycle_start, paused_at_cycle_start),
                    node.last_observation,
                ),
            ],
            else_result=sm.Scalar.const_trinary_unknown(),
        )

    def _create_next_life_cycles(
        self,
    ) -> tuple[list[sm.Scalar], list[sm.Scalar]]:
        """
        Builds the life cycle state every node reaches in a pass, reading the state its
        parent reaches in the same pass, so a node never starts under a parent that stops
        running in that pass.

        Nodes are visited parents first but stored by node index, so an entry is
        ``None`` only until its node is visited; none is left once every node is.

        :return: The life cycle state of every node, and the one its parent alone
            forces it into, see
            :meth:`~cramph.node.StatechartNode.create_life_cycle_forced_by_parent`, both
            in node order.
        """
        next_life_cycles: list[sm.Scalar | None] = [None] * len(self._nodes)
        forced_life_cycles: list[sm.Scalar | None] = [None] * len(self._nodes)
        for node in sorted(self._nodes, key=lambda node: node.depth):
            life_cycles = sm.Vector(
                [
                    self._create_next_life_cycle(
                        node,
                        own_transitions_allowed=sm.logic_not(
                            self._own_transition_taken.variables[node.index]
                        ),
                    ),
                    node.create_life_cycle_forced_by_parent(),
                ]
            )
            if node.parent_node is not None:
                life_cycles = life_cycles.substitute(
                    [node.parent_node.life_cycle_variable],
                    [next_life_cycles[node.parent_node_index]],
                )
            next_life_cycles[node.index], forced_life_cycles[node.index] = life_cycles
        return next_life_cycles, forced_life_cycles

    @staticmethod
    def _create_next_life_cycle(
        node: StatechartNode, own_transitions_allowed: sm.Scalar
    ) -> sm.Scalar:
        """
        :param node: The node to build the life cycle state for.
        :param own_transitions_allowed: Whether `node` may still take a transition
            triggered by its own conditions.
        :return: The life cycle state `node` reaches in a pass.
        """
        return sm.if_eq_cases(
            a=node.life_cycle_variable,
            b_result_cases=node.create_lifecycle_transitions(
                own_transitions_allowed
            ).as_cases(),
            else_result=node.life_cycle_variable,
        )

    def _read_this_pass_observations(
        self,
        life_cycles: sm.Vector,
        observations: list[sm.Scalar],
        last_observations: list[sm.Scalar],
    ) -> sm.Vector:
        """
        :param life_cycles: Life cycle transitions whose conditions still read predicates.
        :param observations: The observation of every node after the pass.
        :param last_observations: The last observation of every node after the pass.
        :return: `life_cycles` with every predicate replaced by what it reads, and every
            observation read in the state this pass computes.
        """
        life_cycles = DerivedConditionVariable.substitute_in(life_cycles)
        return life_cycles.substitute(
            [node.observation_variable for node in self._nodes]
            + [node.last_observation for node in self._nodes],
            observations + last_observations,
        )

    def settle(self) -> list[LifeCycleChange]:
        """
        Runs passes until neither a life cycle state nor an observation changes, writing
        the result of every pass into the statechart.

        :return: Every life cycle change, in the order it happened.
        :raises TickDoesNotSettleError: If a pass returns the statechart
            to a state it already had in this tick, which it would then never
            leave, or if more passes than :attr:`_pass_budget` change it.
        """
        np.copyto(
            self._life_cycle_at_cycle_start.data,
            self.statechart.life_cycle_state.data,
        )
        self._own_transition_taken.data.fill(0)
        self._collect_tick_observations()
        changes: list[LifeCycleChange] = []
        visited_states = set()
        self._compiled_pass.evaluate()
        for passes_taken in range(self._pass_budget):
            if not self._latest_pass_changed_anything():
                return changes
            visited_states.add(self._current_state())
            if self._state_after_latest_pass() in visited_states:
                self._raise_does_not_settle(passes_taken + 1)
            changes.extend(self._life_cycle_changes_of_latest_pass())
            self._take_over_latest_pass()
            self._compiled_pass.evaluate()
        self._raise_does_not_settle(self._pass_budget)

    def _raise_does_not_settle(self, passes_taken: int) -> None:
        """
        :param passes_taken: The passes that changed the statechart so far.
        :raises TickDoesNotSettleError: Always, naming the nodes the latest pass
            changed.
        """
        raise TickDoesNotSettleError(
            pass_limit=self._pass_budget,
            passes_taken=passes_taken,
            unsettled_nodes=self._nodes_changed_by_latest_pass(),
        )

    def _current_state(self) -> bytes:
        """
        :return: Everything a pass reads that passes change, as one value that can be
            compared and remembered.
        """
        return b"".join(
            [
                self.statechart.life_cycle_state.data.tobytes(),
                self.statechart.observation_state.data.tobytes(),
                self.statechart.last_observation_state.data.tobytes(),
                self._own_transition_taken.data.tobytes(),
            ]
        )

    def _state_after_latest_pass(self) -> bytes:
        """
        :return: What :meth:`_current_state` becomes once the latest pass is taken over.
        """
        return b"".join(
            [
                self._next_life_cycle.tobytes(),
                self._next_observation.tobytes(),
                self._next_last_observation.tobytes(),
                self._next_own_transition_taken.tobytes(),
            ]
        )

    def _latest_pass_changed_anything(self) -> bool:
        """
        :return: Whether the latest pass changed a life cycle state, an observation or a
            last observation.
        """
        return not (
            np.array_equal(self._next_life_cycle, self.statechart.life_cycle_state.data)
            and np.array_equal(
                self._next_observation, self.statechart.observation_state.data
            )
            and np.array_equal(
                self._next_last_observation,
                self.statechart.last_observation_state.data,
            )
        )

    def _nodes_changed_by_latest_pass(self) -> list[StatechartNode]:
        """
        :return: The nodes whose life cycle state, observation or last observation the
            latest pass changed, in node order.
        """
        changed = (
            (self._next_life_cycle != self.statechart.life_cycle_state.data)
            | (self._next_observation != self.statechart.observation_state.data)
            | (
                self._next_last_observation
                != self.statechart.last_observation_state.data
            )
        )
        return [self._nodes[index] for index in np.flatnonzero(changed)]

    def _take_over_latest_pass(self) -> None:
        """
        Writes the result of the latest pass into the states the next pass reads.
        """
        np.copyto(self.statechart.life_cycle_state.data, self._next_life_cycle)
        np.copyto(self.statechart.observation_state.data, self._next_observation)
        np.copyto(
            self.statechart.last_observation_state.data,
            self._next_last_observation,
        )
        np.copyto(self._own_transition_taken.data, self._next_own_transition_taken)

    def _collect_tick_observations(self) -> None:
        """
        Calls :meth:`~cramph.node.StatechartNode.on_tick`
        once for every node running at the start of the tick and keeps what it
        returned for the passes.
        """
        self._has_tick_observation.data.fill(0)
        running_indices = np.flatnonzero(
            self._life_cycle_at_cycle_start.data == float(LifeCycleValues.RUNNING)
        )
        for index in running_indices:
            tick_observation = self._nodes[index].on_tick()
            if tick_observation is None:
                continue
            self._tick_observation.data[index] = tick_observation
            self._has_tick_observation.data[index] = 1

    def _life_cycle_changes_of_latest_pass(self) -> list[LifeCycleChange]:
        """
        :return: The life cycle changes of the latest pass, in node order.
        """
        life_cycle = self.statechart.life_cycle_state.data
        return [
            LifeCycleChange(
                node=self._nodes[index],
                previous_state=LifeCycleValues(int(life_cycle[index])),
                new_state=LifeCycleValues(int(self._next_life_cycle[index])),
            )
            for index in np.flatnonzero(self._next_life_cycle != life_cycle)
        ]
