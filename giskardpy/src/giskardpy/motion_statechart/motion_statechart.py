from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import rustworkx as rx
from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    List,
    MutableMapping,
    Optional,
    Self,
    Tuple,
    Type,
)

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.plotters.gantt_chart_plotter import (
    HistoryGanttChartPlotter,
)
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from krrood.symbolic_math.symbolic_math import VariableParameters
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    MotionStatechartJSONKey,
    TransitionKind,
    LifeCycleValues,
    LifeCyclePredicate,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    EmptyMotionStatechartError,
    ConditionScopeError,
    ControlCycleDoesNotSettleError,
    CyclicNodeDependencyError,
)
from giskardpy.motion_statechart.graph_node import (
    DeserializedNodeTracker,
    MotionStatechartNode,
    TransitionCondition,
    CompositeStatechartNode,
    EndMotion,
    CancelMotion,
    GenericMotionStatechartNode,
    ObservationVariable,
    LifeCycleVariable,
    DerivedConditionVariable,
    LastObservationVariable,
    DebugExpression,
)
from giskardpy.motion_statechart.graph_node import (
    SelfDecidingNode,
    SelfFailingNode,
    Task,
)
from giskardpy.motion_statechart.plotters.graphviz import MotionStatechartGraphviz
from giskardpy.qp.constraint_collection import ConstraintCollection
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityReferenceWriter,
)


@dataclass(repr=False, eq=False)
class State(MutableMapping[MotionStatechartNode, float], SubclassJSONSerializer):
    """
    Maps every node of a motion statechart to a scalar value, backed by a single
    contiguous array indexed by :attr:`~MotionStatechartNode.index`.
    """

    motion_statechart: MotionStatechart
    """
    The motion statechart whose nodes are the keys of this mapping.
    """

    default_value: ClassVar[float] = field(init=False)
    """
    The value that :meth:`grow` appends for a newly added node.
    """

    data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    """
    One entry per node, ordered by :attr:`~MotionStatechartNode.index`.
    """

    def grow(self) -> None:
        """
        Appends the default value to :attr:`data`, keeping it in sync with a newly added
        node.
        """
        self.data = np.append(self.data, self.default_value)

    def life_cycle_symbols(self) -> List[LifeCycleVariable]:
        """
        :return: The life cycle variable of every node, in node order.
        """
        return [node.life_cycle_variable for node in self.motion_statechart.nodes]

    def observation_symbols(self) -> List[ObservationVariable]:
        """
        :return: The observation variable of every node, in node order.
        """
        return [node.observation_variable for node in self.motion_statechart.nodes]

    def last_observation_symbols(self) -> List[LastObservationVariable]:
        """
        :return: The last observation variable of every node, in node order.
        """
        return [node.last_observation for node in self.motion_statechart.nodes]

    def __getitem__(self, node: MotionStatechartNode) -> float:
        """
        :param node: The node to look up.
        :return: The value stored for `node`, read from :attr:`data` at :attr:`~MotionStatechartNode.index`.
        """
        return float(self.data[node.index])

    def __setitem__(self, node: MotionStatechartNode, value: float) -> None:
        """
        Writes `value` into :attr:`data` at `node`'s
        :attr:`~MotionStatechartNode.index`.

        :param node: The node to write the value for.
        :param value: The value to store.
        """
        self.data[node.index] = value

    def __delitem__(self, node: MotionStatechartNode) -> None:
        """
        Removes the entry for `node` from :attr:`data`.

        .. warning:: This shifts the indices of all nodes after `node`, but does not update
            their :attr:`~MotionStatechartNode.index`, so the state and the nodes fall out of sync.

        :param node: The node whose entry to remove.
        """
        self.data = np.delete(self.data, node.index)

    def __iter__(self):
        return iter(self.motion_statechart.nodes)

    def __len__(self) -> int:
        return self.data.shape[0]

    def keys(self) -> List[MotionStatechartNode]:
        """
        :return: All nodes of the motion statechart, i.e. the keys of this mapping.
        """
        return self.motion_statechart.nodes

    def items(self) -> List[tuple[MotionStatechartNode, float]]:
        """
        :return: (node, value) pairs for every node of the motion statechart.
        """
        return [(node, self[node]) for node in self.motion_statechart.nodes]

    def values(self) -> List[float]:
        """
        :return: The value of every node, in node order.
        """
        return [self[node] for node in self.keys()]

    def __contains__(self, node: MotionStatechartNode) -> bool:
        return node in self.motion_statechart.nodes

    def __deepcopy__(self, memo) -> Self:
        """
        Create a deep copy of the state.

        :param memo: The memo dict used by :func:`copy.deepcopy` to track already-copied
            objects.
        :return: The deep copy.
        """
        return self.__class__(
            motion_statechart=self.motion_statechart,
            data=self.data.copy(),
        )

    def to_json(self, **kwargs) -> dict[str, Any]:
        """
        :return: The JSON representation of the base class, extended with the raw :attr:`data` array.
        """
        return {**super().to_json(**kwargs), "data": self.data.tolist()}

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        """
        Reconstruct a state from its JSON representation.

        :param data: The JSON dict, as produced by :meth:`to_json`.
        :param kwargs: Must contain the owning `motion_statechart`.
        :return: The deserialized state.
        """
        motion_statechart = kwargs["motion_statechart"]
        return cls(
            motion_statechart=motion_statechart,
            data=np.array(data["data"], dtype=np.float64),
        )

    def __str__(self) -> str:
        return str({str(symbol.name): value for symbol, value in self.items()})

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Self) -> bool:
        """
        :param other: The object to compare against.
        :return: True if `other` is a :class:`State` with the same :attr:`data`.
        .. note:: The owning :attr:`motion_statechart` is not compared.
        """
        if not isinstance(other, State):
            return NotImplemented
        return np.array_equal(self.data, other.data)


@dataclass(repr=False, eq=False)
class LifeCycleState(State):
    """
    The life cycle state of every node in a motion statechart, see
    :class:`MotionStatechart`.
    """

    default_value: ClassVar[float] = LifeCycleValues.NOT_STARTED
    """
    Every node starts out as not started.
    """

    def __getitem__(self, node: MotionStatechartNode) -> LifeCycleValues:
        """
        :param node: The node to look up.
        :return: The life cycle state of `node`, as a :class:`LifeCycleValues` member.
        """
        return LifeCycleValues(super().__getitem__(node))

    def __str__(self) -> str:
        return str(
            {
                str(symbol.name): LifeCycleValues(value).name
                for symbol, value in self.items()
            }
        )


@dataclass(repr=False, eq=False)
class ObservationState(State):
    """
    The observation state of every node in a motion statechart, see
    :class:`MotionStatechart`.
    """

    default_value: ClassVar[ObservationStateValues] = ObservationStateValues.UNKNOWN
    """
    A node that is not running is not observing.
    """

    def __getitem__(self, node: MotionStatechartNode) -> ObservationStateValues:
        """
        :param node: The node to look up.
        :return: What `node` observes, as an :class:`ObservationStateValues` member.
        """
        return ObservationStateValues(super().__getitem__(node))


@dataclass(repr=False, eq=False)
class LastObservationState(State):
    """
    The observation every node of a motion statechart took most recently.

    .. seealso:: :attr:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.last_observation`
    """

    default_value: ClassVar[ObservationStateValues] = ObservationStateValues.UNKNOWN
    """
    A node that has not started has not observed anything.
    """

    def __getitem__(self, node: MotionStatechartNode) -> ObservationStateValues:
        """
        :param node: The node to look up.
        :return: What `node` observed most recently, as an
            :class:`ObservationStateValues` member.
        """
        return ObservationStateValues(super().__getitem__(node))


# %% settling one control cycle


class PassInputKind(StrEnum):
    """
    A value per node that a pass through the motion statechart reads on top of the life
    cycle, observation and last observation states.
    """

    LIFE_CYCLE_AT_CYCLE_START = "life_cycle_at_cycle_start"
    """
    The life cycle state the node entered the control cycle with.
    """

    TICK_OBSERVATION = "tick_observation"
    """
    What :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick`
    returned for the node this control cycle.
    """

    HAS_TICK_OBSERVATION = "has_tick_observation"
    """
    Whether :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick`
    returned an observation for the node this control cycle.
    """

    OWN_TRANSITION_TAKEN = "own_transition_taken"
    """
    Whether the node already took a transition triggered by its own conditions this
    control cycle.
    """


@dataclass
class PassInput:
    """
    One value per node of a motion statechart that a pass reads, together with the
    variables standing for it in the compiled pass.
    """

    variables: List[sm.FloatVariable]
    """
    The variable of every node, in node order.
    """

    data: np.ndarray
    """
    The value of every node, in node order.
    """

    @classmethod
    def create(cls, kind: PassInputKind, nodes: List[MotionStatechartNode]) -> Self:
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

    node: MotionStatechartNode
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

    def run_callback(self, context: MotionStatechartContext) -> None:
        """
        Calls the callback of :attr:`node` that matches this change, e.g.
        :meth:`~MotionStatechartNode.on_start`. A change with no dedicated callback calls
        nothing.

        :param context: The context passed to the callback.
        """
        match (self.previous_state, self.new_state):
            case (_, LifeCycleValues.NOT_STARTED):
                self.node.on_reset(context=context)
            case (LifeCycleValues.NOT_STARTED, LifeCycleValues.RUNNING):
                self.node.on_start(context=context)
            case (LifeCycleValues.RUNNING, LifeCycleValues.PAUSED):
                self.node.on_pause(context=context)
            case (LifeCycleValues.PAUSED, LifeCycleValues.RUNNING):
                self.node.on_unpause(context=context)
            case (
                (LifeCycleValues.RUNNING | LifeCycleValues.PAUSED),
                _,
            ) if self.new_state.is_terminal:
                self.node.on_end(context=context)


@dataclass
class CompiledControlCycle:
    """
    Brings every node of a motion statechart to the state it reaches in one control
    cycle.

    One compiled pass updates every node at once, reading the states the previous pass
    left. Passes repeat until no state changes, so how deeply nodes are nested does not
    change when they react to each other. Within a pass:

    1. A node observes if it was running when the control cycle started and is still
       running or paused. Its observation expression reads the states of the previous
       pass. A node that was paused when the control cycle started and is still running
       or paused keeps its observation. A node that stopped running during this control
       cycle keeps the observation it stopped on until the next one. Every other node
       observes Unknown.
    2. A node that has neither ended nor started the control cycle ended takes over its
       observation as its last observation.
    3. Every node takes its next life cycle transition, reading the observations of this
       pass and the life cycle state its parent reaches in this pass. A node takes at
       most one transition triggered by its own conditions per control cycle;
       transitions its parent forces on it always happen.

    .. note:: Life cycle callbacks run afterwards, once per change and in the order the
        changes happened, so no Python code runs between passes.
    """

    motion_statechart: MotionStatechart
    """
    The motion statechart whose nodes are updated.
    """

    pass_limit: ClassVar[int] = 20
    """
    The most passes that may change the motion statechart within one control cycle, so
    that even at the limit a statechart of a few hundred nodes settles within a 50 Hz
    control cycle.
    """

    _nodes: List[MotionStatechartNode] = field(init=False)
    """
    Every node of :attr:`motion_statechart`, in node order.
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

    _next_observation: np.ndarray = field(init=False)
    """
    The observation of every node after the latest pass, a view on the pass output.
    """

    _next_last_observation: np.ndarray = field(init=False)
    """
    The last observation of every node after the latest pass, a view on the pass output.
    """

    _next_life_cycle: np.ndarray = field(init=False)
    """
    The life cycle state of every node after the latest pass, a view on the pass output.
    """

    _next_own_transition_taken: np.ndarray = field(init=False)
    """
    Whether every node took a transition triggered by its own conditions this control
    cycle, after the latest pass, a view on the pass output.
    """

    def compile(self, context: MotionStatechartContext) -> None:
        """
        Builds one pass through the motion statechart, compiles it and binds its inputs
        to the state arrays it reads.

        :param context: The context whose world and float variable data a pass reads.
        """
        self._nodes = self.motion_statechart.nodes
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
        self._compile_pass(context)

    def _compile_pass(self, context: MotionStatechartContext) -> None:
        """
        Compiles :meth:`_create_pass` and binds every input and output.

        :param context: The context whose world and float variable data a pass reads.
        """
        inputs = [
            (
                [node.life_cycle_variable for node in self._nodes],
                self.motion_statechart.life_cycle_state.data,
            ),
            (
                [node.observation_variable for node in self._nodes],
                self.motion_statechart.observation_state.data,
            ),
            (
                [node.last_observation for node in self._nodes],
                self.motion_statechart.last_observation_state.data,
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
                    LifeCyclePredicate.IS_TERMINATED.expression(
                        node.life_cycle_variable
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
        with_own_transitions, forced_only = self._create_next_life_cycles()
        life_cycles = self._read_this_pass_observations(
            sm.Vector(with_own_transitions + forced_only),
            observations=observations,
            last_observations=last_observations,
        )
        next_life_cycles = list(life_cycles)[: len(self._nodes)]
        forced_life_cycles = list(life_cycles)[len(self._nodes) :]
        own_transitions_taken = [
            sm.if_eq(
                next_life_cycle,
                forced_life_cycle,
                if_result=own_transition_taken,
                else_result=sm.Scalar.const_true(),
            )
            for next_life_cycle, forced_life_cycle, own_transition_taken in zip(
                next_life_cycles,
                forced_life_cycles,
                self._own_transition_taken.variables,
            )
        ]
        return sm.Vector(
            observations + last_observations + next_life_cycles + own_transitions_taken
        )

    def _create_observation(self, node: MotionStatechartNode, index: int) -> sm.Scalar:
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
        return sm.if_else(
            condition=sm.logic_or(
                LifeCyclePredicate.IS_RUNNING.expression(node.life_cycle_variable),
                LifeCyclePredicate.IS_PAUSED.expression(node.life_cycle_variable),
            ),
            if_result=sm.if_eq_cases(
                a=self._life_cycle_at_cycle_start.variables[index],
                b_result_cases=[
                    (int(LifeCycleValues.RUNNING), observed),
                    (int(LifeCycleValues.PAUSED), node.observation_variable),
                ],
                else_result=sm.Scalar.const_trinary_unknown(),
            ),
            else_result=sm.if_eq_cases(
                a=self._life_cycle_at_cycle_start.variables[index],
                b_result_cases=[
                    (int(LifeCycleValues.RUNNING), node.last_observation),
                    (int(LifeCycleValues.PAUSED), node.last_observation),
                ],
                else_result=sm.Scalar.const_trinary_unknown(),
            ),
        )

    def _create_next_life_cycles(
        self,
    ) -> Tuple[List[sm.Scalar], List[sm.Scalar]]:
        """
        Builds the life cycle state every node reaches in a pass, reading the state its
        parent reaches in the same pass, so a node never starts under a parent that stops
        running in that pass.

        :return: The life cycle state of every node, and the one it would reach without
            any transition triggered by its own conditions, both in node order.
        """
        with_own_transitions: List[Optional[sm.Scalar]] = [None] * len(self._nodes)
        forced_only: List[Optional[sm.Scalar]] = [None] * len(self._nodes)
        for node in sorted(self._nodes, key=lambda node: node.depth):
            with_own_transitions[node.index] = self._create_next_life_cycle(
                node,
                own_transitions_allowed=sm.logic_not(
                    self._own_transition_taken.variables[node.index]
                ),
            )
            forced_only[node.index] = self._create_next_life_cycle(
                node, own_transitions_allowed=sm.Scalar.const_false()
            )
            if node.parent_node is None:
                continue
            parent_variable = [node.parent_node.life_cycle_variable]
            parent_next_life_cycle = [with_own_transitions[node.parent_node_index]]
            with_own_transitions[node.index] = with_own_transitions[
                node.index
            ].substitute(parent_variable, parent_next_life_cycle)
            forced_only[node.index] = forced_only[node.index].substitute(
                parent_variable, parent_next_life_cycle
            )
        return with_own_transitions, forced_only

    @staticmethod
    def _create_next_life_cycle(
        node: MotionStatechartNode, own_transitions_allowed: sm.Scalar
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
        observations: List[sm.Scalar],
        last_observations: List[sm.Scalar],
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

    def settle(self, context: MotionStatechartContext) -> List[LifeCycleChange]:
        """
        Runs passes until neither a life cycle state nor an observation changes, writing
        the result of every pass into the motion statechart.

        :param context: The context passed to every
            :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick`.
        :return: Every life cycle change, in the order it happened.
        :raises ControlCycleDoesNotSettleError: If more than :attr:`pass_limit` passes
            change the motion statechart.
        """
        np.copyto(
            self._life_cycle_at_cycle_start.data,
            self.motion_statechart.life_cycle_state.data,
        )
        self._own_transition_taken.data.fill(0)
        self._collect_tick_observations(context)
        changes: List[LifeCycleChange] = []
        self._compiled_pass.evaluate()
        for _ in range(self.pass_limit):
            if not self._latest_pass_changed_anything():
                return changes
            changes.extend(self._life_cycle_changes_of_latest_pass())
            self._take_over_latest_pass()
            self._compiled_pass.evaluate()
        raise ControlCycleDoesNotSettleError(
            pass_limit=self.pass_limit,
            unsettled_nodes=self._nodes_changed_by_latest_pass(),
        )

    def _latest_pass_changed_anything(self) -> bool:
        """
        :return: Whether the latest pass changed a life cycle state, an observation or a
            last observation.
        """
        return not (
            np.array_equal(
                self._next_life_cycle, self.motion_statechart.life_cycle_state.data
            )
            and np.array_equal(
                self._next_observation, self.motion_statechart.observation_state.data
            )
            and np.array_equal(
                self._next_last_observation,
                self.motion_statechart.last_observation_state.data,
            )
        )

    def _nodes_changed_by_latest_pass(self) -> List[MotionStatechartNode]:
        """
        :return: The nodes whose life cycle state, observation or last observation the
            latest pass changed, in node order.
        """
        changed = (
            (self._next_life_cycle != self.motion_statechart.life_cycle_state.data)
            | (self._next_observation != self.motion_statechart.observation_state.data)
            | (
                self._next_last_observation
                != self.motion_statechart.last_observation_state.data
            )
        )
        return [self._nodes[index] for index in np.flatnonzero(changed)]

    def _take_over_latest_pass(self) -> None:
        """
        Writes the result of the latest pass into the states the next pass reads.
        """
        np.copyto(self.motion_statechart.life_cycle_state.data, self._next_life_cycle)
        np.copyto(self.motion_statechart.observation_state.data, self._next_observation)
        np.copyto(
            self.motion_statechart.last_observation_state.data,
            self._next_last_observation,
        )
        np.copyto(self._own_transition_taken.data, self._next_own_transition_taken)

    def _collect_tick_observations(self, context: MotionStatechartContext) -> None:
        """
        Calls :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick`
        once for every node running at the start of the control cycle and keeps what it
        returned for the passes.

        :param context: The context passed to every `on_tick`.
        """
        self._has_tick_observation.data.fill(0)
        running_indices = np.flatnonzero(
            self._life_cycle_at_cycle_start.data == float(LifeCycleValues.RUNNING)
        )
        for index in running_indices:
            tick_observation = self._nodes[index].on_tick(context=context)
            if tick_observation is None:
                continue
            self._tick_observation.data[index] = tick_observation
            self._has_tick_observation.data[index] = 1

    def _life_cycle_changes_of_latest_pass(self) -> List[LifeCycleChange]:
        """
        :return: The life cycle changes of the latest pass, in node order.
        """
        life_cycle = self.motion_statechart.life_cycle_state.data
        return [
            LifeCycleChange(
                node=self._nodes[index],
                previous_state=LifeCycleValues(int(life_cycle[index])),
                new_state=LifeCycleValues(int(self._next_life_cycle[index])),
            )
            for index in np.flatnonzero(self._next_life_cycle != life_cycle)
        ]


@dataclass(repr=False, eq=False)
class StateHistoryItem:
    """
    A snapshot of a :class:`MotionStatechart`'s life cycle and observation state at one
    control cycle.
    """

    control_cycle: int
    """
    The control cycle at which the snapshot was taken.
    """

    life_cycle_state: LifeCycleState
    """
    The life cycle state of every node at that control cycle.
    """

    observation_state: ObservationState
    """
    The observation state of every node at that control cycle.
    """

    def __post_init__(self):
        """
        Deep-copies the given states, so later mutation of the live states does not
        affect this snapshot.
        """
        self.life_cycle_state = deepcopy(self.life_cycle_state)
        self.observation_state = deepcopy(self.observation_state)

    def __eq__(self, other: StateHistoryItem) -> bool:
        """
        :param other: The item to compare against.
        :return: True if `other` has the same life cycle and observation state.
        .. note:: :attr:`control_cycle` is not compared.
        """
        has_life_cycle_changed = np.any(
            other.life_cycle_state.data != self.life_cycle_state.data
        )
        has_observation_changed = np.any(
            other.observation_state.data != self.observation_state.data
        )
        return not has_life_cycle_changed and not has_observation_changed

    def __repr__(self) -> str:
        """
        :return: Every node's name mapped to its observation state and life cycle state name.
        """
        merged = {
            node.name: f"{self.observation_state[node].name} | {life_cycle.name}"
            for node, life_cycle in self.life_cycle_state.items()
        }
        return str(merged)


@dataclass
class StateHistory:
    """
    The recorded sequence of :class:`StateHistoryItem` snapshots of a
    :class:`MotionStatechart`.
    """

    history: List[StateHistoryItem] = field(default_factory=list)
    """
    The snapshots in the order in which they were recorded, without consecutive
    duplicates.
    """

    def append(self, next_item: StateHistoryItem):
        """
        Appends `next_item`, unless it is equal to the last recorded item, in which case
        it is dropped to avoid storing consecutive duplicates.

        :param next_item: The snapshot to append.
        """
        if len(self.history) != 0:
            if next_item == self.history[-1]:
                return
        self.history.append(next_item)

    def get_life_cycle_history_of_node(
        self, node: MotionStatechartNode
    ) -> list[LifeCycleValues]:
        """
        :param node: The node to fetch the recorded life cycle state for.
        :return: The recorded life cycle state of `node` at every control cycle, in order.
        """
        return [history_item.life_cycle_state[node] for history_item in self.history]

    def get_observation_history_of_node(
        self, node: MotionStatechartNode
    ) -> list[ObservationStateValues]:
        """
        :param node: The node to fetch the recorded observation state for.
        :return: The recorded observation state of `node` at every control cycle, in order.
        """
        return [history_item.observation_state[node] for history_item in self.history]

    def __len__(self) -> int:
        return len(self.history)


@dataclass
class MotionStatechart(SubclassJSONSerializer):
    """
    Represents a motion statechart.
    A motion statechart is a directed graph of nodes and edges.
    Nodes have two states: observation state and life cycle state.
    Life cycle states indicate the current state in the life cycle of the node:
        - NOT_STARTED: the node has not started yet.
        - RUNNING: the node is running.
        - PAUSED: the node is paused.
        - SUCCEEDED: the node was ended while it was observing its goal as reached.
        - FAILED: the node was ended while it was not.
        - INTERRUPTED: the node was ended while it was not observing anything decisive,
                       which is no basis for a judgement.
    Out of these 6 states, nodes are only "active" if they are in the RUNNING state, and
    the last 3 are terminal: they are only left by a reset.
    Observation states indicate the current observation of the node:
        - TrinaryFalse: the thing the node is observing is not True.
        - TrinaryUnknown: the node cannot determine the truth value yet, or is not
                          observing at all.
        - TrinaryTrue: the thing the node is observing is True.
    Only a running node observes. A node that has not started or has reached a terminal
    state reports TrinaryUnknown, while a paused node keeps its last observation because
    it resumes and observes again.
    An observation is re-evaluated every tick and may change in both directions, whereas a
    verdict is latched. A condition is two-valued and may read either through a
    predicate: the observation state of a node through `node.observes_true` or
    `node.observes_false`, or its life cycle state through e.g. `node.is_failed`. What a
    node observes is gone once that node ends, so a condition that outlives the node it
    reads has to read something that outlasts it: `node.last_observed_true` keeps whether
    the observation the node took most recently was True, whatever its verdict, and a
    life cycle predicate keeps the verdict. Every tick settles the whole statechart before
    it returns, see :class:`CompiledControlCycle`, so a node waiting on another node's
    verdict starts on the tick that verdict is reached, however deeply either is nested.
    Nodes are connected with edges, or transitions.
    There are 6 types of transitions:
        - start condition: If True, the node transitions from NOT_STARTED to RUNNING.
        - pause condition: If True, the node transitions from RUNNING to PAUSED.
                           If False, the node transitions from PAUSED to RUNNING.
        - success condition: If True, the node ends from RUNNING or PAUSED as SUCCEEDED.
        - fail condition: If True, the node ends from RUNNING or PAUSED as FAILED.
        - interrupt condition: If True, the node ends from RUNNING or PAUSED as
                               INTERRUPTED.
        - reset condition: If True, the node transitions from any state to NOT_STARTED.
    The condition that ends a node decides its verdict; what the node observes at that
    moment has no say in it. A node ending takes its descendants down with it, and each
    of them is INTERRUPTED, however the node ended.
    If multiple conditions are met, the following order is used:
        1. its own reset condition, or its parent has not started
        2. its own success condition
        3. its own fail condition
        4. its own interrupt condition, or its parent has ended
        5. its own pause condition, or its parent is paused
        6. its own start condition, while its parent is running
    How to use this class:
        1. initialized with a world
        2. add nodes.
        3. set the transition conditions of nodes
        4. compile the motion statechart.
        5. call tick() to update the observation state and life cycle state.
            tick() will raise an exception if the cancel motion condition is met.
        6. call is_end_motion() to check if the motion is done.
    """

    rx_graph: rx.PyDiGraph[MotionStatechartNode] = field(
        default_factory=lambda: rx.PyDAG(multigraph=True), init=False, repr=False
    )
    """
    The underlying graph of the motion statechart.
    """

    observation_state: ObservationState = field(init=False)
    """
    Combined representation of the observation state of the motion statechart, to enable
    an efficient tick().
    """

    life_cycle_state: LifeCycleState = field(init=False)
    """
    Combined representation of the life cycle state of the motion statechart, to enable
    an efficient tick().
    """

    last_observation_state: LastObservationState = field(init=False)
    """
    Combined representation of the observation every node took most recently, to enable
    an efficient tick().
    """

    history: StateHistory = field(default_factory=StateHistory, init=False)
    """
    The history of how the state of the motion statechart changed over time.
    """

    _control_cycle: CompiledControlCycle = field(init=False, repr=False)
    """
    Updates every node once per control cycle, created by :meth:`compile`.
    """

    _nodes: List[MotionStatechartNode] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all nodes in index order, appended to in :meth:`add_node`.

    Reading this instead of rebuilding the list from `rx_graph` on every access is what
    keeps :meth:`tick` cheap.
    """

    _cancel_motion_nodes: List[CancelMotion] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all :class:`CancelMotion` nodes, checked every tick in
    :meth:`_raise_if_cancel_motion`.
    """

    _end_motion_nodes: List[EndMotion] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Cache of all :class:`EndMotion` nodes, checked every tick in :meth:`is_end_motion`.
    """

    def __post_init__(self):
        """
        Creates the (initially empty) life cycle, observation and last observation
        states for this motion statechart.
        """
        self.life_cycle_state = LifeCycleState(self)
        self.observation_state = ObservationState(self)
        self.last_observation_state = LastObservationState(self)

    def create_structure_copy(self) -> MotionStatechart:
        """
        Creates a copy of the motion statechart, where all nodes are
        MotionStatechartNodes or Goals.

        This is useful if only the structure of the motion statechart is needed, for
        example, for visualization.

        :return: The structural copy.
        """
        motion_statechart_copy = MotionStatechart()
        # copy nodes in order to make sure index is correct
        for node in self.nodes:
            match node:
                case CompositeStatechartNode():
                    node_copy = CompositeStatechartNode(name=node.name)
                case Task():
                    node_copy = Task(name=node.name)
                case EndMotion():
                    node_copy = EndMotion(name=node.name)
                case CancelMotion():
                    node_copy = CancelMotion(name=node.name, exception=node.exception)
                case _:
                    node_copy = MotionStatechartNode(name=node.name)
            motion_statechart_copy.add_node(node_copy)
        # link parent/child
        for node in self.get_nodes_by_type(CompositeStatechartNode):
            goal_copy: CompositeStatechartNode = (
                motion_statechart_copy.get_node_by_index(node.index)
            )
            for child_node in node.nodes:
                child_node_copy = motion_statechart_copy.get_node_by_index(
                    child_node.index
                )
                child_node_copy.parent_node_index = node.index
                goal_copy.nodes.append(child_node_copy)
        # copy conditions and plot specs
        for node in self.nodes:
            node_copy = motion_statechart_copy.get_node_by_index(node.index)
            node_copy.plot_specifications = deepcopy(node.plot_specifications)
            for transition_kind in TransitionKind:
                node_copy.set_condition(
                    transition_kind,
                    motion_statechart_copy._copy_condition(
                        node.get_condition(transition_kind)
                    ),
                )
        return motion_statechart_copy

    def _copy_condition(self, condition: sm.Scalar) -> sm.Scalar:
        """
        :param condition: A condition of the chart this chart is a structural copy of.
        :return: The same condition, reading the nodes of this chart with the same index.
        """
        variables: List[DerivedConditionVariable] = condition.free_variables()
        if not variables:
            return condition
        return sm.Scalar(condition).substitute(
            variables,
            [
                variable.for_node(
                    self.get_node_by_index(variable.motion_statechart_node.index)
                )
                for variable in variables
            ],
        )

    @property
    def nodes(self) -> List[MotionStatechartNode]:
        """
        :return: All nodes of the motion statechart.
        """
        return list(self._nodes)

    def collect_debug_expressions(self) -> List[DebugExpression]:
        """
        Gather the debug expressions registered by every node into a single flat list.

        :return: The debug expressions of every node.
        """
        return [
            debug_expression
            for node in self.nodes
            for debug_expression in node.debug_expressions
        ]

    @property
    def top_level_nodes(self) -> List[MotionStatechartNode]:
        """
        :return: All nodes that don't belong to a CompositeStatechartNode.
        """
        return [node for node in self.nodes if node.parent_node is None]

    @property
    def edges(self) -> List[TransitionCondition]:
        """
        The edges of the underlying graph.

        .. warning:: This may return duplicate edges if a transition uses multiple nodes.

        :return: The edges of the underlying graph.
        """
        return self.rx_graph.edges()

    @property
    def unique_edges(self) -> List[TransitionCondition]:
        """
        :return: The edges of the motion statechart, without duplicates.
        """
        return list(set(self.edges))

    def add_node(self, node: MotionStatechartNode):
        """
        Adds a node to the motion statechart and finalizes the initialization of the
        node.

        :param node: The node to add.
        """
        node.motion_statechart = self
        node.index = self.rx_graph.add_node(node)
        self.life_cycle_state.grow()
        self.observation_state.grow()
        self.last_observation_state.grow()
        self._nodes.append(node)
        if isinstance(node, CancelMotion):
            self._cancel_motion_nodes.append(node)
        if isinstance(node, EndMotion):
            self._end_motion_nodes.append(node)

    def add_nodes(self, nodes: List[MotionStatechartNode]):
        """
        Adds every node in `nodes` to the motion statechart, see :meth:`add_node`.

        :param nodes: The nodes to add.
        """
        for node in nodes:
            self.add_node(node)

    def get_node_by_index(self, index: int) -> MotionStatechartNode:
        """
        :param index: The :attr:`~MotionStatechartNode.index` of the node to look up.
        :return: The node with the given index.
        """
        return self.rx_graph.get_node_data(index)

    def _add_transitions(self):
        """
        Rebuilds the graph's edges from the current transition conditions of every node.
        """
        self._validate_condition_scopes()
        self.rx_graph.clear_edges()
        for node in self.nodes:
            for condition in node.conditions:
                self._create_edge_for_condition(node, condition)

    def _validate_condition_scopes(self):
        """
        Ensures that every condition only references its owning node or siblings of it.

        .. note:: Must run after goal expansion, because parent relationships are only known then.

        :raises ConditionScopeError: If a condition references a node from a different scope level.
        """
        for node in self.nodes:
            for condition in node.conditions:
                self._validate_condition_scope(node, condition)

    def _validate_condition_scope(
        self, owner: MotionStatechartNode, condition: TransitionCondition
    ):
        """
        Checks that `condition` only depends on `owner` itself, siblings of `owner` or
        direct children of `owner`.

        :param owner: The node that owns `condition`.
        :param condition: The condition to validate.
        :raises ConditionScopeError: If `condition` depends on a node from a different
            scope level.
        """
        for variable in condition.variables:
            dependency = variable.motion_statechart_node
            if dependency is owner:
                continue
            if dependency.parent_node_index == owner.parent_node_index:
                continue
            if dependency.parent_node_index == owner.index:
                continue
            raise ConditionScopeError(
                condition=condition,
                new_expression=condition.expression,
                dependency=dependency,
            )

    def _create_edge_for_condition(
        self, owner: MotionStatechartNode, condition: TransitionCondition
    ):
        """
        Adds an edge from `owner` to every node `condition` depends on.

        :param owner: The node the edges originate from.
        :param condition: The condition whose node dependencies become edge targets.
        """
        for parent_node in condition.node_dependencies:
            self.rx_graph.add_edge(owner.index, parent_node.index, condition)

    def _build_nodes(self, context: MotionStatechartContext):
        """
        Builds every node of the motion statechart and applies its resulting artifacts.

        :param context: The build context passed to every node's build.
        """
        built_node_indices: set[int] = set()
        for node in self.nodes:
            self._build_and_apply_artifacts(node, context, built_node_indices, [])

    def _build_and_apply_artifacts(
        self,
        node: MotionStatechartNode,
        context: MotionStatechartContext,
        built_node_indices: set[int],
        dependency_chain: List[MotionStatechartNode],
    ):
        """
        Builds `node`, recursively building the nodes it depends on and, if it is a
        :class:`CompositeStatechartNode`, its children first, then stores the resulting
        :class:`~giskardpy.motion_statechart.graph_node.NodeArtifacts` on the node.

        Already-built nodes (tracked via `built_node_indices`) are skipped.

        :param node: The node to build.
        :param context: The build context passed to :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.build`.
        :param built_node_indices: The indices of nodes already built, updated in place.
        :param dependency_chain: The nodes currently being built, used to detect cycles.
        """
        if node.index in built_node_indices:
            return
        self._check_no_dependency_cycle(node, dependency_chain)
        chain = dependency_chain + [node]
        for dependency in node.prerequisite_nodes:
            self._build_and_apply_artifacts(
                dependency, context, built_node_indices, chain
            )
        if isinstance(node, CompositeStatechartNode):
            for child_node in node.nodes:
                self._build_and_apply_artifacts(
                    child_node, context, built_node_indices, chain
                )
        built_node_indices.add(node.index)
        artifacts = node.build(context=context)
        node._constraint_collection = artifacts.constraints
        node._constraint_collection.link_to_motion_statechart_node(node)
        # if no observation is set, use the symbol for its observation variable to copy the state from last tick,
        # in case `on_tick` doesn't overwrite it.
        if artifacts.observation is None:
            node._observation_expression = node.observation_variable
        else:
            node._observation_expression = artifacts.observation
        node._error_signal = artifacts.error
        node._debug_expressions = artifacts.debug_expressions

    def _check_no_dependency_cycle(
        self,
        node: MotionStatechartNode,
        dependency_chain: List[MotionStatechartNode],
    ) -> None:
        """
        Raises if `node` already appears in the chain of nodes currently being expanded
        or built, which would otherwise recurse forever.

        :param node: The node to check.
        :param dependency_chain: The nodes currently being expanded or built.
        """
        if node not in dependency_chain:
            return
        cycle_start = dependency_chain.index(node)
        raise CyclicNodeDependencyError(
            node=node, cycle=dependency_chain[cycle_start:] + [node]
        )

    def compile(self, context: MotionStatechartContext):
        """
        Compiles all components of the motion statechart given the provided context.
        This method must be called before tick().

        :param context: The build context required to execute the compilation process.
        """
        self.sanity_check()
        self._expand_goals(context=context)
        self._succeed_self_deciding_nodes_observing_true()
        self._fail_self_failing_nodes_observing_false()
        self._build_nodes(context=context)
        self._add_transitions()
        self._control_cycle = CompiledControlCycle(motion_statechart=self)
        self._control_cycle.compile(context=context)
        self.history.append(
            next_item=StateHistoryItem(
                control_cycle=0,
                life_cycle_state=self.life_cycle_state,
                observation_state=self.observation_state,
            )
        )

    def _succeed_self_deciding_nodes_observing_true(self):
        """
        Gives every :class:`SelfDecidingNode` the success its contract promises, on top
        of whatever else already ends it.

        Runs once every goal has expanded, so no template can wire this away, and late
        enough that the conditions are still the ones a caller wrote while the templates
        were checking them.
        """
        for node in self.get_nodes_by_type(SelfDecidingNode):
            node.success_condition = sm.logic_or(
                node.success_condition, node.observes_true
            )

    def _fail_self_failing_nodes_observing_false(self):
        """
        Gives every :class:`SelfFailingNode` the failure its contract promises, on top
        of whatever else already fails it.

        Runs once every goal has expanded, so no template can wire this away.
        """
        for node in self.get_nodes_by_type(SelfFailingNode):
            node.fail_condition = sm.logic_or(node.fail_condition, node.observes_false)

    def _expand_goals(self, context: MotionStatechartContext):
        """
        Triggers the expansion of all goals in the motion statechart and add its
        children to the motion statechart.

        :param context: The build context passed to every goal's expansion.
        """
        expanded_goal_indices: set[int] = set()
        for goal in self.get_nodes_by_type(CompositeStatechartNode):
            self._expand_goal(goal, context, expanded_goal_indices, [])

    def _expand_goal(
        self,
        goal: CompositeStatechartNode,
        context: MotionStatechartContext,
        expanded_goal_indices: set[int],
        dependency_chain: List[MotionStatechartNode],
    ):
        """
        Expands the goals `goal` depends on, then `goal` itself, then recursively every
        child of `goal` that is itself a :class:`CompositeStatechartNode`.

        Already-expanded goals (tracked via `expanded_goal_indices`) are skipped, so a
        goal that several others depend on is still only expanded once.

        :param goal: The goal to expand.
        :param context: The build context passed to :meth:`~giskardpy.motion_statechart.graph_node.CompositeStatechartNode.expand`.
        :param expanded_goal_indices: The indices of goals already expanded, updated in place.
        :param dependency_chain: The goals currently being expanded, used to detect cycles.
        """
        if goal.index in expanded_goal_indices:
            return
        self._check_no_dependency_cycle(goal, dependency_chain)
        chain = dependency_chain + [goal]
        for dependency in goal.prerequisite_nodes:
            if isinstance(dependency, CompositeStatechartNode):
                self._expand_goal(dependency, context, expanded_goal_indices, chain)
        expanded_goal_indices.add(goal.index)
        goal.expand(context)
        for child_node in goal.nodes:
            if isinstance(child_node, CompositeStatechartNode):
                self._expand_goal(child_node, context, expanded_goal_indices, chain)

    def combine_constraint_collections_of_nodes(self) -> ConstraintCollection:
        """
        :return: The constraint collections of all nodes, merged into one, with each node's
            constraints prefixed by its :attr:`~MotionStatechartNode.unique_name`.
        """
        combined_constraint_collection = ConstraintCollection()
        for node in self.nodes:
            combined_constraint_collection.merge(
                name_prefix=node.unique_name, other=node._constraint_collection
            )
        return combined_constraint_collection

    def tick(self, context: MotionStatechartContext):
        """
        Executes a single tick of the motion statechart.

        Every node is brought to the state it reaches in this control cycle, see
        :class:`CompiledControlCycle`, then the life cycle callbacks of every change run.

        :param context: The context required to execute the tick.
        """
        for change in self._control_cycle.settle(context):
            change.run_callback(context)
        self._raise_if_cancel_motion()
        self.history.append(
            next_item=StateHistoryItem(
                control_cycle=len(self.history),
                life_cycle_state=self.life_cycle_state,
                observation_state=self.observation_state,
            )
        )

    def get_nodes_by_type(
        self, node_type: Type[GenericMotionStatechartNode]
    ) -> List[GenericMotionStatechartNode]:
        """
        :param node_type: The node type to filter for.
        :return: All nodes that are an instance of `node_type`.
        """
        return [node for node in self.nodes if isinstance(node, node_type)]

    def is_end_motion(self) -> bool:
        """
        :return: True if the motion is done, meaning at least one EndMotion is in observation state True, False otherwise.
        """
        return any(
            self.observation_state[node] == ObservationStateValues.TRUE
            for node in self._end_motion_nodes
        )

    def _raise_if_cancel_motion(self):
        """
        Raises the exception of the first :class:`CancelMotion` node whose observation
        state is True.
        """
        for node in self._cancel_motion_nodes:
            if self.observation_state[node] == ObservationStateValues.TRUE:
                raise node.exception

    def cleanup_nodes(self, context: MotionStatechartContext):
        """
        Calls :meth:`~MotionStatechartNode.cleanup` on every node.

        :param context: The context passed to every node's `cleanup`.
        """
        for node in self.nodes:
            node.cleanup(context)

    def draw(self, file_name: str):
        """
        Uses graphviz to draw the motion statechart and safe it at `file_name`.

        :param file_name: Where to save the resulting file.
        """
        MotionStatechartGraphviz(self).to_dot_graph_pdf(file_name=file_name)

    def plot_gantt_chart(
        self,
        path: str = "./ganttchart.pdf",
        context: MotionStatechartContext = None,
        second_length_in_cm: float = 2.0,
    ):
        """
        Renders a Gantt chart of :attr:`history` and saves it at `path`.

        :param path: Where to save the resulting PDF.
        :param context: If given (and it provides `dt`), the x-axis is scaled to seconds
            instead of control cycles.
        :param second_length_in_cm: Width in cm of one second on the x-axis.
        """
        HistoryGanttChartPlotter(
            self, second_width_in_cm=second_length_in_cm, context=context
        ).plot_gantt_chart(path)

    def to_json(self, **kwargs) -> dict[str, Any]:
        """
        World entities are written as references, because whoever reads a motion
        statechart resolves them against its own world, which has the same entities.

        :return: The JSON representation of this motion statechart, including all nodes
            and the transition conditions of every node the document holds.
        """
        kwargs = {**kwargs, **WorldEntityReferenceWriter().create_kwargs()}
        result = super().to_json(**kwargs)
        result[MotionStatechartJSONKey.NODES] = [
            to_json(node, **kwargs)
            for node in sorted(self.nodes, key=lambda n: n.index)
        ]
        result[MotionStatechartJSONKey.CONDITIONS] = [
            condition.to_json(**kwargs)
            for node in self._with_children_not_added(self.nodes)
            for condition in node.conditions
        ]
        return result

    def _with_children_not_added(
        self, nodes: List[MotionStatechartNode]
    ) -> List[MotionStatechartNode]:
        """
        :param nodes: The nodes to start from.
        :return: `nodes`, together with the children of every goal among them, recursively,
            that have not joined this motion statechart yet.
        """
        result = []
        for node in nodes:
            result.append(node)
            if not isinstance(node, CompositeStatechartNode):
                continue
            result.extend(
                self._with_children_not_added(
                    [
                        child
                        for child in node.nodes
                        if child._motion_statechart is not self
                    ]
                )
            )
        return result

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        """
        Reconstructs a motion statechart from its JSON representation, as produced by
        :meth:`to_json`: first all nodes, then the transition conditions of every node
        the document holds, then goal/child parent links. A goal that serializes its own nodes already holds
        them, so it is not handed them a second time.

        :param data: The JSON dict.
        :param kwargs: Forwarded to :func:`~krrood.adapters.json_serializer.from_json`
            for every node.
        :return: The deserialized motion statechart.
        """
        motion_statechart = cls()
        DeserializedNodeTracker.from_kwargs(kwargs)
        for json_data in data[MotionStatechartJSONKey.NODES]:
            node = from_json(json_data, **kwargs)
            motion_statechart.add_node(node)
        for json_data in data[MotionStatechartJSONKey.CONDITIONS]:
            transition = TransitionCondition.from_json(json_data, **kwargs)
            transition.owner._set_transition(transition)
        for node in motion_statechart.nodes:
            if node.parent_node_index is None:
                continue
            parent_node = motion_statechart.get_node_by_index(node.parent_node_index)
            if node not in parent_node.nodes:
                parent_node.nodes.append(node)
        return motion_statechart

    def sanity_check(self):
        """
        Executes a sanity check on the motion statechart to ensure that it is valid.
        """
        if len(self.nodes) == 0:
            raise EmptyMotionStatechartError()
