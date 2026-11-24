"""
States that assign values to ground terms, and add and remove objects, without changing
live objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import TYPE_CHECKING, Any, Iterable, Iterator

from krrood.class_diagrams.collection_kind import CollectionKind
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.evaluation import ConditionEvaluation
from krrood.entity_query_language.factories import (
    ConditionType,
    an,
    entity,
    literal,
)
from krrood.entity_query_language.ground_term import (
    FunctionKey,
    GroundTerm,
    TermReader,
    TermValue,
)
from krrood.entity_query_language.utils import T
from krrood.planning.effect import (
    Assignment,
    CollectionInsertion,
    Effect,
    StateChange,
)
from krrood.planning.exceptions import ActionIsNotApplicable

if TYPE_CHECKING:
    from krrood.planning.action_schema import ActionSchema

# %% states


@dataclass(kw_only=True, frozen=True)
class State(TermReader):
    """
    A valuation of *fluents*: ground terms whose value a state can change, each with its
    :class:`~krrood.entity_query_language.ground_term.FunctionKey` as fluent symbol. A
    derived fluent, such as a predicate, follows the state only where its body is EQL;
    once assigned, it keeps the assigned value and no longer follows its inputs.

    >>> from dataclasses import dataclass
    >>> from krrood.entity_query_language.factories import a, literal
    >>> @dataclass
    ... class Box:
    ...     color: str
    >>> box = Box("red")
    >>> state = State().add_facts(literal(box).color == "blue")
    >>> a(Box).from_(domain=[box]).color.tolist(state=state)
    ['blue']
    >>> box.color
    'red'
    """

    assignments: dict[GroundTerm, Any] = field(default_factory=dict)
    """
    The value of every ground term this state changes.

    Never mutated; a changed state is a new :class:`State`.
    """

    added_objects: dict[int, Any] = field(default_factory=dict)
    """
    The objects this state adds to the live symbol graph's instances, keyed by their
    :func:`id`. A variable without a domain ranges over them too.

    Never mutated; a changed state is a new :class:`State`.
    """

    insertions: dict[GroundTerm, CollectionInsertion] = field(default_factory=dict)
    """
    The elements this state inserts into the collection each ground term holds, on top
    of its assigned or live value.

    Never mutated; a changed state is a new :class:`State`.
    """

    removed_objects: dict[int, Any] = field(default_factory=dict)
    """
    The objects this state removes from the live symbol graph's instances, keyed by
    their :func:`id`. Every variable domain, and every ground term whose value is a
    collection of elements, reads without them; a literal still names its object.

    Never mutated; a changed state is a new :class:`State`.
    """

    def __hash__(self) -> int:
        return hash(
            (
                frozenset(self.assignments),
                frozenset(self.insertions),
                frozenset(self.added_objects),
                frozenset(self.removed_objects),
            )
        )

    @cached_property
    def assigned_functions(self) -> frozenset[FunctionKey]:
        """
        :return: The :attr:`~krrood.entity_query_language.ground_term.GroundTerm.function_key`
            of every ground term this state assigns or inserts into.
        """
        return frozenset(
            term.function_key for term in (*self.assignments, *self.insertions)
        )

    def add_facts(self, *facts: SymbolicExpression) -> State:
        """
        :param facts: Conditions that each fix the value of ground terms, as
            :meth:`~krrood.planning.fact.Fact.from_condition` reads them. A fact whose
            expression has variables fixes the ground term for every binding of them.
        :return: A new state that assigns the values *facts* state on top of this
            state's own assignments, a later value winning where two assign a ground term.
            Each fact reads its bindings and values in the state the facts before it
            build.
        """
        state = self
        for fact in facts:
            state = state.apply_effects(Assignment(fact=fact))
        return state

    def apply_effects(self, *effects: Effect) -> State:
        """
        :param effects: The effects to apply.
        :return: A new state that makes the change of every effect, each read in this
            state, a later change winning where two assign a ground term.
        """
        changes = [
            effect.change_of_binding(binding=binding)
            for effect in effects
            for binding in effect.get_change_to_state().evaluate(state=self)
        ]
        state = self
        for change in changes:
            state = state._apply_change_(change=change)
        return state

    def apply_action(self, *, action: ActionSchema) -> State:
        """
        :param action: A ground action.
        :return: A new state that makes the change of every effect of *action*, as
            :meth:`apply_effects` does.
        :raises ActionIsNotApplicable: If this state does not satisfy the precondition
            of *action*.
        """
        symbolic_action = action.to_symbolic()
        precondition = ConditionEvaluation.from_condition(
            condition=symbolic_action.precondition(), state=self
        )
        if not precondition.is_satisfied:
            raise ActionIsNotApplicable(
                action=action, failed_conditions=precondition.false_statements
            )
        return self.apply_effects(*symbolic_action.effects())

    def satisfies(self, *, condition: ConditionType) -> bool:
        """
        :param condition: A condition, which may name variables.
        :return: Whether *condition* holds in this state for some binding of them.
        """
        query = an(entity(literal(True)).where(condition))
        return next(iter(query.evaluate(state=self)), False)

    def _apply_change_(self, *, change: StateChange) -> State:
        """
        :param change: What an effect changes.
        :return: A new state that makes *change* on top of this state; an assignment
            drops what this state inserted into its ground term.
        """
        insertions = {
            term: insertion
            for term, insertion in self.insertions.items()
            if term not in change.assignments
        }
        for term, insertion in change.insertions.items():
            earlier = insertions.get(term)
            insertions[term] = (
                insertion if earlier is None else earlier.followed_by(later=insertion)
            )
        return (
            State(
                assignments=self.assignments | change.assignments,
                insertions=insertions,
                added_objects=self.added_objects,
                removed_objects=self.removed_objects,
            )
            .with_objects_added(*change.added_objects)
            .with_objects_removed(*change.removed_objects)
        )

    def with_objects_added(self, *objects: Any) -> State:
        """
        :param objects: The objects to add.
        :return: A new state in which *objects* exist, whether this state removed them
            or never had them.
        """
        added_ids = {id(added) for added in objects}
        return State(
            assignments=self.assignments,
            insertions=self.insertions,
            added_objects=self.added_objects | {id(added): added for added in objects},
            removed_objects={
                identifier: removed
                for identifier, removed in self.removed_objects.items()
                if identifier not in added_ids
            },
        )

    def with_objects_removed(self, *objects: Any) -> State:
        """
        :param objects: The objects to remove.
        :return: A new state in which *objects* do not exist, whether they are live or
            this state added them.
        """
        removed_ids = {id(removed) for removed in objects}
        return State(
            assignments=self.assignments,
            insertions=self.insertions,
            added_objects={
                identifier: added
                for identifier, added in self.added_objects.items()
                if identifier not in removed_ids
            },
            removed_objects=self.removed_objects
            | {id(removed): removed for removed in objects},
        )

    def value_of_term(self, *, term: GroundTerm) -> TermValue | None:
        if term.function_key not in self.assigned_functions:
            return self._without_removed_objects_(term_value=term.live_term_value())
        if term in self.assignments:
            term_value = TermValue(value=self.assignments[term])
        else:
            term_value = term.live_term_value()
        insertion = self.insertions.get(term)
        if term_value is not None and insertion is not None:
            term_value = TermValue(
                value=insertion.insert_into(collection=term_value.value)
            )
        return self._without_removed_objects_(term_value=term_value)

    def _without_removed_objects_(
        self, *, term_value: TermValue | None
    ) -> TermValue | None:
        """
        :param term_value: The value of a ground term, or ``None`` where it has none.
        :return: *term_value* as :meth:`without_removed_objects` reads its value.
        """
        if term_value is None:
            return None
        return TermValue(value=self.without_removed_objects(value=term_value.value))

    def without_removed_objects(self, *, value: Any) -> Any:
        """
        :param value: The value of a ground term.
        :return: *value* without the objects this state removes when it is a collection
            of elements that holds any of them, as a new collection of the same type;
            otherwise *value* itself.
        """
        if not isinstance(value, CollectionKind.ELEMENT_SEQUENCE.value):
            return value
        if not any(id(element) in self.removed_objects for element in value):
            return value
        return type(value)(
            element for element in value if id(element) not in self.removed_objects
        )

    def existing_values(self, *, values: Iterable[T]) -> Iterator[T]:
        return (value for value in values if id(value) not in self.removed_objects)

    def added_instances_of(self, *, type_: type[T]) -> Iterator[T]:
        return (
            added for added in self.added_objects.values() if isinstance(added, type_)
        )
