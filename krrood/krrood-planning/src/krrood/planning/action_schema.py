"""
Action schemas - actions whose parameters an EQL precondition constrains and whose
effects change a state - and the ground actions they stand for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from typing_extensions import Any, Mapping, Self

from krrood.entity_query_language.class_diagram_lookup import get_field_type_endpoint
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import (
    ConditionType,
    literal,
    set_of,
    variable,
)
from krrood.planning.effect import Effect
from krrood.planning.exceptions import ActionIsNotApplicable
from krrood.planning.state import State

# %% action schemas


@dataclass(kw_only=True)
class ActionSchema(ABC):
    """
    An action whose dataclass fields are its parameters. An instance whose fields hold
    objects is a *ground action*.

    :meth:`precondition` and :meth:`effects` read the parameters as EQL expressions: they
    are called on an instance whose fields hold a literal of each object of a ground
    action, or a variable of each field's type while the schema is grounded.

    >>> from dataclasses import dataclass
    >>> from krrood.entity_query_language.factories import a
    >>> from krrood.planning.effect import Assignment
    >>> from krrood.symbol_graph.symbol_graph import Symbol
    >>> @dataclass(kw_only=True, eq=False)
    ... class Lamp(Symbol):
    ...     on: bool
    >>> @dataclass(kw_only=True)
    ... class SwitchOn(ActionSchema):
    ...     lamp: Lamp
    ...     def precondition(self):
    ...         return self.lamp.on == False
    ...     def effects(self):
    ...         return [Assignment(fact=self.lamp.on == True)]
    >>> lamp = Lamp(on=False)
    >>> (action,) = SwitchOn.applicable_actions(state=State())
    >>> a(Lamp).on.tolist(state=action.apply(state=State()))
    [True]
    """

    @abstractmethod
    def precondition(self) -> ConditionType:
        """
        :return: The condition on the parameters a state satisfies where this action is
            applicable.
        """

    @abstractmethod
    def effects(self) -> list[Effect]:
        """
        :return: What applying this action changes, each effect read in the state the
            action is applied in.
        """

    @classmethod
    def applicable_actions(cls, *, state: State) -> list[Self]:
        """
        :param state: The state the actions are applied in.
        :return: A ground action for every binding of the parameters that satisfies the
            precondition in *state*, each parameter ranging over the instances of its
            field's type that exist in *state*.
        """
        parameters = {
            parameter.name: variable(get_field_type_endpoint(cls, parameter.name))
            for parameter in fields(cls)
        }
        return [
            cls(**{name: binding[value] for name, value in parameters.items()})
            for binding in cls._bindings_satisfying_precondition_(
                parameters=parameters, state=state
            )
        ]

    def is_applicable(self, *, state: State) -> bool:
        """
        :param state: The state the action would be applied in.
        :return: Whether *state* satisfies the precondition of this ground action.
        """
        return bool(
            self._bindings_satisfying_precondition_(
                parameters=self._parameter_literals_(), state=state
            )
        )

    def apply(self, *, state: State) -> State:
        """
        :param state: The state the action is applied in.
        :return: A new state that makes the change of every effect of this ground
            action, each computed in *state*, a later effect winning where two assign a
            ground term.
        :raises ActionIsNotApplicable: If *state* does not satisfy the precondition.
        """
        if not self.is_applicable(state=state):
            raise ActionIsNotApplicable(action=self)
        changes = [
            effect.change(state=state)
            for effect in type(self)(**self._parameter_literals_()).effects()
        ]
        for change in changes:
            state = state.apply_change(change=change)
        return state

    @classmethod
    def _bindings_satisfying_precondition_(
        cls, *, parameters: dict[str, SymbolicExpression], state: State
    ) -> list[Mapping[SymbolicExpression, Any]]:
        """
        :param parameters: The expression each parameter is read as.
        :param state: The state the precondition is read in.
        :return: The values of *parameters* for every binding that satisfies the
            precondition in *state*.
        """
        precondition = cls(**parameters).precondition()
        return set_of(*parameters.values()).where(precondition).tolist(state=state)

    def _parameter_literals_(self) -> dict[str, SymbolicExpression]:
        """
        :return: A literal of the object each parameter of this ground action holds.
        """
        values = vars(self)
        return {
            parameter.name: literal(values[parameter.name])
            for parameter in fields(self)
        }
