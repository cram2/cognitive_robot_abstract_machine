"""
Action schemas - actions whose parameters an EQL precondition constrains and whose
effects change a state - and the ground actions they stand for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from typing_extensions import Self

from krrood.entity_query_language.class_diagram_lookup import get_field_type_endpoint
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.factories import (
    ConditionType,
    an,
    inference,
    literal,
    variable,
)
from krrood.entity_query_language.query.query import Entity
from krrood.planning.effect import Effect

# %% action schemas


@dataclass(kw_only=True)
class ActionSchema(ABC):
    """
    An action whose dataclass fields are its parameters. An instance whose fields hold
    objects is a *ground action*; :meth:`~krrood.planning.state.State.apply_action`
    applies one.

    :meth:`precondition` and :meth:`effects` read the parameters as EQL expressions: a
    variable of each field's type in :meth:`applicable_actions`, or the literals
    :meth:`to_symbolic` gives a ground action.

    >>> from dataclasses import dataclass
    >>> from krrood.entity_query_language.factories import a
    >>> from krrood.planning.effect import Assignment
    >>> from krrood.planning.state import State
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
    >>> (action,) = SwitchOn.applicable_actions().tolist(state=State())
    >>> a(Lamp).on.tolist(state=State().apply_action(action=action))
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
    def applicable_actions(cls) -> Entity[Self]:
        """
        :return: The query whose results, evaluated in a state, are a ground action for
            every binding of the parameters that satisfies the precondition there, each
            parameter ranging over the instances of its field's type that exist there.
        """
        variables = {
            parameter.name: variable(get_field_type_endpoint(cls, parameter.name))
            for parameter in fields(cls)
        }
        return an(inference(cls)(**variables)).where(cls(**variables).precondition())

    def to_symbolic(self) -> Self:
        """
        :return: A copy of this ground action whose fields hold a literal of the object
            each field of this action holds.
        """
        values = vars(self)
        parameter_literals: dict[str, Literal] = {
            parameter.name: literal(values[parameter.name])
            for parameter in fields(self)
        }
        return type(self)(**parameter_literals)
