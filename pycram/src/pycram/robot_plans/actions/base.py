from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields, Field
from functools import cached_property

from typing_extensions import (
    Any,
    Callable,
    TypeVar,
    Dict,
    List,
    Union,
    Iterable,
    Generator,
)

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.entity import evaluate_condition
from krrood.entity_query_language.factories import variable, a, set_of
from ...datastructures.dataclasses import Context
from pycram.designator import DesignatorDescription
from pycram.failures import PlanFailure, ConditionNotSatisfied

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActionDescription(DesignatorDescription, ABC):
    _pre_perform_callbacks = []
    _post_perform_callbacks = []

    def perform(self) -> Any:
        """
        Full execution: pre-check, plan, post-check
        """
        logger.info(f"Performing action {self.__class__.__name__}")

        for pre_cb in self._pre_perform_callbacks:
            pre_cb(self)

        if self.plan.context.evaluate_conditions:
            self.evaluate_pre_condition()

        result = None
        try:
            result = self.execute()
        except PlanFailure as e:
            raise e
        finally:
            pass
            # for post_cb in self._post_perform_callbacks:
            #     post_cb(self)

        return result

    @abstractmethod
    def execute(self) -> Any:
        """
        Symbolic plan. Should only call motions or sub-actions.
        """
        pass

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @classmethod
    def pre_perform(cls, func) -> Callable:
        cls._pre_perform_callbacks.append(func)
        return func

    @classmethod
    def post_perform(cls, func) -> Callable:
        cls._post_perform_callbacks.append(func)
        return func

    @cached_property
    def bound_variables(self) -> Dict[T, Variable[T] | T]:
        return self._create_variables()

    # @cached_property
    # def unbound_variables(self) -> Dict[T, Variable[T] | T]:
    #     return self._create_variables(False)

    def _create_variables(self) -> Dict[str, Variable[T] | T]:
        """
        Creates krrood variables for all parameter of this action

        :return: A dict with action parameters as keys and variables as values.
        """
        return {
            f.name: variable(
                type(getattr(self, f.name)),
                ([getattr(self, f.name)]),
            )
            for f in self.fields
        }

    def evaluate_pre_condition(self) -> bool:
        condition = self.pre_condition(
            self.bound_variables,
            self.context,
            self.slots,
        )
        evaluation = evaluate_condition(condition)
        if evaluation:
            return True
        raise ConditionNotSatisfied(True, self.__class__, condition)

    def evaluate_post_condition(self) -> bool:
        condition = self.post_condition(
            self.bound_variables,
            self.context,
            self.slots,
        )
        evaluation = evaluate_condition(condition)
        if evaluation:
            return True
        raise ConditionNotSatisfied(False, self.__class__, condition)

    def find_possible_parameter(self) -> Generator[Dict[str, Any]]:
        """
        Queries the world using the pre_condition and yields possible parameters for this action which satisfy the
        precondition.

        :return: A dict that maps the name of the parameter to a possible value
        """
        unbound_condition = self.pre_condition(False)
        query = a(set_of(*self.unbound_variables.values()).where(unbound_condition))
        var_to_field = dict(zip(self.unbound_variables.values(), self.fields))
        for result in query.evaluate():
            bindings = result.data
            yield {var_to_field[k].name: v for k, v in bindings.items()}


ActionType = TypeVar("ActionType", bound=ActionDescription)
type DescriptionType[T] = Union[Iterable[T], T, ...]
