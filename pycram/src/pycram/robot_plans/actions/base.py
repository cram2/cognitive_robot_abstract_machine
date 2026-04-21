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
    Optional,
)

from pycram.exceptions import ContextIsUnavailable, ConditionNotSatisfied
from pycram.plans.failures import PlanFailure
from semantic_digital_twin.world import World

from pycram.plans.plan_node import PlanNode
from pycram.plans.designator import Designator
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    variable,
    a,
    set_of,
    evaluate_condition,
)
from ...datastructures.dataclasses import Context

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActionDescription(Designator):
    """
    Abstract base class for all actions.
    Actions are like builders for plans.
    An action has a set of parameters (its fields) from which it builds a symbolic plan and hence can be viewed as
    an easy abstraction of concrete low-level behavior that makes sense in certain contexts.
    """

    @property
    def world(self) -> Optional[World]:
        if self.plan is None:
            raise ContextIsUnavailable(self)
        return self.plan.world

    def perform(self) -> Any:
        """
        Perform the entire action including precondition and postcondition validation.
        """
        logger.info(f"Performing action {self.__class__.__name__}")

        if self.plan.context.evaluate_conditions:
            self.evaluate_pre_condition()

        result = None

        result = self.execute()

        return result

    @abstractmethod
    def execute(self) -> Any:
        """
        Create the symbolic plan for this action.
        This method should only use Motions or Actions and mount them under itself, such that the plan can manage the
        entire execution.
        """
        pass

    @staticmethod
    def pre_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        return True

    @cached_property
    def bound_variables(self) -> Dict[T, Variable[T] | T]:
        return self._create_variables()

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
            self.designator_parameter,
        )
        evaluation = evaluate_condition(condition)
        if evaluation:
            return True
        raise ConditionNotSatisfied(
            pre_condition=True, action=self.__class__, condition=condition
        )

    def evaluate_post_condition(self) -> bool:
        condition = self.post_condition(
            self.bound_variables,
            self.context,
            self.designator_parameter,
        )
        evaluation = evaluate_condition(condition)
        if evaluation:
            return True
        raise ConditionNotSatisfied(False, self.__class__, condition)

    def add_subplan(self, subplan_root: PlanNode) -> PlanNode:
        subplan_root = self.plan._migrate_nodes_from_plan(subplan_root.plan)
        self.plan.add_edge(self.plan_node, subplan_root)
        return subplan_root


ActionType = TypeVar("ActionType", bound=ActionDescription)
type DescriptionType[T] = Union[Iterable[T], T, ...]
