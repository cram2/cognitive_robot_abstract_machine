from __future__ import annotations

import logging
from abc import abstractmethod
from copy import copy
from dataclasses import dataclass
from functools import cached_property

from typing_extensions import (
    Any,
    TypeVar,
    Dict,
    Union,
    Iterable,
    Optional,
)

from coraplex.datastructures.dataclasses import Context
from coraplex.exceptions import ContextIsUnavailable
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.designator import Designator
from coraplex.plans.plan_node import PlanNode, ActionNode
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import (
    variable,
)
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActionDescription(Designator):
    """
    Abstract base class for all actions.

    Actions are like builders for plans. An action has a set of parameters (its fields)
    from which it builds a symbolic plan and hence can be viewed as an easy abstraction
    of concrete low-level behavior that makes sense in certain contexts.
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

    @property
    def action_plan(self) -> PlanNode:

        sub_plan_root = self._action_plan
        action_node = ActionNode(designator=copy(self))

        pre_condition_node = ConditionNode(
            condition=self.pre_condition(
                self.bound_variables,
                self.context,
                self.designator_parameter,
            ),
            pre_condition=True,
            action_node=action_node,
        )

        sub_plan_root.plan.add_edge(action_node, pre_condition_node)

        sub_plan_root.plan.add_edge(action_node, sub_plan_root)

        post_condition_node = ConditionNode(
            condition=self.post_condition(
                self.bound_variables,
                self.context,
                self.designator_parameter,
            ),
            pre_condition=False,
            action_node=action_node,
        )

        sub_plan_root.plan.add_edge(action_node, post_condition_node)

        return action_node

    @property
    @abstractmethod
    def _action_plan(self) -> PlanNode:
        """
        Creates the whole plan for this action.

        :return: The root node of the plan of this action
        """
        ...

    def expand(self) -> PlanNode:

        return self.add_subplan(self.action_plan)

    def execute(self) -> Any:
        """
        Create the symbolic plan for this action.

        This method should only use Motions or Actions and mount them under itself, such
        that the plan can manage the entire execution.
        """
        self.add_subplan(self.action_plan)

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
        Creates krrood variables for all parameter of this action.

        :return: A dict with action parameters as keys and variables as values.
        """
        return {
            f.name: variable(
                type(getattr(self, f.name)),
                ([getattr(self, f.name)]),
            )
            for f in self.fields
        }

    def add_subplan(self, subplan_root: PlanNode) -> PlanNode:
        subplan_root = self.plan._migrate_nodes_from_plan(subplan_root.plan)
        self.plan.add_edge(self.plan_node, subplan_root)
        self.plan.simplify()
        return subplan_root


ActionType = TypeVar("ActionType", bound=ActionDescription)
type DescriptionType[T] = Union[Iterable[T], T, ...]
