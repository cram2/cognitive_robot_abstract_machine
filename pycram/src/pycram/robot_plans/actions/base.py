from __future__ import annotations

import abc
import logging
from dataclasses import dataclass

from typing_extensions import Any, Optional

from pycram.exceptions import ContextIsUnavailable
from pycram.plans.failures import PlanFailure
from semantic_digital_twin.world import World

from pycram.plans.plan_node import PlanNode
from pycram.plans.designator import Designator

logger = logging.getLogger(__name__)


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

        self.validate_precondition()

        result = None
        try:
            result = self.execute()
        except PlanFailure as e:
            raise e
        finally:
            self.validate_postcondition(result)

        return result

    @abc.abstractmethod
    def execute(self) -> Any:
        """
        Create the symbolic plan for this action.
        This method should only use Motions or Actions and mount them under itself, such that the plan can manage the
        entire execution.
        """
        pass

    def validate_precondition(self):
        """
        Symbolic/world state precondition validation.
        """
        pass

    def validate_postcondition(self, result: Optional[Any] = None):
        """
        Symbolic/world state postcondition validation.
        """
        pass

    def add_subplan(self, subplan_root: PlanNode) -> PlanNode:
        subplan_root = self.plan._migrate_nodes_from_plan(subplan_root.plan)
        self.plan.add_edge(self.plan_node, subplan_root)
        return subplan_root
