from __future__ import annotations

import sys
from dataclasses import dataclass, field, Field, fields
from typing import TYPE_CHECKING, Dict

from typing_extensions import Optional, List, Any, get_type_hints

from pycram.exceptions import ContextIsUnavailable
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from pycram.plans.plan import Plan
    from pycram.plans.plan_node import PlanNode
    from pycram.datastructures.dataclasses import Context


@dataclass
class Designator:
    """
    Abstract base class for designators.
    Designators are objects that can be executed and are managed by a plan node.
    """

    plan_node: Optional[PlanNode] = field(
        kw_only=True, default=None, repr=False, init=False
    )
    """
    The plan node that manages the designator.
    """

    @property
    def plan(self) -> Plan:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan_node.plan

    @property
    def robot(self) -> AbstractRobot:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan.robot

    @property
    def world(self) -> World:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan_node.plan.world

    @property
    def context(self) -> Context:
        return self.plan.context

    @classmethod
    @property
    def fields(cls) -> List[Field]:
        """
        The fields of this action, returns only the fields defined in the class and not inherit fields of parents

        :return: The fields of this action
        """
        self_fields = list(fields(cls))
        [self_fields.remove(parent_field) for parent_field in fields(Designator)]
        type_hints = cls.get_type_hints()
        for field in self_fields:
            field.type = type_hints[field.name]
        return self_fields

    @property
    def designator_parameter(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.fields}

    @classmethod
    def get_type_hints(cls) -> Dict[str, Any]:
        """
        Returns the type hints of the __init__ method of this designator_description description.

        :return:
        """
        global_namespace = sys.modules[cls.__module__].__dict__
        return get_type_hints(cls.__init__, globalns=global_namespace)
