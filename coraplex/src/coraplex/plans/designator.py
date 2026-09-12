from __future__ import annotations

import sys
from dataclasses import dataclass, field, Field
from typing import TYPE_CHECKING, Dict

from typing_extensions import Optional, List, Any, get_type_hints

from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
)
from krrood.ormatic.utils import classproperty
from coraplex.exceptions import ContextIsUnavailable
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan
    from coraplex.plans.plan_node import PlanNode
    from coraplex.datastructures.dataclasses import Context


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

    @classproperty
    def fields(cls) -> List[Field]:
        """
        The parameters of this designator: the fields a caller constructs it with.

        Fields that are not constructible -- the plan node it is mounted on, and
        whatever a designator works out for itself -- are not parameters and are left
        out, which is what
        :class:`~krrood.class_diagrams.attribute_introspector.DataclassOnlyIntrospector`
        already means by a public field.

        :return: The fields of this designator.
        """
        return [
            discovered.field for discovered in DataclassOnlyIntrospector().discover(cls)
        ]

    @property
    def designator_parameter(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.fields}

    @classmethod
    def get_type_hints(cls) -> Dict[str, Any]:
        """
        Returns the type hints of the __init__ method of this designator_description
        description.

        :return:
        """
        global_namespace = sys.modules[cls.__module__].__dict__
        return get_type_hints(cls.__init__, globalns=global_namespace)
