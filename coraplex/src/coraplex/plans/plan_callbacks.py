from dataclasses import dataclass

from coraplex.plans.plan import Plan
from coraplex.plans.plan_entity import PlanEntity
from coraplex.plans.plan_node import PlanNode


@dataclass
class PlanCallback(PlanEntity):

    def on_start(self, node: PlanNode): ...

    def on_end(self, node: PlanNode): ...
