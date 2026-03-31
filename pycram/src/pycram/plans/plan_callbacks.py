from dataclasses import dataclass

from pycram.plans.plan import Plan
from pycram.plans.plan_entity import PlanEntity
from pycram.plans.plan_node import PlanNode


@dataclass
class PlanCallback(PlanEntity):

    def on_start(self, node: PlanNode): ...

    def on_end(self, node: PlanNode): ...
