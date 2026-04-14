"""PyCRAM bridge — the single boundary layer between llmr and pycram.

All pycram imports for the entire llmr package are funnelled through
the adapter module in this package. Nothing outside pycram_bridge imports
pycram directly.

Public surface:
  PycramContext       — structural protocol matching pycram Context
  PycramPlanNode      — structural protocol matching pycram PlanNode
  execute_single      — wraps pycram.plans.factories.execute_single
  discover_action_classes — scans pycram action package tree
  PycramIntrospector  — classifies action dataclass fields into FieldKind
  introspect          — convenience wrapper around PycramIntrospector
  FieldKind / FieldSpec / ActionSchema — introspection data types
"""
from llmr.pycram_bridge.adapter import (
    PycramContext,
    PycramPlanNode,
    discover_action_classes,
    execute_single,
)
from llmr.pycram_bridge.introspector import (
    ActionSchema,
    FieldKind,
    FieldSpec,
    PycramIntrospector,
    introspect,
)

__all__ = [
    "ActionSchema",
    "FieldKind",
    "FieldSpec",
    "PycramContext",
    "PycramPlanNode",
    "PycramIntrospector",
    "discover_action_classes",
    "execute_single",
    "introspect",
]
