"""
The recorded demo scene as an EQL (Entity Query Language) knowledge base.

EQL is krrood's pythonic relational query language. This package models the recorded
coraplex/giskardpy episode — bench objects, robot parts, action episodes, per-joint
motion — as plain dataclasses and exposes:

fresh_namespace()  -> dict for evaluating one EQL query (fresh variables)
run_query(code)    -> execute an EQL query string, return JSON-able result
graph_payload()    -> nodes/edges/details/presets for the UI knowledge graph
view_payload(name) -> one of the graph-panel tabs (knowledge / kinematics / plan /
chart)

krrood is imported lazily: without it the static viewer still works, only the EQL panel
is unavailable. Scene bundles are read from paths.scenes_dir().
"""

from __future__ import annotations

from cram_viz.knowledge.architecture_entities import (
    Package as Package,
    PythonClass as PythonClass,
)
from cram_viz.knowledge.eql_session import run_query as run_query
from cram_viz.knowledge.graph_payload import graph_payload as graph_payload
from cram_viz.knowledge.knowledge_base import (
    get_knowledge_base as get_knowledge_base,
    reset_knowledge_base as reset_knowledge_base,
)
from cram_viz.knowledge.presets import get_presets as get_presets
from cram_viz.knowledge.scene_bundle import load_scene as load_scene
from cram_viz.knowledge.views import (
    expand_node as expand_node,
    view_payload as view_payload,
)
from cram_viz.knowledge.views.architecture import ArchitectureViews

#: re-exported for the test suite and any external caller inspecting the truncation
#: limits; the module itself now scopes these onto ArchitectureViews, not a global.
CLASS_CAP = ArchitectureViews.CLASS_CAP
SUBCLASS_CAP = ArchitectureViews.SUBCLASS_CAP
from cram_viz.knowledge.views.plan_tree import (
    shorten_action_label as shorten_action_label,
)
