"""
The executed-plan-tree drill-down/tab view.

Named ``plan_tree`` rather than ``plan`` to keep it distinct from coraplex's own
``Plan``/``PlanNode`` types: this module renders the serialized tree of plan nodes
recorded in a scene bundle, not a coraplex ``Plan`` itself.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from cramera import paths
from cramera.knowledge.enums import EdgeKind, PlanNodeGroup
from cramera.knowledge.scene_bundle import SceneBundle
from cramera.recording_fields import SceneField

if TYPE_CHECKING:
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase

from cramera.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    GraphPanelPayload,
    LegendEntry,
    SubgraphAccumulator,
    TreePosition,
)

PLAN_LEGEND: Tuple[LegendEntry, ...] = tuple(
    LegendEntry(group, group.label) for group in PlanNodeGroup.legend()
)
"""
Legend rows of the plan view, one per :class:`PlanNodeGroup`.
"""


@dataclass(kw_only=True)
class PlanViewPayload(GraphPanelPayload):
    """
    The executed plan as a tree, one node per plan node the demo ran.
    """

    TAB: ClassVar[Optional[str]] = "plan"

    breadcrumb: str = "executed plan"
    """
    Breadcrumb label shown above the tree.
    """

    empty_message: str = "No plan tree was captured in this recording."
    """
    What the panel shows when the bundle recorded no plan at all.
    """

    is_live: bool = False
    """
    Whether the viewed scene follows the currently executing plan.
    """

    def panel_options(self) -> Dict[str, Any]:
        """
        The plan legend and the status flags the plan tab is rendered with.
        """
        options = {
            "breadcrumb": self.breadcrumb,
            "legend": [asdict(entry) for entry in PLAN_LEGEND],
            "layout": "hier",
            "statusLegend": False,
            "empty": self.empty_message,
        }
        if self.is_live:
            options["live"] = self.TAB
        return options

    @staticmethod
    def _shorten_action_label(label: str) -> str:
        """
        Drop the redundant ``Action`` suffix from a plan-node label.

        Only the suffix goes: a label that merely *contains* the word, such as
        ``ActionNode``, is left alone.

        :param label: The plan-node label to shorten.
        """
        return label.removesuffix("Action") or label

    @classmethod
    def count_nodes(cls, trees: List[Dict[str, Any]]) -> int:
        """
        How many nodes a bundle's recorded plan trees hold in total.

        :param trees: The serialized plan trees, as ``scene.json`` records them.
        """
        return sum(
            1 + cls.count_nodes(tree.get("children", []) or []) for tree in trees
        )

    @classmethod
    def _add_plan_node(
        cls,
        view: SubgraphAccumulator,
        node_ids: Iterator[int],
        tree: Dict[str, Any],
        parent: Optional[str],
    ) -> None:
        """
        Add one plan node, with a freshly assigned id, and recurse into its children.

        :param view: The subgraph the node and its edge are added to.
        :param node_ids: Counter handing out ids, shared across the whole tree walk.
        :param tree: The serialized plan node to add.
        :param parent: Id of the node's parent entry, or None for the root.
        """
        node_id = "plan_tree_node_%d" % next(node_ids)
        kind = tree.get("kind", "PlanNode")
        lines = ["a " + kind]
        if tree.get("arm"):
            lines.append("arm: " + tree["arm"])
        if tree.get("target"):
            lines.append("target: " + tree["target"])
        view.add(
            node_id,
            cls._shorten_action_label(tree.get("label", "?")),
            PlanNodeGroup.of_plan_node_kind(kind),
            lines,
            tree_position=TreePosition(kind=kind, parent=parent),
        )
        if parent:
            view.add_edge(parent, node_id, EdgeKind.PROPERTY, "has step")
        for child in tree.get("children", []):
            cls._add_plan_node(view, node_ids, child, node_id)

    @classmethod
    def of_tab(cls, knowledge_base: EpisodeKnowledgeBase) -> PlanViewPayload:
        """
        The executed plan as a tree, one node per plan node the demo ran.

        Without a status: which node was running when is only meaningful while a demo is
        actually performing the plan, and the bridge streams that. A recording is
        scrubbed back and forth, so a status recorded per node would be a claim about one
        moment shown at every other one; the replayed tree therefore shows structure
        only.

        :param knowledge_base: Identifies the recording or live scene being inspected.
        """
        scene = SceneBundle.of_scene(knowledge_base.scene_name).scene
        trees = scene.get(SceneField.PLAN_TREES) or []
        view = SubgraphAccumulator()
        node_ids = itertools.count()
        for tree in trees:
            cls._add_plan_node(view, node_ids, tree, None)
        return cls(
            nodes=view.nodes,
            edges=view.edges,
            details=view.details,
            is_live=(knowledge_base.scene_name or SceneBundle.active_name())
            == paths.LIVE_SCENE_NAME,
        )
