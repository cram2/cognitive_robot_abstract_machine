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

from cramera.knowledge.enums import EdgeKind, PlanNodeGroup
from cramera.knowledge.scene_bundle import SceneBundle

if TYPE_CHECKING:
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase

from cramera.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    GraphPanelPayload,
    LegendEntry,
    SubgraphAccumulator,
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

    empty_message: str = "No plan tree in this bundle — re-run cramera-onboard."
    """
    What the panel shows when the bundle recorded no plan at all.
    """

    def panel_options(self) -> Dict[str, Any]:
        """
        The plan legend and the live/status flags the plan tab is rendered with.
        """
        return {
            "breadcrumb": self.breadcrumb,
            "legend": [asdict(entry) for entry in PLAN_LEGEND],
            "layout": "hier",
            "live": "plan",
            "statusLegend": True,
            "empty": self.empty_message,
        }

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
        status = tree.get("status") or "CREATED"
        lines = ["a " + tree.get("kind", "PlanNode"), "status: " + status]
        if tree.get("arm"):
            lines.append("arm: " + tree["arm"])
        if tree.get("target"):
            lines.append("target: " + tree["target"])
        view.add(
            node_id,
            cls._shorten_action_label(tree.get("label", "?")),
            PlanNodeGroup.of_plan_node_kind(tree.get("kind")),
            lines,
            status=status,
        )
        if parent:
            view.add_edge(parent, node_id, EdgeKind.PROPERTY, "has step")
        for child in tree.get("children", []):
            cls._add_plan_node(view, node_ids, child, node_id)

    @classmethod
    def of_tab(cls, knowledge_base: EpisodeKnowledgeBase) -> PlanViewPayload:
        """
        The executed plan as a tree, one node per plan node the demo ran.

        The recorded statuses are thin on purpose: coraplex performs only the
        plan *root* (``Plan.perform`` → ``root.perform``), while
        ``ActionNode.notify`` merely expands its children into the merged motion
        statechart. So every inner node of a recorded tree reads ``CREATED``, and
        real per-step progress only shows up while the live bridge is attached
        (it derives it from the statechart life cycle).

        :param knowledge_base: Unused — the plan tree is read from the scene bundle.
        """
        scene = SceneBundle.of_scene(knowledge_base.scene_name).scene
        trees = scene.get("planTrees") or []
        view = SubgraphAccumulator()
        node_ids = itertools.count()
        for tree in trees:
            cls._add_plan_node(view, node_ids, tree, None)
        return cls(nodes=view.nodes, edges=view.edges, details=view.details)
