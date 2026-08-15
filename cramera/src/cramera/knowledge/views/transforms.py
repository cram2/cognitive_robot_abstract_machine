"""
The live-only transform-graph tab.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, ClassVar, Dict, Optional, TYPE_CHECKING

from cramera.knowledge.subgraph import GraphPanelPayload

if TYPE_CHECKING:
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase


@dataclass(kw_only=True)
class TransformViewPayload(GraphPanelPayload):
    """
    The (live-only) transform tab.

    Which frame currently hangs from which, and how long ago each of those transforms
    last changed, only exists while a demo runs: a recording holds the poses it went
    through, not the moment each connection was last written. The UI fills this view
    from the bridge's ``/transforms`` while attached.
    """

    TAB: ClassVar[Optional[str]] = "transforms"

    breadcrumb: str = "transform graph"
    """
    Breadcrumb label shown above the (initially empty) graph.
    """

    empty_message: str = (
        "The transform graph is read off the executing world. "
        "Start the demo with cramera-live and press ◉ Live — "
        "every connection appears here, ringed by how recently it moved."
    )
    """
    What the panel shows until a live bridge is attached.
    """

    @classmethod
    def of_tab(cls, knowledge_base: EpisodeKnowledgeBase) -> TransformViewPayload:
        """
        The transform tab, which only ever has content while a demo is running.

        :param knowledge_base: Unused — the bridge fills this view at run time.
        """
        return cls()

    def panel_options(self) -> Dict[str, Any]:
        """
        The breadcrumb and the live/layout flags the transform tab is rendered with.
        """
        return {
            "breadcrumb": self.breadcrumb,
            "layout": "hier",
            "live": "transforms",
            "empty": self.empty_message,
        }
