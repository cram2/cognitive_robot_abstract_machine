"""
The motion-statechart tab: live from the bridge, or replayed from a recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, ClassVar, Dict, Optional, TYPE_CHECKING

from cramera.knowledge.recorded_statecharts import RecordedStatecharts
from cramera.knowledge.subgraph import GraphPanelPayload

if TYPE_CHECKING:
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase


@dataclass(kw_only=True)
class ChartViewPayload(GraphPanelPayload):
    """
    The statechart tab.

    Motion statecharts only exist while giskardpy executes them: one is compiled per
    merged motion group and thrown away afterwards. The UI fills this view from the
    bridge's ``/chart`` while attached, and from :attr:`recorded` while replaying a
    recording that captured what its run ticked. A scene with neither has nothing to
    show.
    """

    TAB: ClassVar[Optional[str]] = "chart"

    breadcrumb: str = "motion statechart"
    """
    Breadcrumb label shown above the (initially empty) chart.
    """

    recorded: RecordedStatecharts = field(default_factory=RecordedStatecharts)
    """
    The statecharts the played scene recorded, empty for a scene that recorded none.
    """

    empty_message: str = (
        "Motion statecharts are built and ticked at execution time. "
        "Start the demo with cramera-live and press ◉ Live — "
        "the statechart of the running motion group appears here, "
        "coloured by its node life cycle."
    )
    """
    What the panel shows until a live bridge is attached.
    """

    between_motions_message: str = (
        "No motion was executing at this moment of the recording — "
        "play on, or scrub to one."
    )
    """
    What a replay shows at a recorded moment that had no statechart running.
    """

    @classmethod
    def of_tab(cls, knowledge_base: EpisodeKnowledgeBase) -> ChartViewPayload:
        """
        The statechart tab, filled from whatever the played scene recorded.

        :param knowledge_base: The knowledge base whose scene is being viewed.
        """
        return cls(recorded=RecordedStatecharts.of_scene(knowledge_base.scene_name))

    def panel_options(self) -> Dict[str, Any]:
        """
        The breadcrumb and the live/layout flags the statechart tab is rendered with,
        plus the recorded statecharts a replay follows.
        """
        options = {
            "breadcrumb": self.breadcrumb,
            "layout": "hier",
            "live": "chart",
            "empty": self.empty_message,
        }
        if not self.recorded.is_empty():
            options["recorded"] = self.recorded.to_payload()
            options["empty"] = self.between_motions_message
        return options
