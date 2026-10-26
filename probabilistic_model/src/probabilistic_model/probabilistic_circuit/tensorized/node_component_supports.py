from __future__ import annotations

from dataclasses import dataclass

from random_events.product_algebra import Event
from typing_extensions import List


@dataclass
class NodeComponentSupports:
    """
    What the components of one node of a layer model.
    """

    node: int
    """
    Which node of the layer it is.
    """

    supports: List[Event]
    """
    What each of that node's components models, one event per component.
    """

    def mixes(self) -> bool:
        """
        :return: Whether the node draws on more than one component, which is what makes
            it a mixture rather than a node passing one component through.
        """
        return len(self.supports) > 1
