"""
Visualization helpers for the tool-based action experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Shape
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Optional

logger = logging.getLogger(__name__)

try:
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    rclpy = None
    VizMarkerPublisher = None
    logger.info(
        "Could not import VizMarkerPublisher. This is probably because you are not "
        "running ROS."
    )


def start_visualization_with_collision_markers(world: World) -> None:
    """
    Publish the world, its tf tree, and closest-point collision results to RViz.

    Does nothing if ROS is not available.
    """
    if VizMarkerPublisher is None:
        return
    rclpy.init()
    node = rclpy.create_node("viz_marker")
    VizMarkerPublisher(_world=world, node=node).with_tf_and_collision_visualization()


@dataclass(frozen=True)
class ShapeColor:
    """
    The color one shape had before it was dyed, so it can be restored.
    """

    shape: Shape
    """
    The shape whose color was replaced.
    """

    color: Color
    """
    The shape's color before the highlight.
    """


@dataclass
class TargetHighlight:
    """
    Dyes a target body in a highlight color while the robot approaches and acts on it,
    and restores the original colors afterwards.

    Use as a context manager around the action performance. The world's visualization
    publishers pick the color change up through the model change notification, so the
    highlighted target stands out in RViz.
    """

    world: World
    """
    The world the target lives in.
    """

    body: Optional[Body]
    """
    The body to highlight, or None for targets that are pure poses (e.g. wiping
    patches), which are left untouched.
    """

    color: Color = field(default_factory=lambda: Color(R=0.1, G=0.4, B=1.0))
    """
    The color the target is dyed with while it is highlighted.
    """

    _original_colors: List[ShapeColor] = field(init=False, default_factory=list)
    """
    The shape colors to restore when the highlight ends.
    """

    def __enter__(self) -> TargetHighlight:
        """
        Dye the target body in the highlight color, remembering the original colors.

        :return: This highlight.
        """
        if self.body is None:
            return self
        with self.world.modify_world():
            for shape in self.body.visual.shapes:
                self._original_colors.append(
                    ShapeColor(shape=shape, color=shape.color)
                )
                shape.color = self.color
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        """
        Restore the original colors of the target body, also when the action failed.
        """
        if self.body is None:
            return
        with self.world.modify_world():
            for original in self._original_colors:
                original.shape.color = original.color
        self._original_colors.clear()
