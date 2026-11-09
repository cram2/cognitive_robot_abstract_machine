"""
Publishes spatial types as RViz markers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from krrood.symbolic_math.symbolic_math import (
    CompiledFunctionWithViews,
    FloatVariable,
    Matrix,
    VariableParameters,
)
from krrood_spatial_math.spatial_types import SpatialType
from semantic_digital_twin.adapters.ros.ros2_node import HasROS2Node
from semantic_digital_twin.adapters.ros.visualization.exceptions import (
    WorldNotResolvableError,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeMarkerRenderer,
    SpatialTypeVisualization,
)
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.world_description.geometry import Color

if TYPE_CHECKING:
    from rclpy.publisher import Publisher

    from semantic_digital_twin.world import World

# %% evaluated visualizations


@dataclass(frozen=True)
class MarkerIdentity:
    """
    Identifies a published marker by the fields RViz uses to track it.
    """

    namespace: str
    """
    The marker namespace (``Marker.ns``).
    """

    marker_id: int
    """
    The marker id within the namespace (``Marker.id``).
    """


@dataclass
class EvaluatedSpatialTypeVisualization:
    """
    A visualization request together with the value its markers are drawn from.
    """

    visualization: SpatialTypeVisualization
    """
    The spatial type and its display metadata.
    """

    rendered_value: np.ndarray
    """
    The evaluated :meth:`SpatialTypeMarkerRenderer.rendered_expression` of the spatial
    type.
    """


# %% publishing evaluated spatial types


@dataclass(eq=False)
class EvaluatedSpatialTypePublisher(HasROS2Node):
    """
    Publishes already evaluated spatial type visualizations as one RViz marker array.

    Markers of the previous call that are no longer rendered are deleted. It evaluates
    no expression, so it may run on a thread other than the one evaluating them.

    .. warning:: Publish through one instance from one thread at a time.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The topic the markers are published on.
    """

    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """
    The QoS profile of the publisher.

    Latched so RViz sees the last markers on connect.
    """

    publisher: Publisher = field(init=False)
    """
    The ROS2 marker array publisher.
    """

    _published_marker_identities: set[MarkerIdentity] = field(
        init=False, default_factory=set
    )
    """
    Identities of the markers published on the previous call, used to delete stale ones.
    """

    def __post_init__(self):
        self.publisher = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )

    def publish(
        self,
        evaluated_visualizations: Iterable[EvaluatedSpatialTypeVisualization],
        root_frame_name: str,
    ) -> None:
        """
        Render the visualizations and publish them, deleting markers no longer rendered.

        .. warning:: The rendered values must not change during the call; pass copies
            when they are views on a buffer another thread writes to.

        :param evaluated_visualizations: The visualizations to show; all others are deleted.
        :param root_frame_name: The frame used when a spatial type has no reference frame.
        """
        marker_array = MarkerArray()
        for evaluated_visualization in evaluated_visualizations:
            marker_array.markers.extend(
                SpatialTypeMarkerRenderer.render(
                    evaluated_visualization.visualization,
                    evaluated_visualization.rendered_value,
                    root_frame_name,
                )
            )
        current_identities = {
            MarkerIdentity(marker.ns, marker.id) for marker in marker_array.markers
        }
        for identity in self._published_marker_identities - current_identities:
            marker_array.markers.append(self._delete_marker(identity))
        self.publisher.publish(marker_array)
        self._published_marker_identities = current_identities

    @staticmethod
    def _delete_marker(identity: MarkerIdentity) -> Marker:
        """
        Build a marker that deletes the previously published marker with this identity.
        """
        marker = Marker()
        marker.action = Marker.DELETE
        marker.ns = identity.namespace
        marker.id = identity.marker_id
        return marker


# %% live world-state visualization


@dataclass(eq=False)
class SpatialTypePublisher(StateChangeCallback, HasROS2Node):
    """
    Publishes spatial types as RViz markers and keeps them updated as the robot moves.

    The registered spatial types are compiled together once per change of the
    registration and evaluated once per joint-state update, so symbolic expressions
    track motion.
    """

    marker_publisher: EvaluatedSpatialTypePublisher = field(init=False)
    """
    Publishes the evaluated spatial types on :attr:`node`.
    """

    _requests: list[SpatialTypeVisualization] = field(init=False, default_factory=list)
    """
    The spatial types that are re-evaluated and republished on every state change.
    """

    _variables: list[FloatVariable] = field(init=False, default_factory=list)
    """
    The free variables of the registered spatial types, in the order
    :attr:`_compiled_rendered_values` takes their values.
    """

    _compiled_rendered_values: CompiledFunctionWithViews = field(init=False)
    """
    Evaluates the rendered expression of every registered spatial type at once.
    """

    def __post_init__(self):
        super().__post_init__()
        self.marker_publisher = EvaluatedSpatialTypePublisher(node=self.node)
        self._compile()

    def add(self, request: SpatialTypeVisualization) -> None:
        """
        Register a spatial type for visualization and publish immediately.
        """
        self.set_requests([*self._requests, request])

    def add_all(self, requests: Iterable[SpatialTypeVisualization]) -> None:
        """
        Register several spatial types for visualization and publish immediately.
        """
        self.set_requests([*self._requests, *requests])

    def set_requests(self, requests: Iterable[SpatialTypeVisualization]) -> None:
        """
        Replace all registered spatial types and publish immediately.
        """
        self._requests = list(requests)
        self._compile()
        self.publish()

    def clear(self) -> None:
        """
        Remove all registered spatial types and delete their markers.
        """
        self.set_requests([])

    def on_state_change(self, **kwargs) -> None:
        """
        Republish all registered spatial types whenever the world state changes.
        """
        self.publish()

    def publish(self) -> None:
        """
        Evaluate all registered spatial types against the current world state and
        publish them, deleting the markers of spatial types no longer registered.
        """
        values = np.array(
            [variable.resolve() for variable in self._variables], dtype=np.float64
        )
        rendered_values = self._compiled_rendered_values(values)
        self.marker_publisher.publish(
            [
                EvaluatedSpatialTypeVisualization(
                    visualization=request, rendered_value=rendered_value.copy()
                )
                for request, rendered_value in zip(self._requests, rendered_values)
            ],
            str(self._world.root.name),
        )

    def _compile(self) -> None:
        """
        Compile the rendered expressions of the registered spatial types into one
        function of their free variables.
        """
        rendered_expressions = [
            SpatialTypeMarkerRenderer.find_spatial_type_renderer(
                request.spatial_type
            ).rendered_expression(request.spatial_type)
            for request in self._requests
        ]
        self._variables = Matrix.vstack(rendered_expressions).free_variables()
        self._compiled_rendered_values = CompiledFunctionWithViews(
            expressions=rendered_expressions,
            parameters=VariableParameters.from_lists(self._variables),
        )

    @classmethod
    def publish_once(
        cls,
        *,
        spatial_type: SpatialType,
        node: Node,
        color: Color | None = None,
        label: str | None = None,
        world: World | None = None,
    ) -> SpatialTypePublisher:
        """
        Render and publish a single spatial type immediately.

        Convenience entry point for debugging: resolves the world from the spatial
        type's reference frame when not given. The returned publisher keeps tracking
        the spatial type until :meth:`stop` is called.

        :raises WorldNotResolvableError: When no world is given and the spatial type has no reference frame.
        """
        if world is None:
            reference_frame = spatial_type.reference_frame
            if reference_frame is None:
                raise WorldNotResolvableError(spatial_type=spatial_type)
            world = reference_frame._world
        publisher = cls(node=node, _world=world)
        publisher.add(
            SpatialTypeVisualization(
                spatial_type=spatial_type, color=color or Color(), label=label
            )
        )
        return publisher
