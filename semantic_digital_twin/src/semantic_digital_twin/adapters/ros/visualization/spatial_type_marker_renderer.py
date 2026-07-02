from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import (
    Point as RosPoint,
    Pose as RosPose,
    Quaternion as RosQuaternion,
    Vector3 as RosVector3,
)
from std_msgs.msg import ColorRGBA, Header
from typing_extensions import ClassVar, TypeVar
from visualization_msgs.msg import Marker

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    ColorToRos2Converter,
)
from semantic_digital_twin.adapters.ros.visualization.exceptions import (
    CannotRenderSpatialTypeError,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Pose2D,
    Quaternion,
    RotationMatrix,
    SpatialType,
    Vector3,
)
from semantic_digital_twin.world_description.geometry import Color

SpatialTypeInput = TypeVar("SpatialTypeInput", bound=SpatialType)


@dataclass
class SpatialTypeVisualization:
    """A spatial type together with the display metadata needed to render it as markers."""

    spatial_type: SpatialType
    """The spatial type to visualize. Its current value is resolved from the live world state."""

    color: Color = field(default_factory=Color)
    """The color of the markers. Ignored for the fixed RGB axes of pose-like spatial types."""

    namespace: str = "spatial_type"
    """The marker namespace. RViz scopes marker ids per namespace, so it must be unique per request."""

    marker_id_offset: int = 0
    """The offset added to each marker's local id. The first marker of a request uses it directly, further markers add small increments."""

    label: str | None = None
    """An optional text label rendered alongside the markers."""

    lifetime_seconds: float = 0.0
    """How long the markers stay alive. ``0`` keeps them indefinitely."""

    arrow_length: float = 0.25
    """The length of the axis arrows used to visualize orientations."""

    sphere_diameter: float = 0.05
    """The diameter of the sphere used to visualize a point."""


@dataclass
class SpatialTypeMarkerRenderer(SubClassSafeGeneric[SpatialTypeInput], ABC):
    """
    Renders a single category of spatial type into one or more RViz markers.

    Subclasses handle one spatial-type category each, keeping the hierarchy open for
    extension: a new spatial type only needs a new subclass, no existing code changes.
    """

    @classmethod
    def input_type(cls) -> type[SpatialTypeInput]:
        """The spatial type category handled by this renderer."""
        return cls.get_generic_type()

    @classmethod
    def can_render(cls, spatial_type: SpatialType) -> bool:
        """
        Whether this renderer can render the given spatial type.

        :param spatial_type: The spatial type to check.
        """
        return isinstance(spatial_type, cls.input_type())

    @classmethod
    def renderer_for(cls, spatial_type: SpatialType) -> type[SpatialTypeMarkerRenderer]:
        """
        Find the renderer responsible for the given spatial type.

        :param spatial_type: The spatial type to find a renderer for.
        :return: The renderer responsible for rendering the spatial type.
        """
        for subclass in recursive_subclasses(cls):
            if subclass.can_render(spatial_type):
                return subclass
        raise CannotRenderSpatialTypeError(spatial_type_type=type(spatial_type))

    @classmethod
    def render(
        cls, request: SpatialTypeVisualization, root_frame_name: str
    ) -> list[Marker]:
        """
        Render the requested spatial type into markers.

        :param request: The spatial type and its display metadata.
        :param root_frame_name: The frame used when the spatial type has no reference frame.
        :return: The markers representing the spatial type.
        """
        renderer = cls.renderer_for(request.spatial_type)
        return renderer.render_markers(request, root_frame_name)

    @classmethod
    @abstractmethod
    def render_markers(
        cls, request: SpatialTypeVisualization, root_frame_name: str
    ) -> list[Marker]:
        """
        Build the markers for the spatial type of the given request.

        :param request: The spatial type and its display metadata.
        :param root_frame_name: The frame used when the spatial type has no reference frame.
        :return: The markers representing the spatial type.
        """
        raise NotImplementedError

    @staticmethod
    def _reference_frame_name(spatial_type: SpatialType, root_frame_name: str) -> str:
        """
        The name of the spatial type's reference frame, falling back to the root frame.

        :param spatial_type: The spatial type to resolve the reference frame of.
        :param root_frame_name: The frame used when the spatial type has no reference frame.
        :return: The name of the spatial type's reference frame, or the root frame.
        """
        reference_frame = spatial_type.reference_frame
        if reference_frame is None:
            return root_frame_name
        return str(reference_frame.name)

    @classmethod
    def _base_marker(
        cls, request: SpatialTypeVisualization, marker_id: int, frame_id: str
    ) -> Marker:
        """
        Create a marker with the shared fields filled in.

        :param request: The spatial type and its display metadata.
        :param marker_id: The marker's local id.
        :param root_frame_name: The frame used when the spatial type has no reference frame.
        :return: The marker with the shared fields filled in.
        """
        marker = Marker()
        marker.action = Marker.ADD
        marker.ns = request.namespace
        marker.id = request.marker_id_offset + marker_id
        marker.header = Header(frame_id=frame_id)
        marker.frame_locked = True
        marker.color = ColorToRos2Converter.convert(request.color)
        marker.lifetime = Duration(sec=round(request.lifetime_seconds))
        return marker

    @classmethod
    def _text_marker(
        cls,
        request: SpatialTypeVisualization,
        marker_id: int,
        frame_id: str,
        pose: RosPose,
    ) -> Marker:
        """
        Create the optional label marker shared by the renderers.

        :param request: The spatial type and its display metadata.
        :param marker_id: The marker's local id.
        :param pose: The pose of the label.
        :return: The marker representing the label.
        """
        marker = cls._base_marker(request, marker_id, frame_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.text = request.label
        marker.pose = pose
        marker.scale = RosVector3(z=0.1)
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        return marker


@dataclass
class Point3MarkerRenderer(SpatialTypeMarkerRenderer[Point3]):
    """Renders a point as a single sphere at its current position."""

    @classmethod
    def render_markers(
        cls, request: SpatialTypeVisualization, root_frame_name: str
    ) -> list[Marker]:
        """
        Build a sphere marker at the point's position, plus an optional label.
        """
        frame_id = cls._reference_frame_name(request.spatial_type, root_frame_name)
        position = request.spatial_type.evaluate()
        marker = cls._base_marker(request, 0, frame_id)
        marker.type = Marker.SPHERE
        marker.pose.position = RosPoint(x=position[0], y=position[1], z=position[2])
        marker.pose.orientation = RosQuaternion(w=1.0)
        diameter = request.sphere_diameter
        marker.scale = RosVector3(x=diameter, y=diameter, z=diameter)
        markers = [marker]
        if request.label:
            markers.append(cls._text_marker(request, 1, frame_id, marker.pose))
        return markers


@dataclass
class Vector3MarkerRenderer(SpatialTypeMarkerRenderer[Vector3]):
    """Renders a vector as a single arrow anchored at the origin of its visualisation frame."""

    @classmethod
    def render_markers(
        cls, request: SpatialTypeVisualization, root_frame_name: str
    ) -> list[Marker]:
        """
        Build an arrow marker from the origin along the vector, plus an optional label.
        """
        vector = request.spatial_type
        if vector.visualisation_frame is not None:
            frame_id = str(vector.visualisation_frame.name)
        else:
            frame_id = cls._reference_frame_name(vector, root_frame_name)
        components = vector.evaluate()
        marker = cls._base_marker(request, 0, frame_id)
        marker.type = Marker.ARROW
        marker.points = [
            RosPoint(),
            RosPoint(x=components[0], y=components[1], z=components[2]),
        ]
        marker.scale = RosVector3(x=0.025, y=0.05, z=0.1)
        markers = [marker]
        if request.label:
            markers.append(cls._text_marker(request, 1, frame_id, RosPose()))
        return markers


@dataclass
class PoseLikeMarkerRenderer(SpatialTypeMarkerRenderer[Pose]):
    """Renders any orientation-carrying spatial type as an RGB axis triad with an optional label."""

    _renderable_types: ClassVar[tuple[type[SpatialType], ...]] = (
        Pose,
        Pose2D,
        HomogeneousTransformationMatrix,
        RotationMatrix,
        Quaternion,
    )
    """The spatial types rendered as an axis triad."""

    @classmethod
    def can_render(cls, spatial_type: SpatialType) -> bool:
        return isinstance(spatial_type, cls._renderable_types)

    @classmethod
    def render_markers(
        cls, request: SpatialTypeVisualization, root_frame_name: str
    ) -> list[Marker]:
        """
        Build an RGB axis triad at the pose, plus an optional label.
        """
        frame_id = cls._reference_frame_name(request.spatial_type, root_frame_name)
        position, orientation = cls._position_and_orientation(request.spatial_type)
        pose = RosPose(
            position=RosPoint(x=position[0], y=position[1], z=position[2]),
            orientation=RosQuaternion(
                x=orientation[0],
                y=orientation[1],
                z=orientation[2],
                w=orientation[3],
            ),
        )
        markers = [
            cls._axis_marker(request, axis_index, frame_id, pose)
            for axis_index in range(3)
        ]
        if request.label:
            markers.append(cls._text_marker(request, 3, frame_id, pose))
        return markers

    @classmethod
    def _axis_marker(
        cls,
        request: SpatialTypeVisualization,
        axis_index: int,
        frame_id: str,
        pose: RosPose,
    ) -> Marker:
        """
        Create one colored arrow for the given axis of the triad.

        :param request: The spatial type and its display metadata.
        :param axis_index: The index of the axis to render.
        :param frame_id: The frame used when the spatial type has no reference frame.
        :param pose: The pose of the triad.
        """
        end_point = [0.0, 0.0, 0.0]
        end_point[axis_index] = request.arrow_length
        axis_color = [0.0, 0.0, 0.0, 1.0]
        axis_color[axis_index] = 1.0
        marker = cls._base_marker(request, axis_index, frame_id)
        marker.type = Marker.ARROW
        marker.pose = pose
        marker.points = [
            RosPoint(),
            RosPoint(x=end_point[0], y=end_point[1], z=end_point[2]),
        ]
        marker.scale = RosVector3(x=0.025, y=0.05, z=0.05)
        marker.color = ColorRGBA(
            r=axis_color[0], g=axis_color[1], b=axis_color[2], a=axis_color[3]
        )
        return marker

    @staticmethod
    def _position_and_orientation(
        spatial_type: SpatialType,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resolve the current position and orientation quaternion of the spatial type.

        :param spatial_type: The spatial type to resolve.
        """
        if isinstance(spatial_type, Quaternion):
            return np.zeros(3), spatial_type.evaluate()
        if isinstance(spatial_type, RotationMatrix):
            return np.zeros(3), spatial_type.to_quaternion().evaluate()
        return (
            spatial_type.to_position().evaluate()[:3],
            spatial_type.to_quaternion().evaluate(),
        )
