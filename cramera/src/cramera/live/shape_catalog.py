"""
Per-shape geometry payloads for the live object overlay.

The overlay publishes any world body the way RViz would render it: each of the body's
shapes travels with its kind, dimensions, colour and local pose, so the viewer can
rebuild the body without knowing how the world was constructed.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import List, Optional

from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Shape,
    Sphere,
)

from cramera.body_geometry import POSE_PRECISION

SIZE_PRECISION = 4
"""
Decimal places shape dimensions are rounded to before publishing.
"""


class ShapeKind(StrEnum):
    """
    How one published shape is rendered by the viewer.
    """

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    MESH = "mesh"


@dataclass(frozen=True)
class ShapeEntry:
    """
    One of a published body's shapes, in the form the viewer builds it from.
    """

    kind: ShapeKind
    """
    The primitive or mesh this entry describes.
    """

    position: List[float]
    """
    The shape's position within its body, as ``[x, y, z]`` in metres.
    """

    quaternion: List[float]
    """
    The shape's orientation within its body, as ``[qx, qy, qz, qw]``.
    """

    color: str
    """
    The shape's colour as a ``#rrggbb`` hex string.
    """

    opacity: float = 1.0
    """
    The shape's opacity between 0 and 1.
    """

    size: Optional[List[float]] = None
    """
    Box extent in metres, set only when :attr:`kind` is ``BOX``.
    """

    radius: Optional[float] = None
    """
    Radius in metres, set when :attr:`kind` is ``CYLINDER`` or ``SPHERE``.
    """

    height: Optional[float] = None
    """
    Height in metres, set only when :attr:`kind` is ``CYLINDER``.
    """

    mesh: Optional[str] = None
    """
    URL the shape's mesh file is served from, set only when :attr:`kind` is ``MESH``.
    """

    mtl: Optional[str] = None
    """
    URL of an OBJ mesh's companion ``.mtl`` file, or None when it has none.
    """

    format: Optional[str] = None
    """
    Mesh file extension, set only when :attr:`kind` is ``MESH``.
    """

    scale: Optional[List[float]] = None
    """
    Mesh scale multiplier per axis, set only when :attr:`kind` is ``MESH``.
    """


def color_to_hex(color: Color) -> str:
    """
    A colour as the ``#rrggbb`` hex string the viewer applies.

    :param color: The colour to convert, with channels between 0 and 1.
    """
    return "#%02x%02x%02x" % (
        round(color.R * 255),
        round(color.G * 255),
        round(color.B * 255),
    )


def is_default_white(color: Color) -> bool:
    """
    Whether a colour is the untouched default, meaning no colour was chosen at all.

    :param color: The colour to check.
    """
    return (color.R, color.G, color.B, color.A) == (1.0, 1.0, 1.0, 1.0)


def served_mesh_file(shape: Shape) -> Optional[str]:
    """
    The mesh file a shape can be served from, or None when it has none.

    :param shape: The shape whose backing file is looked up.
    """
    if not isinstance(shape, Mesh) or not shape.filename:
        return None
    if not Path(shape.filename).is_file():
        return None
    return shape.filename


def companion_mtl_url(mesh_file: str, mesh_url: str) -> Optional[str]:
    """
    The URL an OBJ mesh's companion ``.mtl`` is served from, or None without one.

    The companion is served as a side asset of the mesh itself, so the viewer's material
    loader can fetch it (and the textures it references) relative to the mesh's own URL.

    :param mesh_file: The mesh file's path on disk.
    :param mesh_url: The URL the mesh itself is served from.
    """
    mesh_path = Path(mesh_file)
    if mesh_path.suffix.lower() != ".obj":
        return None
    companion = mesh_path.with_suffix(".mtl")
    if not companion.is_file():
        return None
    return "%s&side=%s" % (mesh_url, urllib.parse.quote(companion.name))


def shape_entry(
    shape: Shape,
    mesh_url: Optional[str],
    fallback_size: List[float],
    fallback_color: str,
) -> ShapeEntry:
    """
    One shape as the viewer builds it.

    A mesh whose backing file is gone degrades to a fallback-sized box, so the body
    still occupies its place in the scene instead of vanishing.

    :param shape: The shape to publish.
    :param mesh_url: URL the shape's mesh is served from, or None for primitives and for
        meshes without a servable file.
    :param fallback_size: Box extent used when a mesh has no servable file.
    :param fallback_color: Colour used when the shape carries no colour of its own.
    """
    local_pose = [
        round(value, POSE_PRECISION)
        for value in shape.origin.to_position_quaternion_list()
    ]
    position, quaternion = local_pose[:3], local_pose[3:]
    color = (
        fallback_color if is_default_white(shape.color) else color_to_hex(shape.color)
    )
    opacity = float(shape.color.A)
    if isinstance(shape, Box):
        return ShapeEntry(
            kind=ShapeKind.BOX,
            position=position,
            quaternion=quaternion,
            color=color,
            opacity=opacity,
            size=_rounded_axes(shape.scale),
        )
    if isinstance(shape, Cylinder):
        return ShapeEntry(
            kind=ShapeKind.CYLINDER,
            position=position,
            quaternion=quaternion,
            color=color,
            opacity=opacity,
            radius=round(shape.radius, SIZE_PRECISION),
            height=round(shape.height, SIZE_PRECISION),
        )
    if isinstance(shape, Sphere):
        return ShapeEntry(
            kind=ShapeKind.SPHERE,
            position=position,
            quaternion=quaternion,
            color=color,
            opacity=opacity,
            radius=round(shape.radius, SIZE_PRECISION),
        )
    if isinstance(shape, Mesh) and mesh_url is not None:
        return ShapeEntry(
            kind=ShapeKind.MESH,
            position=position,
            quaternion=quaternion,
            color=color,
            opacity=opacity,
            mesh=mesh_url,
            mtl=companion_mtl_url(shape.filename, mesh_url),
            format=Path(shape.filename).suffix.lstrip(".").lower(),
            scale=_rounded_axes(shape.scale),
        )
    return ShapeEntry(
        kind=ShapeKind.BOX,
        position=position,
        quaternion=quaternion,
        color=color,
        opacity=opacity,
        size=list(fallback_size),
    )


def _rounded_axes(scale: Scale) -> List[float]:
    """
    A scale's axes as a rounded ``[x, y, z]`` list.

    :param scale: The scale to read the axes from.
    """
    return [
        round(float(scale.x), SIZE_PRECISION),
        round(float(scale.y), SIZE_PRECISION),
        round(float(scale.z), SIZE_PRECISION),
    ]
