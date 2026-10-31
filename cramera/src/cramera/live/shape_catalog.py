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
    Cylinder,
    Mesh,
    Scale,
    Shape,
    Sphere,
)

from cramera.body_geometry import NumericPose
from cramera.onboard.bundle_urdf import companion_material_library

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


def served_mesh_file(shape: Shape) -> Optional[str]:
    """
    The mesh file a shape can be served from, or None when it has none.

    :param shape: The shape whose backing file is looked up.
    :return: The resolved native mesh file, or None for geometry without a file.
    """
    if not isinstance(shape, Mesh):
        return None
    mesh_file = shape.local_file
    if not mesh_file.is_file():
        return None
    return str(mesh_file)


def companion_mtl_url(mesh_file: str, mesh_url: str) -> Optional[str]:
    """
    The URL an OBJ mesh's companion ``.mtl`` is served from, or None without one.

    The companion is served as a side asset of the mesh itself, so the viewer's material
    loader can fetch it (and the textures it references) relative to the mesh's own URL.

    :param mesh_file: The mesh file's path on disk.
    :param mesh_url: The URL the mesh itself is served from.
    :return: The declared material library's URL, or None without one.
    """
    mesh_path = Path(mesh_file)
    companion = companion_material_library(mesh_path.parent, mesh_path.name)
    if companion is None:
        return None
    return "%s&side=%s" % (mesh_url, urllib.parse.quote(companion, safe=""))


def shape_entry(
    shape: Shape,
    mesh_url: Optional[str],
) -> ShapeEntry:
    """
    One shape as the viewer builds it.

    :param shape: The shape to publish.
    :param mesh_url: URL the shape's mesh is served from, or None for primitives and for
        meshes without a servable file.
    :return: The shape's native geometry and appearance in the browser payload.
    :raises FileNotFoundError: When the mesh has no servable file.
    :raises TypeError: When the shape is not a supported native geometry type.
    """
    local_pose = NumericPose.of_matrix(shape.origin.to_np()).rounded()
    position, quaternion = local_pose[:3], local_pose[3:]
    color = shape.color.to_hex()
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
        mesh_file = shape.local_file
        return ShapeEntry(
            kind=ShapeKind.MESH,
            position=position,
            quaternion=quaternion,
            color=color,
            opacity=opacity,
            mesh=mesh_url,
            mtl=companion_mtl_url(str(mesh_file), mesh_url),
            format=mesh_file.suffix.lstrip(".").lower(),
            scale=_rounded_axes(shape.scale),
        )
    if isinstance(shape, Mesh):
        raise FileNotFoundError(shape.filename)
    raise TypeError(f"Unsupported shape type: {type(shape).__name__}")


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
