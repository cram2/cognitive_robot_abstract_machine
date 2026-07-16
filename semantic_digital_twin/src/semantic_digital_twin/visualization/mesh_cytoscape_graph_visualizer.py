from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field

import numpy as np
import trimesh
from typing_extensions import Any, ClassVar, Dict, List, Optional, Tuple, TYPE_CHECKING

from rustworkx_utils.visualization.cytoscape_graph_visualizer import (
    CytoscapeElement,
    CytoscapeGraphVisualizer,
)
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from flask import Flask

MESH_THUMBNAIL_SIZE_PIXELS = 128
"""The pixel width and height of the rendered mesh thumbnails."""

NODE_SIZE_PIXELS = 120
"""The on-screen pixel width and height of every node, mesh or default circle alike; bigger than
the base :class:`~krrood.rustworkx_utils.cytoscape_graph_visualizer.CytoscapeGraphVisualizer`'s
default so mesh thumbnails stay legible."""

MESH_THUMBNAIL_FACE_COLOR = (0.475, 0.65, 0.824, 1.0)
"""The RGBA face color used when rendering a body's visual mesh to a thumbnail."""

MESH_THUMBNAIL_MINIMUM_SHADE = 0.15
"""The darkest a triangle can be shaded, so faces pointing away from the camera stay visible."""

MESH_ANIMATION_FRAME_COUNT = 16
"""How many rotation angles a full spin is split into; each is rendered and cached lazily, only
when the browser actually requests that frame."""

MESH_ANIMATION_FRAME_DURATION_MS = 100
"""How long each rotation frame is shown for; ``MESH_ANIMATION_FRAME_COUNT`` frames at this
duration give a full spin period of 1.6 seconds."""

_VIEW_DIRECTION = np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
"""The (unit) direction thumbnails are viewed from, used for both projection and shading."""


@dataclass
class MeshCytoscapeGraphVisualizer(CytoscapeGraphVisualizer):
    """Render a live rustworkx graph of world entities with Cytoscape.js, drawing each ``Body``
    node using a slowly spinning thumbnail of its :attr:`~semantic_digital_twin.world_description.world_entity.Body.visual`
    mesh instead of a flat circle.

    Nodes without visual geometry, and nodes whose payload is not a
    :class:`~semantic_digital_twin.world_description.world_entity.Body`, keep the default gradient
    circle from :class:`~krrood.rustworkx_utils.cytoscape_graph_visualizer.CytoscapeGraphVisualizer`.

    .. note::
        Thumbnails are rendered server-side with Matplotlib's non-interactive Agg backend, which
        needs no display or OpenGL context. A full spin is split into
        :data:`MESH_ANIMATION_FRAME_COUNT` fixed rotation angles, each rendered and cached only the
        first time the browser's animation loop actually requests it, rather than all up front —
        for a world with hundreds of bodies, eagerly rendering every frame of every mesh before the
        graph could even be shown once caused exactly the kind of pile-up that made the graph
        appear to never load (see :meth:`_mesh_frame_image`). Rendering is serialized with a lock
        for the same reason: the Flask dev server handles each request on its own thread, and
        without serializing, two overlapping requests for the same not-yet-cached frame would both
        render it redundantly.
    """

    required_modules: ClassVar[Tuple[str, ...]] = ("flask", "matplotlib")
    """The libraries needed to serve the application and render mesh thumbnails."""

    _mesh_frame_cache: Dict[Tuple[int, int], Optional[bytes]] = field(
        default_factory=dict, init=False, repr=False
    )
    """Rendered PNG bytes for each (node index, rotation frame) pair, cached after the first
    render; ``None`` when the node has no visual mesh."""

    _mesh_render_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """Serializes mesh rendering across concurrently handled requests; see the class note."""

    def _node_element(self, node_index: int) -> CytoscapeElement:
        """
        :param node_index: The rustworkx index of the node.
        :return: The Cytoscape.js element for the node. Nodes with visual geometry get an
            ``image`` field pointing at their first rotation frame and a ``frameCount`` field the
            browser's animation loop uses to cycle through the rest; this does not render
            anything, it only checks whether the body has visual geometry at all.
        """
        element = super()._node_element(node_index)
        if self._node_mesh(node_index) is not None:
            element["data"]["image"] = f"mesh/{node_index}/0.png"
            element["data"]["frameCount"] = MESH_ANIMATION_FRAME_COUNT
        return element

    def extra_node_styles(self) -> List[Dict[str, Any]]:
        """
        :return: Style rules that enlarge every node and fill nodes carrying an ``image`` field
            with that image instead of the default gradient, on top of the base node styles.

        Mesh nodes use a rounded square shape with ``background-fit: contain`` rather than the
        default circle, so the whole thumbnail is visible uncropped instead of having its corners
        clipped away by a circular mask.
        """
        hover_size = NODE_SIZE_PIXELS * 1.25
        return super().extra_node_styles() + [
            {
                "selector": "node",
                "style": {"width": NODE_SIZE_PIXELS, "height": NODE_SIZE_PIXELS},
            },
            {
                # Overrides the base template's fixed 38px hover size, which would otherwise
                # shrink nodes bigger than that on hover instead of growing them.
                "selector": "node.hovered",
                "style": {"width": hover_size, "height": hover_size},
            },
            {
                "selector": "node[image]",
                "style": {
                    "shape": "round-rectangle",
                    "background-image": "data(image)",
                    "background-fit": "contain",
                    "background-opacity": 0,
                },
            },
        ]

    def extra_script(self) -> str:
        """
        :return: A JavaScript animation loop that cycles every mesh node's ``image`` field through
            its rotation frames, making it appear to spin in place.
        """
        return f"""
(function() {{
  setInterval(() => {{
    cy.nodes("[frameCount]").forEach((node) => {{
      const frameCount = node.data("frameCount");
      const frame = Math.floor(Date.now() / {MESH_ANIMATION_FRAME_DURATION_MS}) % frameCount;
      node.data("image", "mesh/" + node.id() + "/" + frame + ".png");
    }});
  }}, {MESH_ANIMATION_FRAME_DURATION_MS});
}})();
"""

    def register_additional_routes(self, application: Flask) -> None:
        """
        Serve each node's rendered mesh rotation frames at ``/mesh/<node_index>/<frame>.png``.

        :param application: The Flask application to add the route to.
        """
        super().register_additional_routes(application)
        from flask import Response

        @application.route("/mesh/<int:node_index>/<int:frame>.png")
        def mesh_image(node_index: int, frame: int):
            image_bytes = self._mesh_frame_image(node_index, frame)
            if image_bytes is None:
                return Response(status=404)
            return Response(image_bytes, mimetype="image/png")

    def _node_mesh(self, node_index: int) -> Optional[trimesh.Trimesh]:
        """
        :param node_index: The rustworkx index of the node.
        :return: The node's combined visual mesh, or ``None`` if it has no visual geometry. This
            only assembles the mesh; it never renders an image.
        """
        payload = self.graph[node_index]
        if not isinstance(payload, Body) or not payload.visual:
            return None
        return payload.visual.combined_mesh

    def _mesh_frame_image(self, node_index: int, frame: int) -> Optional[bytes]:
        """
        :param node_index: The rustworkx index of the node.
        :param frame: The rotation frame, wrapped into ``[0, MESH_ANIMATION_FRAME_COUNT)``.
        :return: The node's rendered mesh thumbnail at that rotation as PNG bytes, cached after the
            first render, or ``None`` if it has no visual geometry.
        """
        key = (node_index, frame % MESH_ANIMATION_FRAME_COUNT)
        if key in self._mesh_frame_cache:
            return self._mesh_frame_cache[key]
        with self._mesh_render_lock:
            if key not in self._mesh_frame_cache:
                mesh = self._node_mesh(node_index)
                if mesh is None:
                    self._mesh_frame_cache[key] = None
                else:
                    angle = 2 * np.pi * key[1] / MESH_ANIMATION_FRAME_COUNT
                    self._mesh_frame_cache[key] = self._render_mesh_thumbnail(
                        mesh, angle
                    )
            return self._mesh_frame_cache[key]

    @staticmethod
    def _render_mesh_thumbnail(
        mesh: trimesh.Trimesh, rotation_angle: float = 0.0
    ) -> bytes:
        """
        Render a mesh to a transparent PNG thumbnail using Matplotlib's non-interactive Agg backend.

        The mesh is rotated around the vertical (Z) axis by ``rotation_angle`` and then projected
        by hand with a fixed isometric-style view direction, drawn back-to-front (painter's
        algorithm) as flat-shaded 2D polygons, rather than going through ``mpl_toolkits.mplot3d``,
        since that toolkit ties its version tightly to the Matplotlib install and is prone to
        breaking when multiple Matplotlib versions are present.

        :param mesh: The mesh to render.
        :param rotation_angle: The angle, in radians, to spin the mesh around its vertical axis
            before rendering.
        :return: The rendered thumbnail as PNG bytes.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import PolyCollection
        from matplotlib.figure import Figure

        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        spin = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
        triangles = (mesh.vertices[mesh.faces]) @ spin.T  # (F, 3, 3)
        face_normals = mesh.face_normals @ spin.T

        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, _VIEW_DIRECTION)
        right /= np.linalg.norm(right)
        camera_up = np.cross(_VIEW_DIRECTION, right)

        # _VIEW_DIRECTION points from the camera toward the mesh, so nearer triangles have a
        # *lower* depth; sort descending to get back-to-front (farthest painted first).
        order = np.argsort(-(triangles.mean(axis=1) @ _VIEW_DIRECTION))

        polygons = np.stack([triangles @ right, triangles @ camera_up], axis=-1)[order]

        shade = np.clip(
            face_normals[order] @ -_VIEW_DIRECTION, MESH_THUMBNAIL_MINIMUM_SHADE, 1.0
        )
        colors = np.tile(np.array(MESH_THUMBNAIL_FACE_COLOR), (len(order), 1))
        colors[:, :3] *= shade[:, None]

        bounds = polygons.reshape(-1, 2)
        center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        radius = max((bounds.max(axis=0) - bounds.min(axis=0)).max() / 2, 1e-6)

        inches = MESH_THUMBNAIL_SIZE_PIXELS / 100
        figure = Figure(figsize=(inches, inches), dpi=100)
        FigureCanvasAgg(figure)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_axis_off()
        axes.set_aspect("equal")
        axes.set_xlim(center[0] - radius, center[0] + radius)
        axes.set_ylim(center[1] - radius, center[1] + radius)
        axes.add_collection(
            PolyCollection(polygons, facecolor=colors, edgecolor="none")
        )

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", transparent=True)
        return buffer.getvalue()
