from __future__ import annotations

import threading
from dataclasses import dataclass, field

import trimesh
from typing_extensions import Any, ClassVar, Dict, Optional, Tuple, TYPE_CHECKING

from rustworkx_utils.visualization.three_graph_visualizer import ThreeGraphVisualizer
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from flask import Flask

_THREE_VERSION = "0.147.0"
"""The three.js release used for the separately loaded ``GLTFLoader``.

Pinned to the newest release that still ships the classic, non-module
``examples/js/loaders/GLTFLoader.js`` (three.js moved to ES-modules-only examples at 0.148.0, and
dropped the classic ``build/three.min.js`` UMD bundle entirely at 0.161.0); the page template loads
scripts as plain ``<script>`` tags with no bundler or import map. Needs to be recent enough for
Object3D compatibility with whatever three.js 3d-force-graph bundles internally — 0.128.0 (pre
r133) caused ``TypeError: e.removeFromParent is not a function`` when 3d-force-graph tried to add a
loaded mesh to the scene, since it bundles a newer three.js than that. Kept separate from the base
class rather than shared: the base class's node labels are plain DOM overlays now (no separate
three.js needed there at all) after ``three-spritetext``'s Sprite-based approach ran into a similar
but *unfixable* version mismatch — 3d-force-graph's internal three.js bundle is newer than any
release that still ships a classic (non-module) build, and Sprite materials are compiled against
shader chunks that must match exactly, unlike plain meshes/geometries loaded through GLTFLoader,
which tolerate the version gap fine.
"""

MESH_ROTATION_RADIANS_PER_SECOND = 0.6
"""How fast mesh nodes spin around their vertical axis."""

MESH_TARGET_SIZE_UNITS = 25.0
"""The on-screen size every loaded mesh is normalized to, regardless of its real-world dimensions.

A body's visual mesh is scaled in meters (a screw and a wardrobe differ by orders of magnitude),
while 3d-force-graph's layout space uses much larger arbitrary units unrelated to any real-world
scale (its default node sphere has a radius on the order of a few units) — without normalizing,
most real-world meshes would render far too small to see next to the graph's node spacing.
"""


@dataclass
class MeshThreeGraphVisualizer(ThreeGraphVisualizer):
    """Render a live rustworkx graph of world entities with 3d-force-graph, drawing each ``Body``
    node using its actual :attr:`~semantic_digital_twin.world_description.world_entity.Body.visual`
    mesh, loaded once in the browser via glTF and spun continuously in real WebGL 3D — rather than
    :class:`~semantic_digital_twin.visualization.mesh_cytoscape_graph_visualizer.MeshCytoscapeGraphVisualizer`'s
    server-rendered rotating sprite frames.

    Nodes without visual geometry, and nodes whose payload is not a
    :class:`~semantic_digital_twin.world_description.world_entity.Body`, get a plain sphere instead
    of a mesh. Every node keeps the base class's always-visible label.

    .. note::
        Exporting a mesh to glTF (via ``trimesh``) is cheap compared to rendering it, so unlike the
        Cytoscape sprite-frame variant this needs no server-side rendering at all — just a one-time
        export per node, cached and guarded by a lock for the same reason
        :class:`~semantic_digital_twin.visualization.mesh_cytoscape_graph_visualizer.MeshCytoscapeGraphVisualizer`
        needs one: overlapping polls must not export the same mesh redundantly.
    """

    required_modules: ClassVar[Tuple[str, ...]] = ("flask", "trimesh")
    """The libraries needed to serve the application and export meshes to glTF."""

    _mesh_glb_cache: Dict[int, Optional[bytes]] = field(
        default_factory=dict, init=False, repr=False
    )
    """Exported glTF-binary (GLB) bytes for each node index, cached after the first export; ``None``
    when the node has no visual mesh."""

    _mesh_export_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """Serializes GLB export across concurrently handled requests; see the class note."""

    def node_extra_data(self, node_index: int) -> Dict[str, Any]:
        """
        :param node_index: The rustworkx index of the node.
        :return: A ``meshUrl`` field when the node's
            :class:`~semantic_digital_twin.world_description.world_entity.Body` has visual geometry,
            for the browser to load and spin; empty otherwise.
        """
        if self._node_mesh(node_index) is None:
            return {}
        return {"meshUrl": f"mesh/{node_index}.glb"}

    def register_additional_routes(self, application: Flask) -> None:
        """
        Serve each node's exported mesh at ``/mesh/<node_index>.glb``.

        :param application: The Flask application to add the route to.
        """
        super().register_additional_routes(application)
        from flask import Response

        @application.route("/mesh/<int:node_index>.glb")
        def mesh_glb(node_index: int):
            glb_bytes = self._mesh_glb(node_index)
            if glb_bytes is None:
                return Response(status=404)
            return Response(glb_bytes, mimetype="model/gltf-binary")

    def extra_head(self) -> str:
        """
        :return: Blocking ``<script>`` tags for three.js and ``GLTFLoader``.

        These load *before* the main script runs (see the base class note on why this matters:
        3d-force-graph only resolves a node's rendered object once, so ``THREE``/``GLTFLoader``
        must already be available the first time :meth:`extra_script`'s ``nodeThreeObject``
        callback runs, not merely "eventually" after an async load).
        """
        return f"""
<script src="https://unpkg.com/three@{_THREE_VERSION}/build/three.min.js"></script>
<script src="https://unpkg.com/three@{_THREE_VERSION}/examples/js/loaders/GLTFLoader.js"></script>
"""

    def extra_script(self) -> str:
        """
        :return: JavaScript that gives every node with a ``meshUrl`` its loaded mesh (or a plain
            sphere otherwise) as a continuously spinning custom node object. The base class's
            always-visible labels are unaffected — they're plain DOM overlays positioned from each
            node's ``x``/``y``/``z``, independent of whatever object represents the node visually.

        This fully replaces the base class's ``nodeThreeObject``/``nodeThreeObjectExtend`` setup
        (calling ``graph.nodeThreeObject(...)`` again overrides the previous registration outright,
        it doesn't compose with it), since a loaded mesh must fully replace the default sphere
        rather than sit alongside it.
        """
        return f"""
(function() {{
  const loader = new THREE.GLTFLoader();
  const radiansPerSecond = {MESH_ROTATION_RADIANS_PER_SECOND};
  const targetSize = {MESH_TARGET_SIZE_UNITS};
  let lastFrameTime = performance.now();

  // Keyed by node id rather than cached on the node object itself: each poll's refresh() parses a
  // fresh JSON payload, so the node objects passed to graphData() are brand new instances every
  // time (custom properties set on a previous instance don't carry over). Without this cache,
  // every graphData() update would re-fetch and re-load the mesh, flooding the server with
  // duplicate requests.
  const nodeGroupsById = new Map();

  graph.nodeThreeObjectExtend(false);
  graph.nodeThreeObject((node) => {{
    if (nodeGroupsById.has(node.id)) {{ return nodeGroupsById.get(node.id); }}
    const group = new THREE.Group();
    nodeGroupsById.set(node.id, group);

    if (!node.meshUrl) {{
      group.add(new THREE.Mesh(
        new THREE.SphereGeometry(targetSize / 2),
        new THREE.MeshLambertMaterial({{ color: node.color }})
      ));
      return group;
    }}

    loader.load(
      node.meshUrl,
      (gltf) => {{
        const bounds = new THREE.Box3().setFromObject(gltf.scene);
        const size = bounds.getSize(new THREE.Vector3());
        const largestDimension = Math.max(size.x, size.y, size.z) || 1;
        gltf.scene.scale.setScalar(targetSize / largestDimension);
        // Bodies' meshes come from arbitrary sources (URDF/mesh files, trimesh's glTF export)
        // whose face winding isn't guaranteed to match three.js's front-face convention. With the
        // default single-sided material, a mesh with reversed winding renders solid from one
        // viewing hemisphere and disappears (backface-culled) from the other as it spins or the
        // camera orbits. Rendering both sides trades a little performance for always being visible.
        gltf.scene.traverse((child) => {{
          if (!child.isMesh) {{ return; }}
          const materials = Array.isArray(child.material) ? child.material : [child.material];
          materials.forEach((material) => {{
            material.side = THREE.DoubleSide;
            // glTF's PBR metallic-roughness model defaults an unset metallic factor to 1 (fully
            // metallic), which trimesh's export doesn't override for meshes without their own
            // material info. A fully metallic surface has almost no diffuse response and only
            // reflects light via an environment map, which this scene doesn't provide — under
            // plain ambient/directional lighting it renders near-black except for a thin specular
            // streak wherever the light direction happens to catch the camera, i.e. visible from
            // only one direction. Treating meshes as non-metallic makes them respond to the
            // scene's lights the same ordinary diffuse way the fallback sphere's Lambert material
            // does.
            material.metalness = 0;
          }});
        }});
        group.add(gltf.scene);
      }},
      undefined,
      (error) => {{ console.error("Failed to load mesh for node", node.id, error); }}
    );
    return group;
  }});

  function spin(now) {{
    const deltaSeconds = (now - lastFrameTime) / 1000;
    lastFrameTime = now;
    nodeGroupsById.forEach((group) => {{ group.rotation.y += radiansPerSecond * deltaSeconds; }});
    requestAnimationFrame(spin);
  }}
  requestAnimationFrame(spin);
}})();
"""

    def _node_mesh(self, node_index: int) -> Optional[trimesh.Trimesh]:
        """
        :param node_index: The rustworkx index of the node.
        :return: The node's combined visual mesh, or ``None`` if it has no visual geometry.
        """
        payload = self.graph[node_index]
        if not isinstance(payload, Body) or not payload.visual:
            return None
        return payload.visual.combined_mesh

    def _mesh_glb(self, node_index: int) -> Optional[bytes]:
        """
        :param node_index: The rustworkx index of the node.
        :return: The node's mesh exported as glTF-binary (GLB) bytes, cached after the first
            export, or ``None`` if it has no visual geometry.
        """
        if node_index in self._mesh_glb_cache:
            return self._mesh_glb_cache[node_index]
        with self._mesh_export_lock:
            if node_index not in self._mesh_glb_cache:
                mesh = self._node_mesh(node_index)
                self._mesh_glb_cache[node_index] = (
                    None if mesh is None else mesh.export(file_type="glb")
                )
            return self._mesh_glb_cache[node_index]
