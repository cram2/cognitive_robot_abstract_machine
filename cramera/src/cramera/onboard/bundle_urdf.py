"""
Make a URDF (or xacro) self-contained for the web viewer.

Resolves every mesh reference (package://, file://, absolute or relative),
copies the meshes plus their side assets (textures for .dae, .mtl + textures
for .obj) into ``<out>/meshes/...``, rewrites the references to those relative
paths, and writes ``<out>/<name>.urdf``. The result loads in the browser with
no ROS installed.

Standalone use::

    python -m cramera.onboard.bundle_urdf path/or/package://... \
        --name apartment --out ~/.cramera/scenes/my_scene

:mod:`cramera.onboard.demo` also calls :func:`bundle_urdf` directly, feeding
it the exact uri->path resolutions recorded while the demo ran.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
)

from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.exceptions import ParsingError

from cramera import paths
from cramera.logging_setup import get_logger
from cramera.mesh_format import MeshFormat
from cramera.robot_parts import model_identity

logger = get_logger(__name__)

MISSING_ASSETS_LOGGED = 20
"""
How many unresolved assets ``main`` lists before truncating.
"""


# %% reference resolution
@dataclass(frozen=True)
class MeshReference:
    """
    One ``filename="..."`` reference as a URDF writes it, and where it resolves to.

    A reference may be a ``package://`` URI, a ``file://`` one, an absolute path or one
    relative to the URDF — resolving it is this class's job, and so is deciding where
    its file lands inside the bundle.
    """

    PACKAGE_SCHEME: ClassVar[str] = "package://"

    FILE_SCHEME: ClassVar[str] = "file://"

    LOCAL_MESH_DIRECTORY: ClassVar[str] = "_local"
    """
    Directory bundled meshes land in when their reference names no ROS package.
    """

    uri: str
    """
    The reference exactly as written in the URDF.
    """

    def resolve(
        self,
        hints: Optional[Dict[str, str]] = None,
        base_directory: Optional[str] = None,
    ) -> Optional[str]:
        """
        The existing file this reference points at, or None if nothing matched.

        :param hints: Resolutions recorded while a demo ran, which win over any search.
        :param base_directory: Directory a relative reference is resolved against.
        """
        if hints and self.uri in hints:
            return hints[self.uri]
        if self.uri.startswith(self.PACKAGE_SCHEME):
            return self._resolved_package_path()
        if self.uri.startswith(self.FILE_SCHEME):
            path = self.uri[len(self.FILE_SCHEME) :]
            return path if os.path.isfile(path) else None
        if os.path.isabs(self.uri):
            return self.uri if os.path.isfile(self.uri) else None
        if base_directory:
            path = os.path.join(base_directory, self.uri)
            return path if os.path.isfile(path) else None
        return None

    def bundled_relative_path(self) -> str:
        """
        Where this reference's file lands inside ``<out>/meshes/``.

        Package references keep their package directory so same-named meshes from
        different packages cannot collide; everything else is flattened.
        """
        if self.uri.startswith(self.PACKAGE_SCHEME):
            package, _, relative_path = self.uri[len(self.PACKAGE_SCHEME) :].partition(
                "/"
            )
            return os.path.join(package, relative_path)
        name = (
            self.uri[len(self.FILE_SCHEME) :]
            if self.uri.startswith(self.FILE_SCHEME)
            else self.uri
        )
        return os.path.join(self.LOCAL_MESH_DIRECTORY, os.path.basename(name))

    def _resolved_package_path(self) -> Optional[str]:
        """
        Resolve a ``package://`` URI via :class:`PackageUriResolver`.

        This module has to work on a machine with no ROS installed at all, which is the
        whole point of bundling - :class:`PackageUriResolver`'s default locator chain
        already covers that case (an ament index, ``ROS_PACKAGE_PATH``, and a plain
        filesystem search of common install prefixes), so failure to resolve is
        reported rather than raised.
        """
        try:
            resolved = PackageUriResolver().resolve(self.uri)
        except (ParsingError, OSError) as error:
            logger.debug(
                "the CRAM package resolver could not resolve %s: %s", self.uri, error
            )
            return None
        return resolved if os.path.isfile(resolved) else None


# %% copying assets into the bundle
@dataclass
class BundledAssets:
    """
    The files copied into a bundle, and the references that resolved to no file.

    Owns the already-copied memo, so a mesh referenced by several links, and a texture
    shared by several meshes, are each copied exactly once.
    """

    UNRESOLVED_REFERENCE: ClassVar[str] = "<unresolved>"
    """
    Stands in for a reference the bundler could not resolve to any file.
    """

    TEXTURE_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r"[\w./\-]+\.(?:png|jpg|jpeg|tga|tif)", re.IGNORECASE
    )
    """
    Side assets a mesh file itself references.
    """

    MATERIAL_LIBRARY_PATTERN: ClassVar[Pattern[str]] = re.compile(r"mtllib\s+(.+)")

    TEXTURE_MAP_PATTERN: ClassVar[Pattern[str]] = re.compile(r"map_\w+\s+(.+)")

    copied: Dict[str, str] = field(default_factory=dict)
    """
    Source path to the path it was copied to inside the bundle.
    """

    missing: List[str] = field(default_factory=list)
    """
    References that could not be resolved to any file.
    """

    bundle_root: Optional[str] = None
    """
    Directory nothing may be written outside of, or None to allow any destination.
    """

    def _is_within_bundle(self, path: str) -> bool:
        """
        Whether a destination stays inside :attr:`bundle_root`.

        :param path: The destination path to test.
        """
        if self.bundle_root is None:
            return True
        root = os.path.abspath(self.bundle_root)
        return os.path.commonpath([os.path.abspath(path), root]) == root

    @property
    def mesh_suffixes(self) -> List[str]:
        """
        Sorted, deduplicated file suffixes of everything copied.
        """
        return sorted({os.path.splitext(path)[1].lower() for path in self.copied})

    def copy(self, source: Optional[str], destination: str) -> bool:
        """
        Copy one asset into the bundle, at most once.

        :param source: The resolved path, or None when the reference could not be
            resolved.
        :param destination: Where the asset belongs inside the bundle.
        :return: Whether the asset is present in the bundle afterwards.
        """
        if source in self.copied:
            return True
        if not source or not os.path.isfile(source):
            self.missing.append(source or self.UNRESOLVED_REFERENCE)
            return False
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        self.copied[source] = destination
        return True

    def copy_side_assets(self, source_mesh: str, bundled_mesh: str) -> None:
        """
        Copy the textures a ``.dae`` references, or the ``.mtl`` plus its textures for
        an ``.obj``.

        :param source_mesh: Path of the resolved source mesh.
        :param bundled_mesh: Path the mesh was copied to inside the bundle.
        """
        if not os.path.isfile(source_mesh):
            return
        source_directory = os.path.dirname(source_mesh)
        bundled_directory = os.path.dirname(bundled_mesh)
        mesh_text = Path(source_mesh).read_bytes().decode("utf-8", "replace")
        mesh_format = MeshFormat.of_path(source_mesh)
        if mesh_format is MeshFormat.DAE:
            references = set(self.TEXTURE_PATTERN.findall(mesh_text))
        elif mesh_format is MeshFormat.OBJ:
            references = self._object_side_references(
                mesh_text, source_directory, bundled_directory
            )
        else:
            return
        for reference in references:
            relative_reference = reference.strip()
            source = os.path.normpath(
                os.path.join(source_directory, relative_reference)
            )
            destination = os.path.normpath(
                os.path.join(bundled_directory, relative_reference)
            )
            # the reference is mirrored at the same relative place next to the bundled
            # mesh, so a parent-relative one (``../materials/…``, the Gazebo model
            # layout) resolves in the browser exactly as it did on disk
            if os.path.isfile(source) and self._is_within_bundle(destination):
                self.copy(source, destination)

    def _object_side_references(
        self, mesh_text: str, source_directory: str, bundled_directory: str
    ) -> Set[str]:
        """
        The material libraries an ``.obj`` names, copied on the way, plus the textures
        those libraries name.

        :param mesh_text: The decoded contents of the ``.obj``.
        :param source_directory: Directory the source mesh lives in.
        :param bundled_directory: Directory the mesh was copied to inside the bundle.
        """
        references = {
            material_library.strip()
            for material_library in self.MATERIAL_LIBRARY_PATTERN.findall(mesh_text)
        }
        for material_library in list(references):
            material_source = os.path.join(source_directory, material_library)
            if not os.path.isfile(material_source):
                continue
            self.copy(
                material_source, os.path.join(bundled_directory, material_library)
            )
            material_text = (
                Path(material_source).read_bytes().decode("utf-8", "replace")
            )
            references |= {
                texture.strip()
                for texture in self.TEXTURE_MAP_PATTERN.findall(material_text)
            }
        return references


# %% bundling
@dataclass
class BundleReport:
    """
    What :func:`bundle_urdf` wrote for one URDF or xacro model.
    """

    FIXED_JOINT_TYPE: ClassVar[str] = "fixed"
    """
    The one URDF joint type that cannot move.
    """

    MESH_REFERENCE_PATTERN: ClassVar[Pattern[str]] = re.compile(r'filename="([^"]+)"')
    """
    What the bundler reads out of a URDF.
    """

    LINK_PATTERN: ClassVar[Pattern[str]] = re.compile(r'<link\s+name="([^"]+)"')

    JOINT_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r'<joint\s+name="([^"]+)"\s+type="([^"]+)"'
    )

    name: str
    """
    Output model name, as passed to :func:`bundle_urdf`.
    """

    urdf: str
    """
    Path of the written, reference-rewritten URDF.
    """

    source: str
    """
    Resolved path of the URDF/xacro the bundle was built from.
    """

    links: List[str]
    """
    Names of every ``<link>`` in the written URDF, in document order.
    """

    joints: List[str]
    """
    Names of every ``<joint>`` in the written URDF, in document order.
    """

    movable_joints: List[str]
    """
    Names of the joints among :attr:`joints` whose type is not ``fixed``.
    """

    meshes_copied: int
    """
    Count of distinct mesh files copied into the bundle.
    """

    mesh_suffixes: List[str]
    """
    Sorted, deduplicated file extensions of the copied meshes.
    """

    references_rewritten: int
    """
    Count of mesh references rewritten to their bundled, relative path.
    """

    missing: List[str]
    """
    References that could not be resolved to any file.
    """

    @classmethod
    def of_source(
        cls,
        source: str,
        name: str,
        output_directory: str,
        hints: Optional[Dict[str, str]] = None,
    ) -> "BundleReport":
        """
        Bundle one URDF or xacro with every mesh it references.

        :param source: Path or ``package://`` URI of the URDF/xacro to bundle.
        :param name: Output model name, used for ``<output_directory>/<name>.urdf``.
        :param output_directory: Directory the URDF and its ``meshes/`` tree are written to.
        :param hints: Resolutions recorded while a demo ran.
        :return: A report of what was written, including any unresolved references.
        :raises FileNotFoundError: If the source itself cannot be found.
        """
        source_path = MeshReference(source).resolve(hints=hints) or source
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                "URDF source not found: %s (from %s)" % (source_path, source)
            )
        if source_path.endswith(".xacro"):
            # expanded in-process, so no ROS installation is needed to bundle
            urdf_text = URDFParser.from_xacro(source_path).urdf
        else:
            urdf_text = Path(source_path).read_text(encoding="utf-8", errors="replace")
        base_directory = os.path.dirname(source_path)

        os.makedirs(output_directory, exist_ok=True)
        assets = BundledAssets()
        rewritten = 0
        for reference in sorted(set(cls.MESH_REFERENCE_PATTERN.findall(urdf_text))):
            if MeshFormat.of_path(reference) is None:
                continue  # plugins (.so) and other non-geometry references
            mesh_reference = MeshReference(reference)
            resolved = mesh_reference.resolve(
                hints=hints, base_directory=base_directory
            )
            relative_path = mesh_reference.bundled_relative_path()
            bundled = os.path.join(output_directory, "meshes", relative_path)
            if assets.copy(resolved, bundled):
                assets.copy_side_assets(resolved, bundled)
            urdf_text = urdf_text.replace(
                '"%s"' % reference,
                '"meshes/%s"' % relative_path.replace(os.sep, "/"),
            )
            rewritten += 1

        urdf_out = os.path.join(output_directory, "%s.urdf" % name)
        Path(urdf_out).write_text(urdf_text, encoding="utf-8")
        links = cls.LINK_PATTERN.findall(urdf_text)
        joints = cls.JOINT_PATTERN.findall(urdf_text)
        return cls(
            name=name,
            urdf=urdf_out,
            source=source_path,
            links=links,
            joints=[joint_name for joint_name, _ in joints],
            movable_joints=[
                joint_name
                for joint_name, joint_type in joints
                if joint_type != cls.FIXED_JOINT_TYPE
            ],
            meshes_copied=len(assets.copied),
            mesh_suffixes=assets.mesh_suffixes,
            references_rewritten=rewritten,
            missing=assets.missing,
        )


# %% a bundled model within a composed world
@dataclass
class BundledModel:
    """
    One bundled model source, as the ``models`` list of ``scene.json`` records it.
    """

    name: str
    """
    Model name, which is also its URDF's basename inside the bundle.
    """

    prefix: str
    """
    The model's world-name prefix in the composed world, or ``""`` when none was found.
    """

    is_robot: bool
    """
    Whether this model is the recorded robot rather than an environment model.
    """

    report: BundleReport
    """
    What the bundler wrote for this model.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the viewer's model list expects.
        """
        return {
            "name": self.name,
            "urdf": "%s.urdf" % self.name,
            "prefix": self.prefix,
            "robot": self.is_robot,
            "links": len(self.report.links),
            "movableJoints": self.report.movable_joints,
        }


def bundle_model(
    source: str,
    bundler: Callable[..., BundleReport],
    world_body_names: List[str],
    base_body: Optional[str],
    output_directory: str,
    probe_link_count: int,
    hints: Optional[Dict[str, str]] = None,
) -> BundledModel:
    """
    Bundle one model source and turn its report into a ``models`` scene entry.

    Shared by onboarding (bundling a finished recording to disk) and live model
    serving (bundling a running demo's *current* world to disk) — the two differ only
    in where ``world_body_names``/``base_body``/``hints`` come from.

    :param source: Path or URI of the model's source file.
    :param bundler: Bundles the source into ``output_directory``
        (:meth:`BundleReport.of_source`, or one of :class:`~cramera.onboard.
        bundle_world.BundledWorld`'s Gazebo/MJCF equivalents).
    :param world_body_names: Every body name in the composed world, used to find the
        model's prefix.
    :param base_body: The robot's base link name, used to tell a robot model apart
        from an environment model, or None when no robot is bound.
    :param output_directory: Directory the model's URDF and meshes are written into.
    :param probe_link_count: How many of the model's first links to check for a
        prefix (see :func:`~cramera.robot_parts.model_identity`).
    :param hints: Resolutions recorded while a demo ran, which win over any search.
    :return: The model's ``models`` scene entry.
    """
    base_name = os.path.splitext(os.path.basename(source))[0]
    report = bundler(source, base_name, output_directory, hints=hints)
    prefix, is_robot = model_identity(
        links=report.links,
        world_body_names=world_body_names,
        base_body=base_body,
        probe_link_count=probe_link_count,
    )
    logger.info(
        "bundled %-28s prefix=%-12s robot=%s meshes=%d missing=%d",
        base_name,
        prefix or "-",
        is_robot,
        report.meshes_copied,
        len(report.missing),
    )
    return BundledModel(name=base_name, prefix=prefix, is_robot=is_robot, report=report)


def main() -> None:
    """
    Bundle one URDF/xacro from the command line.

    Exits with status 2 when any referenced asset could not be resolved.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="URDF/xacro path or package:// URI")
    parser.add_argument("--name", help="output model name (default: source basename)")
    parser.add_argument(
        "--out",
        default=str(paths.scenes_directory()),
        help="output directory (default: CRAMERA_SCENES or ~/.cramera/scenes)",
    )
    arguments = parser.parse_args()
    name = arguments.name or os.path.splitext(os.path.basename(arguments.source))[0]
    report = BundleReport.of_source(arguments.source, name, arguments.out)
    logger.info(
        "wrote %s  (%d links, %d joints, %d meshes %s)",
        report.urdf,
        len(report.links),
        len(report.joints),
        report.meshes_copied,
        report.mesh_suffixes,
    )
    if report.missing:
        logger.warning("missing %d assets:", len(report.missing))
        for missing_asset in report.missing[:MISSING_ASSETS_LOGGED]:
            logger.warning("   %s", missing_asset)
        sys.exit(2)


if __name__ == "__main__":
    main()
