"""
Make a URDF (or xacro) self-contained for the web viewer.

Resolves every mesh reference (package://, file://, absolute or relative),
copies the meshes plus their side assets (textures for .dae, .mtl + textures
for .obj) into ``<out>/meshes/...``, rewrites the references to those relative
paths, and writes ``<out>/<name>.urdf``. The result loads in the browser with
no ROS installed.

Standalone use::

    python -m cram_viz.onboard.bundle_urdf path/or/package://... \
        --name apartment --out ~/.cram_viz/scenes/my_scene

:mod:`cram_viz.onboard.demo` also calls :func:`bundle_urdf` directly, feeding
it the exact uri->path resolutions recorded while the demo ran.
"""

import argparse
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Dict, List, Optional

from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.exceptions import ParsingError

from cram_viz import get_logger, paths

logger = get_logger(__name__)

#: mesh formats the web viewer's loaders can read
MESH_SUFFIXES = (".dae", ".stl", ".obj")

#: how many unresolved assets ``main`` lists before truncating
MISSING_ASSETS_LOGGED = 20

#: the one URDF joint type that cannot move
FIXED_JOINT_TYPE = "fixed"

#: stands in for a reference the bundler could not resolve to any file
UNRESOLVED_REFERENCE = "<unresolved>"

#: what the bundler reads out of a URDF
MESH_REFERENCE_PATTERN = re.compile(r'filename="([^"]+)"')
LINK_PATTERN = re.compile(r'<link\s+name="([^"]+)"')
JOINT_PATTERN = re.compile(r'<joint\s+name="([^"]+)"\s+type="([^"]+)"')

#: side assets a mesh file itself references
TEXTURE_PATTERN = re.compile(r"[\w./\-]+\.(?:png|jpg|jpeg|tga|tif)", re.IGNORECASE)
MATERIAL_LIBRARY_PATTERN = re.compile(r"mtllib\s+(.+)")
TEXTURE_MAP_PATTERN = re.compile(r"map_\w+\s+(.+)")


PACKAGE_SCHEME = "package://"
FILE_SCHEME = "file://"

#: directory bundled meshes land in when their reference names no ROS package
LOCAL_MESH_DIRECTORY = "_local"


# %% reference resolution
def _resolve_package_uri(uri: str) -> Optional[str]:
    """
    Resolve a ``package://`` URI via :class:`PackageUriResolver`.

    This module has to work on a machine with no ROS installed at all, which is the
    whole point of bundling - :class:`PackageUriResolver`'s default locator chain
    already covers that case (an ament index, ``ROS_PACKAGE_PATH``, and a plain
    filesystem search of common install prefixes), so failure to resolve is reported
    rather than raised.

    :param uri: The ``package://`` URI to resolve.
    """
    try:
        resolved = PackageUriResolver().resolve(uri)
    except (ParsingError, OSError) as error:
        logger.debug("the CRAM package resolver could not resolve %s: %s", uri, error)
        return None
    return resolved if os.path.isfile(resolved) else None


def resolve_uri(
    uri: str, hints: Optional[Dict[str, str]] = None, base_dir: Optional[str] = None
) -> Optional[str]:
    """
    Resolve a mesh or URDF reference to an existing absolute file path.

    :param uri: The reference as written in the URDF.
    :param hints: Resolutions recorded while a demo ran, which win over any search.
    :param base_dir: Directory a relative reference is resolved against.
    :return: The file's path, or None if nothing matched.
    """
    if hints and uri in hints:
        return hints[uri]
    if uri.startswith(PACKAGE_SCHEME):
        return _resolve_package_uri(uri)
    if uri.startswith(FILE_SCHEME):
        path = uri[len(FILE_SCHEME) :]
        return path if os.path.isfile(path) else None
    if os.path.isabs(uri):
        return uri if os.path.isfile(uri) else None
    if base_dir:
        path = os.path.join(base_dir, uri)
        return path if os.path.isfile(path) else None
    return None


def _bundled_relative_path(uri: str) -> str:
    """
    Where a reference lands inside ``<out>/meshes/``.

    Package references keep their package directory so same-named meshes from different
    packages cannot collide; everything else is flattened.

    :param uri: The reference as written in the URDF.
    """
    if uri.startswith(PACKAGE_SCHEME):
        package, _, relative_path = uri[len(PACKAGE_SCHEME) :].partition("/")
        return os.path.join(package, relative_path)
    name = uri[len(FILE_SCHEME) :] if uri.startswith(FILE_SCHEME) else uri
    return os.path.join(LOCAL_MESH_DIRECTORY, os.path.basename(name))


# %% copying assets into the bundle
def _copy_file(
    source: Optional[str],
    destination: str,
    copied: Dict[str, str],
    missing: List[str],
) -> bool:
    """
    Copy one asset into the bundle, at most once.

    :param source: The resolved path, or None when the reference could not be resolved.
    :param destination: Where the asset belongs inside the bundle.
    :param copied: Source path to bundled path, doubling as the already-copied memo.
    :param missing: Collects references that could not be copied.
    :return: Whether the asset is present in the bundle afterwards.
    """
    if source in copied:
        return True
    if not source or not os.path.isfile(source):
        missing.append(source or UNRESOLVED_REFERENCE)
        return False
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    copied[source] = destination
    return True


def _copy_side_assets(
    source_mesh: str, bundled_mesh: str, copied: Dict[str, str], missing: List[str]
) -> None:
    """
    Copy the textures a ``.dae`` references, or the ``.mtl`` plus its textures for an
    ``.obj``.

    :param source_mesh: Path of the resolved source mesh.
    :param bundled_mesh: Path the mesh was copied to inside the bundle.
    :param copied: Source path to bundled path, doubling as the already-copied memo.
    :param missing: Collects references that could not be copied.
    """
    source_directory = os.path.dirname(source_mesh)
    bundled_directory = os.path.dirname(bundled_mesh)
    suffix = source_mesh.lower().rsplit(".", 1)[-1]
    if not os.path.isfile(source_mesh):
        return
    mesh_text = Path(source_mesh).read_bytes().decode("utf-8", "replace")
    references = set()
    if suffix == "dae":
        references |= set(TEXTURE_PATTERN.findall(mesh_text))
    elif suffix == "obj":
        for material_library in MATERIAL_LIBRARY_PATTERN.findall(mesh_text):
            references.add(material_library.strip())
        for material_library in list(references):
            material_source = os.path.join(source_directory, material_library)
            if not os.path.isfile(material_source):
                continue
            _copy_file(
                material_source,
                os.path.join(bundled_directory, material_library),
                copied,
                missing,
            )
            material_text = (
                Path(material_source).read_bytes().decode("utf-8", "replace")
            )
            for texture in TEXTURE_MAP_PATTERN.findall(material_text):
                references.add(texture.strip())
    for reference in references:
        relative_reference = reference.strip().lstrip("./")
        source = os.path.join(source_directory, relative_reference)
        if os.path.isfile(source):
            _copy_file(
                source,
                os.path.join(bundled_directory, relative_reference),
                copied,
                missing,
            )


# %% xacro
def xacro_to_urdf_text(path: str) -> str:
    """
    Expand a xacro file to URDF text in-process, without any ROS installed.

    :param path: Path of the xacro (or plain URDF) source file.
    """
    return URDFParser.from_xacro(path).urdf


# %% bundling
@dataclass
class BundleReport:
    """
    What :func:`bundle_urdf` wrote for one URDF or xacro model.
    """

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

    mesh_exts: List[str]
    """
    Sorted, deduplicated file extensions of the copied meshes.
    """

    refs_rewritten: int
    """
    Count of mesh references rewritten to their bundled, relative path.
    """

    missing: List[str]
    """
    References that could not be resolved to any file.
    """


def bundle_urdf(
    source: str, name: str, out_dir: str, hints: Optional[Dict[str, str]] = None
) -> BundleReport:
    """
    Bundle one URDF or xacro with every mesh it references.

    :param source: Path or ``package://`` URI of the URDF/xacro to bundle.
    :param name: Output model name, used for ``<out_dir>/<name>.urdf``.
    :param out_dir: Directory the URDF and its ``meshes/`` tree are written to.
    :param hints: Resolutions recorded while a demo ran.
    :return: A report of what was written, including any unresolved references.
    :raises FileNotFoundError: If the source itself cannot be found.
    """
    source_path = resolve_uri(source, hints=hints) or source
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            "URDF source not found: %s (from %s)" % (source_path, source)
        )
    if source_path.endswith(".xacro"):
        urdf_text = xacro_to_urdf_text(source_path)
    else:
        urdf_text = Path(source_path).read_text(encoding="utf-8", errors="replace")
    base_dir = os.path.dirname(source_path)

    os.makedirs(out_dir, exist_ok=True)
    copied: Dict[str, str] = {}
    missing: List[str] = []
    rewritten = 0
    for reference in sorted(set(MESH_REFERENCE_PATTERN.findall(urdf_text))):
        if not reference.lower().endswith(MESH_SUFFIXES):
            continue  # plugins (.so) and other non-geometry references
        resolved = resolve_uri(reference, hints=hints, base_dir=base_dir)
        relative_path = _bundled_relative_path(reference)
        bundled = os.path.join(out_dir, "meshes", relative_path)
        if _copy_file(resolved, bundled, copied, missing):
            _copy_side_assets(resolved, bundled, copied, missing)
        urdf_text = urdf_text.replace(
            '"%s"' % reference,
            '"meshes/%s"' % relative_path.replace(os.sep, "/"),
        )
        rewritten += 1

    urdf_out = os.path.join(out_dir, "%s.urdf" % name)
    Path(urdf_out).write_text(urdf_text, encoding="utf-8")
    links = LINK_PATTERN.findall(urdf_text)
    joints = JOINT_PATTERN.findall(urdf_text)
    suffixes = sorted({os.path.splitext(path)[1].lower() for path in copied})
    return BundleReport(
        name=name,
        urdf=urdf_out,
        source=source_path,
        links=links,
        joints=[joint_name for joint_name, _ in joints],
        movable_joints=[
            joint_name
            for joint_name, joint_type in joints
            if joint_type != FIXED_JOINT_TYPE
        ],
        meshes_copied=len(copied),
        mesh_exts=suffixes,
        refs_rewritten=rewritten,
        missing=missing,
    )


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
        default=str(paths.scenes_dir()),
        help="output directory (default: CRAM_VIZ_SCENES or ~/.cram_viz/scenes)",
    )
    arguments = parser.parse_args()
    name = arguments.name or os.path.splitext(os.path.basename(arguments.source))[0]
    report = bundle_urdf(arguments.source, name, arguments.out)
    logger.info(
        "wrote %s  (%d links, %d joints, %d meshes %s)",
        report.urdf,
        len(report.links),
        len(report.joints),
        report.meshes_copied,
        report.mesh_exts,
    )
    if report.missing:
        logger.warning("missing %d assets:", len(report.missing))
        for missing_asset in report.missing[:MISSING_ASSETS_LOGGED]:
            logger.warning("   %s", missing_asset)
        sys.exit(2)


if __name__ == "__main__":
    main()
