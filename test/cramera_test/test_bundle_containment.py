"""
Geometry sources cannot choose writable paths outside an exported bundle.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from xml.etree import ElementTree

import pytest

from cramera.onboard.bundle_urdf import (
    BundledAssets,
    BundleReport,
    InvalidBundlePath,
    MeshReference,
)
from cramera.mesh_format import MeshFormat


# %% source references
@pytest.fixture
def mesh_source(tmp_path: Path) -> Path:
    """
    Provide an existing mesh for independently chosen source references.

    :param tmp_path: Isolated source directory.
    """
    source = tmp_path / "source.stl"
    source.write_bytes(b"solid source\nendsolid source\n")
    return source


@pytest.fixture
def source_document(tmp_path: Path) -> Path:
    """
    Copy the small URDF whose mesh reference each case supplies.

    :param tmp_path: Isolated source directory.
    """
    source = tmp_path / "source.urdf"
    shutil.copyfile(Path(__file__).parent / "dataset" / "asset_mesh.urdf", source)
    return source


@pytest.mark.parametrize(
    "reference",
    [
        "package://demo/../../../outside.stl",
        "package://demo//outside.stl",
        "package:///outside.stl",
        "package://../outside.stl",
        "package://demo/%2e%2e/outside.stl",
        "package://demo/meshes\\..\\outside.stl",
    ],
)
def test_invalid_package_layout_is_rejected(reference: str) -> None:
    """
    Package paths remain relative within their own package directory.

    :param reference: A package reference with an unsafe path component.
    """
    with pytest.raises(InvalidBundlePath):
        MeshReference(reference).bundled_relative_path()


def test_source_bundle_rejects_package_traversal(
    source_document: Path, mesh_source: Path, tmp_path: Path
) -> None:
    """
    A resolved hint cannot turn package traversal into an external write.

    :param source_document: URDF source containing one mesh.
    :param mesh_source: Existing mesh used as the resolved URI target.
    :param tmp_path: Isolated output and outside paths.
    """
    reference = "package://demo/../../../outside.stl"
    tree = ElementTree.parse(source_document)
    tree.find(".//mesh").set("filename", reference)
    tree.write(source_document)
    with pytest.raises(InvalidBundlePath):
        BundleReport.of_source(
            str(source_document),
            "demo",
            str(tmp_path / "bundle"),
            {reference: str(mesh_source)},
        )
    assert not (tmp_path / "outside.stl").exists()


@pytest.mark.parametrize("link_kind", ["directory", "file"])
def test_copies_do_not_follow_output_symlinks(
    mesh_source: Path, tmp_path: Path, link_kind: str
) -> None:
    """
    Symlink targets outside the bundle remain untouched.

    :param mesh_source: Existing mesh to copy.
    :param tmp_path: Isolated bundle and outside directory.
    :param link_kind: The symlink component traversed by the requested copy.
    """
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside"
    bundle.mkdir()
    outside.mkdir()
    target = outside / "source.stl"
    target.write_bytes(b"unchanged")
    if link_kind == "directory":
        (bundle / "meshes").symlink_to(outside, target_is_directory=True)
        destination = bundle / "meshes" / target.name
    else:
        destination = bundle / target.name
        destination.symlink_to(target)
    assets = BundledAssets(bundle_root=str(bundle))
    assert assets.copy(str(mesh_source), str(destination)) is False
    assert target.read_bytes() == b"unchanged"


def test_source_bundle_keeps_material_traversal_inside_output(
    source_document: Path, tmp_path: Path
) -> None:
    """
    OBJ material declarations cannot overwrite a file above the bundle.

    :param source_document: URDF source containing one mesh.
    :param tmp_path: Isolated source, output and sentinel paths.
    """
    source_directory = tmp_path / "source" / "deep" / "nested" / "meshes"
    source_directory.mkdir(parents=True)
    material = tmp_path / "source" / "outside.mtl"
    material.write_bytes(
        (Path(__file__).parent / "dataset" / "material.mtl").read_bytes()
    )
    mesh = source_directory / "object.obj"
    mesh.write_text(
        (Path(__file__).parent / "dataset" / "material.obj")
        .read_text()
        .replace("surface.mtl", "../../../outside.mtl")
    )
    tree = ElementTree.parse(source_document)
    tree.find(".//mesh").set("filename", str(mesh))
    tree.write(source_document)
    output = tmp_path / "bundle"
    sentinel = tmp_path / "outside.mtl"
    sentinel.write_bytes(b"unchanged")
    BundleReport.of_source(str(source_document), "demo", str(output))
    assert sentinel.read_bytes() == b"unchanged"


def test_material_textures_resolve_relative_to_their_library(tmp_path: Path) -> None:
    """
    A material in a sibling directory keeps its own texture references.

    :param tmp_path: Isolated source model and bundle directories.
    """
    source = tmp_path / "source"
    meshes = source / "meshes"
    materials = source / "materials"
    meshes.mkdir(parents=True)
    materials.mkdir()
    fixtures = Path(__file__).parent / "dataset"
    mesh = meshes / "object.obj"
    mesh.write_text(
        (fixtures / "material.obj")
        .read_text()
        .replace("surface.mtl", "../materials/surface.mtl")
    )
    shutil.copyfile(fixtures / "material.mtl", materials / "surface.mtl")
    texture = materials / "color.png"
    texture.write_bytes(b"texture")
    output = tmp_path / "bundle"
    bundled_mesh = output / "meshes" / mesh.name
    assets = BundledAssets(bundle_root=str(output))
    assert assets.copy(str(mesh), str(bundled_mesh))
    assets.copy_side_assets(str(mesh), str(bundled_mesh))
    assert (output / "materials" / texture.name).read_bytes() == texture.read_bytes()


@pytest.mark.parametrize("reference_kind", ["absolute", "file", "relative"])
def test_valid_local_mesh_references_stay_inside_bundle(
    source_document: Path, mesh_source: Path, tmp_path: Path, reference_kind: str
) -> None:
    """
    Supported local URI forms preserve the mesh contents under the output root.

    :param source_document: URDF source containing one mesh.
    :param mesh_source: Existing mesh to bundle.
    :param tmp_path: Isolated bundle directory.
    :param reference_kind: Local reference syntax under test.
    """
    reference = {
        "absolute": str(mesh_source),
        "file": mesh_source.as_uri(),
        "relative": mesh_source.name,
    }[reference_kind]
    tree = ElementTree.parse(source_document)
    tree.find(".//mesh").set("filename", reference)
    tree.write(source_document)
    output = tmp_path / "bundle"
    report = BundleReport.of_source(str(source_document), "demo", str(output))
    copied = output / "meshes" / "_local" / mesh_source.name
    assert copied.read_bytes() == mesh_source.read_bytes()
    assert report.missing == []


# %% document output
def test_valid_xml_attribute_forms_preserve_meshes_and_structure(
    mesh_source: Path, tmp_path: Path
) -> None:
    """
    Single XML quotes and reordered joint attributes retain all model metadata.

    :param mesh_source: Existing mesh referenced by the XML fixture.
    :param tmp_path: Isolated source and output directory.
    """
    source = tmp_path / "attributes.urdf"
    shutil.copyfile(Path(__file__).parent / "dataset" / "asset_attributes.urdf", source)
    output = tmp_path / "bundle"
    report = BundleReport.of_source(str(source), "model", str(output))
    description = ElementTree.parse(report.urdf)
    mesh_path = description.find(".//mesh").get("filename")
    assert (output / mesh_path).read_bytes() == mesh_source.read_bytes()
    assert report.links == ["base", "mesh"]
    assert report.joints == ["mount"]


def test_two_references_to_one_mesh_use_its_copied_destination(
    source_document: Path, mesh_source: Path, tmp_path: Path
) -> None:
    """
    Every rewritten alias resolves to the single memoized mesh copy.

    :param source_document: Source URDF with one mesh element to extend.
    :param mesh_source: One file addressed by two source URIs.
    :param tmp_path: Isolated output directory.
    """
    tree = ElementTree.parse(source_document)
    references = ["package://first/one.stl", "package://second/two.stl"]
    mesh = tree.find(".//mesh")
    mesh.set("filename", references[0])
    visual = ElementTree.SubElement(tree.find("link"), "visual")
    geometry = ElementTree.SubElement(visual, "geometry")
    ElementTree.SubElement(geometry, "mesh", {"filename": references[1]})
    tree.write(source_document)
    output = tmp_path / "bundle"
    report = BundleReport.of_source(
        str(source_document),
        "model",
        str(output),
        {reference: str(mesh_source) for reference in references},
    )
    rewritten = [
        mesh.get("filename") for mesh in ElementTree.parse(report.urdf).iter("mesh")
    ]
    assert len(set(rewritten)) == 1
    assert (output / rewritten[0]).read_bytes() == mesh_source.read_bytes()
    assert report.meshes_copied == 1


@pytest.mark.parametrize("mesh_format", [MeshFormat.OBJ, MeshFormat.DAE])
def test_shared_side_assets_exist_at_each_mesh_reference(
    source_document: Path, tmp_path: Path, mesh_format: MeshFormat
) -> None:
    """
    Preserve relative material and texture paths for meshes in separate directories.

    :param source_document: Source URDF extended with two distinct mesh references.
    :param tmp_path: Isolated source files and output directory.
    :param mesh_format: Format whose side-asset declarations remain unchanged.
    """
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    fixtures = Path(__file__).parent / "dataset"
    material = source_directory / "surface.mtl"
    shutil.copyfile(fixtures / "material.mtl", material)
    texture = source_directory / "color.png"
    texture.write_bytes(b"shared texture")
    tree = ElementTree.parse(source_document)
    hints = {}
    for directory in ("first", "second"):
        source_mesh = source_directory / f"{directory}{mesh_format.value}"
        shutil.copyfile(fixtures / f"material{mesh_format.value}", source_mesh)
        reference = f"package://demo/{directory}/{source_mesh.name}"
        hints[reference] = str(source_mesh)
        if not len(hints) == 1:
            visual = ElementTree.SubElement(tree.find("link"), "visual")
            geometry = ElementTree.SubElement(visual, "geometry")
            ElementTree.SubElement(geometry, "mesh", {"filename": reference})
        else:
            tree.find(".//mesh").set("filename", reference)
    tree.write(source_document)
    output = tmp_path / "bundle"

    report = BundleReport.of_source(str(source_document), "model", str(output), hints)

    for mesh in ElementTree.parse(report.urdf).iter("mesh"):
        directory = (output / mesh.get("filename")).parent
        assert (directory / texture.name).read_bytes() == texture.read_bytes()
        if mesh_format is MeshFormat.OBJ:
            assert (directory / material.name).read_bytes() == material.read_bytes()
    assert report.missing == []


@pytest.mark.parametrize("absolute", [False, True])
def test_model_names_cannot_escape_the_bundle(
    source_document: Path, tmp_path: Path, absolute: bool
) -> None:
    """
    The model document obeys the same destination boundary as its assets.

    :param source_document: URDF source containing one mesh.
    :param tmp_path: Isolated output directory.
    :param absolute: Whether the invalid model name is absolute or parent-relative.
    """
    with pytest.raises(InvalidBundlePath):
        BundleReport.of_source(
            str(source_document),
            str(tmp_path / "outside") if absolute else "../outside",
            str(tmp_path / "bundle"),
        )
