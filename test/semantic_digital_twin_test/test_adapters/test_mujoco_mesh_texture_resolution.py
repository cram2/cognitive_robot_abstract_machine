"""
Tests for :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoMeshConverter._resolve_
texture_file_path`.

Some meshes (for example RoboCasa's "clear glass" oven/microwave door panes) carry a
programmatically generated ``PIL.Image.Image`` with no backing file on disk at all,
rather than one loaded from a texture file; ``_resolve_texture_file_path`` must
recognise that case and return ``None`` instead of raising.
"""

from dataclasses import dataclass

import PIL.Image

from semantic_digital_twin.adapters.multi_sim import MujocoMeshConverter


@dataclass
class FakeMaterial:
    """
    Mimics a trimesh ``TextureVisuals.material``.
    """

    name: str
    """The material's name, sometimes itself a texture file path."""

    image: PIL.Image.Image
    """
    The material's image.
    """


def test_resolves_texture_from_material_name_when_it_is_a_real_file(tmp_path):
    texture_file = tmp_path / "texture.png"
    PIL.Image.new("RGB", (1, 1)).save(texture_file)
    material = FakeMaterial(name=str(texture_file), image=PIL.Image.new("RGB", (1, 1)))

    assert MujocoMeshConverter._resolve_texture_file_path(
        material, str(tmp_path)
    ) == str(texture_file)


def test_resolves_texture_from_image_filename_when_material_name_is_not_a_file(
    tmp_path,
):
    texture_file = tmp_path / "texture.png"
    PIL.Image.new("RGB", (1, 1)).save(texture_file)
    material = FakeMaterial(name="material_0", image=PIL.Image.open(texture_file))

    assert MujocoMeshConverter._resolve_texture_file_path(
        material, str(tmp_path)
    ) == str(texture_file)


def test_resolves_texture_from_image_info_file_path(tmp_path):
    texture_file = tmp_path / "texture.png"
    PIL.Image.new("RGB", (1, 1)).save(texture_file)
    image = PIL.Image.new("RGB", (1, 1))
    image.info["file_path"] = str(texture_file)
    material = FakeMaterial(name="material_0", image=image)

    assert MujocoMeshConverter._resolve_texture_file_path(
        material, str(tmp_path)
    ) == str(texture_file)


def test_returns_none_for_a_programmatically_generated_texture_with_no_backing_file(
    tmp_path,
):
    material = FakeMaterial(name="material_0", image=PIL.Image.new("RGB", (1, 1)))

    assert (
        MujocoMeshConverter._resolve_texture_file_path(material, str(tmp_path)) is None
    )


def test_resolves_texture_named_relative_to_its_mesh(tmp_path):
    """
    Trimesh reports a texture path relative to the mesh that named it, which is only
    meaningful next to that mesh, so resolution starts from the mesh's directory rather
    than the process working directory.

    Without this the texture is silently not found and the geom renders with MuJoCo's
    default gray instead of its texture.
    """
    texture_file = tmp_path / "texture.png"
    PIL.Image.new("RGB", (1, 1)).save(texture_file)
    image = PIL.Image.new("RGB", (1, 1))
    image.info["file_path"] = "texture.png"
    material = FakeMaterial(name="material_0", image=image)

    assert MujocoMeshConverter._resolve_texture_file_path(
        material, str(tmp_path)
    ) == str(texture_file)
