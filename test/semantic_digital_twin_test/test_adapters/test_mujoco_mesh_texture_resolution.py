"""
Tests for :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoMeshConverter._resolve_texture_file_path`.

Some meshes (for example RoboCasa's "clear glass" oven/microwave door panes) carry a
programmatically generated ``PIL.Image.Image`` with no backing file on disk at all, rather
than one loaded from a texture file; ``_resolve_texture_file_path`` must recognise that case
and return ``None`` instead of raising.
"""

from dataclasses import dataclass, field

from typing_extensions import Dict, Optional

from semantic_digital_twin.adapters.multi_sim import MujocoMeshConverter


@dataclass
class FakeImageWithFilename:
    """Mimics a PIL image opened from a named file."""

    filename: str
    """The path the image was opened from."""


@dataclass
class FakeImageWithoutFilename:
    """Mimics a programmatically generated PIL.Image.Image with no backing file."""

    info: Dict[str, str] = field(default_factory=dict)
    """Arbitrary metadata PIL attaches to the image."""


@dataclass
class FakeMaterial:
    """Mimics a trimesh ``TextureVisuals.material``."""

    name: str
    """The material's name, sometimes itself a texture file path."""

    image: object
    """The material's image, a real or fake PIL image."""


def test_resolves_texture_from_material_name_when_it_is_a_real_file(tmp_path):
    texture_file = tmp_path / "texture.png"
    texture_file.write_bytes(b"fake png data")
    material = FakeMaterial(name=str(texture_file), image=FakeImageWithoutFilename())

    assert MujocoMeshConverter._resolve_texture_file_path(material) == str(texture_file)


def test_resolves_texture_from_image_filename_when_material_name_is_not_a_file(tmp_path):
    texture_file = tmp_path / "texture.png"
    texture_file.write_bytes(b"fake png data")
    material = FakeMaterial(
        name="material_0", image=FakeImageWithFilename(filename=str(texture_file))
    )

    assert MujocoMeshConverter._resolve_texture_file_path(material) == str(texture_file)


def test_resolves_texture_from_image_info_file_path(tmp_path):
    texture_file = tmp_path / "texture.png"
    texture_file.write_bytes(b"fake png data")
    material = FakeMaterial(
        name="material_0",
        image=FakeImageWithoutFilename(info={"file_path": str(texture_file)}),
    )

    assert MujocoMeshConverter._resolve_texture_file_path(material) == str(texture_file)


def test_returns_none_for_a_programmatically_generated_texture_with_no_backing_file():
    material = FakeMaterial(name="material_0", image=FakeImageWithoutFilename())

    assert MujocoMeshConverter._resolve_texture_file_path(material) is None
