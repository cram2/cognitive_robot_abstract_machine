from PIL import Image
import trimesh

from semantic_digital_twin.world_description.geometry import Mesh


def test_mesh_loader_preserves_embedded_texture(tmp_path):
    texture_path = tmp_path / "texture.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(texture_path)

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=[[0.0, 0.0]] * len(mesh.vertices), image=Image.open(texture_path)
    )
    mesh_path = tmp_path / "textured.obj"
    mesh.export(mesh_path)

    loaded = Mesh.from_file(str(mesh_path)).mesh

    assert loaded.visual.kind == trimesh.visual.texture.TextureVisuals().kind


def test_mesh_loader_applies_color_to_plain_mesh(tmp_path):
    mesh = trimesh.creation.box()
    mesh_path = tmp_path / "plain.obj"
    mesh.export(mesh_path)

    loaded = Mesh.from_file(str(mesh_path)).mesh

    assert loaded.visual.kind != trimesh.visual.texture.TextureVisuals().kind
    assert loaded.visual.vertex_colors is not None
