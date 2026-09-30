"""
The material library a bundled mesh brings with it.

An OBJ names its ``.mtl`` by the name it had at authoring time, while the bundler files
the mesh under the body's key, so the two names differ and the scene has to record the
one that actually resolves inside the bundle.
"""

from pathlib import Path

from cramera.onboard.bundle_urdf import companion_material_library

OBJ_WITH_A_LIBRARY = "\n".join(
    ["mtllib crate.mtl", "o crate", "v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"]
)
"""
An OBJ as an authoring tool writes it: the library by name, next to the mesh.
"""


def bundle_with(
    tmp_path: Path, obj_name: str, obj_text: str, library: str | None
) -> Path:
    """
    A bundle holding one object mesh, and its material library when it has one.

    :param tmp_path: The directory to build the bundle in.
    :param obj_name: File name the mesh is filed under.
    :param obj_text: The mesh file's content.
    :param library: File name of the material library to write, or None to write none.
    """
    objects = tmp_path / "meshes" / "objects"
    objects.mkdir(parents=True)
    (objects / obj_name).write_text(obj_text)
    if library is not None:
        (objects / library).write_text("newmtl crate\nmap_Kd crate.png\n")
    return tmp_path


class TestCompanionMaterialLibrary:
    def test_the_library_an_obj_names_is_answered_by_its_bundle_path(self, tmp_path):
        """
        The mesh was filed under the body's key -- ``crate.obj.obj`` -- while the
        library it names kept its own name, so the answer is not derived from the
        mesh's.
        """
        root = bundle_with(tmp_path, "crate.obj.obj", OBJ_WITH_A_LIBRARY, "crate.mtl")

        assert (
            companion_material_library(root, "meshes/objects/crate.obj.obj")
            == "meshes/objects/crate.mtl"
        )

    def test_an_obj_naming_a_library_the_bundle_lacks_answers_nothing(self, tmp_path):
        """
        Serving a path that is not there would only make the viewer fetch a 404.
        """
        root = bundle_with(tmp_path, "crate.obj.obj", OBJ_WITH_A_LIBRARY, None)

        assert companion_material_library(root, "meshes/objects/crate.obj.obj") is None

    def test_an_obj_without_a_library_answers_nothing(self, tmp_path):
        root = bundle_with(tmp_path, "crate.obj.obj", "o crate\nv 0 0 0\n", None)

        assert companion_material_library(root, "meshes/objects/crate.obj.obj") is None

    def test_a_mesh_that_is_no_obj_answers_nothing(self, tmp_path):
        """
        Only an OBJ carries a material library; an STL has no materials to look for.
        """
        root = bundle_with(tmp_path, "crate.stl.stl", "solid crate\nendsolid\n", None)

        assert companion_material_library(root, "meshes/objects/crate.stl.stl") is None

    def test_a_mesh_the_bundle_does_not_have_answers_nothing(self, tmp_path):
        assert companion_material_library(tmp_path, "meshes/objects/ghost.obj") is None
