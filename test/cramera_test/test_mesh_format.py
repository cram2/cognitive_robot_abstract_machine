"""
Tests for recognising the mesh formats the viewer can load.
"""

import pytest

from cramera.mesh_format import MeshFormat


class TestMeshFormatOfPath:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("milk.stl", MeshFormat.STL),
            ("meshes/table.obj", MeshFormat.OBJ),
            ("package://pr2_description/kitchen.dae", MeshFormat.DAE),
            ("MILK.STL", MeshFormat.STL),
        ],
    )
    def test_a_mesh_reference_names_its_format(self, path, expected):
        assert MeshFormat.of_path(path) == expected

    @pytest.mark.parametrize("path", ["libgazebo_plugin.so", "robot.urdf", "milk"])
    def test_a_reference_to_something_else_names_no_format(self, path):
        assert MeshFormat.of_path(path) is None


class TestMeshFormatSuffixes:
    def test_every_member_contributes_its_suffix(self):
        assert MeshFormat.suffixes() == (".stl", ".obj", ".dae")
