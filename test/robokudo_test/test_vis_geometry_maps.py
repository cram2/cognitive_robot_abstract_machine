import numpy as np
import open3d as o3d
import pytest
from typing_extensions import Type, List, Dict, Union
from robokudo.vis.multiprocessed_o3d_visualizer import (
    Geometry3DMemoryMapFactory,
    SharedMemoryManager,
)


RANDOM_SEED = 42


def get_random_mesh(
    mesh_type: Type, rng: np.random.Generator
) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.TetraMesh]:
    """
    Get a random open3d mesh of the given type.

    Accepted types:

    * MeshBase
    * TriangleMesh
    * TetraMesh

    :param mesh_type: The type of mesh to return.
    :param rng: Random number generator to use.
    :return: A mesh object of the type.
    :raises: ValueError if the mesh type is not supported.
    """
    input_mesh = mesh_type()

    input_mesh.vertices = o3d.utility.Vector3dVector(rng.random((1000, 3)))
    input_mesh.vertex_colors = o3d.utility.Vector3dVector(rng.random((1000, 3)))
    input_mesh.vertex_normals = o3d.utility.Vector3dVector(rng.random((1000, 3)))
    if mesh_type == o3d.geometry.TriangleMesh:
        input_mesh.triangles = o3d.utility.Vector3iVector(rng.random((1000, 3)))
        input_mesh.triangle_normals = o3d.utility.Vector3dVector(rng.random((1000, 3)))
        input_mesh.triangle_uvs = o3d.utility.Vector2dVector(rng.random((1000, 2)))
        input_mesh.triangle_material_ids = o3d.utility.IntVector(
            rng.integers(0, 10, size=1000, dtype=np.int32)
        )
        input_mesh.adjacency_list = [{3, 8, 9}, {0, 4}, {0, 2, 4}]
        input_mesh.textures = [
            o3d.geometry.Image(rng.integers(0, 255, size=(512, 512), dtype=np.uint8)),
            o3d.geometry.Image(rng.integers(0, 255, size=(512, 512), dtype=np.uint8)),
        ]
        return input_mesh
    elif mesh_type == o3d.geometry.TetraMesh:
        input_mesh.tetras = o3d.utility.Vector4iVector(
            rng.integers(0, 255, size=(1000, 4))
        )
        return input_mesh
    elif mesh_type == o3d.geometry.MeshBase:
        return input_mesh
    else:
        raise ValueError(f"Unknown mesh type: {type}")


def assert_mesh_equal(mesh1, mesh2) -> None:
    """
    Assert that the two meshes contain the same mesh data.

    :param mesh1: The first mesh to compare.
    :param mesh2: The second mesh to compare.
    """
    assert type(mesh1) == type(
        mesh2
    ), f"Meshes must be of the same type but are mesh1={type(mesh1)} mesh2={type(mesh2)}"

    assert np.all(np.asarray(mesh2.vertices) == np.asarray(mesh1.vertices))
    assert np.all(np.asarray(mesh2.vertex_colors) == np.asarray(mesh1.vertex_colors))
    assert np.all(np.asarray(mesh2.vertex_normals) == np.asarray(mesh1.vertex_normals))

    if isinstance(mesh1, o3d.geometry.TriangleMesh):
        assert np.all(np.asarray(mesh2.triangles) == np.asarray(mesh1.triangles))
        assert np.all(
            np.asarray(mesh2.triangle_normals) == np.asarray(mesh1.triangle_normals)
        )
        assert np.all(np.asarray(mesh2.triangle_uvs) == np.asarray(mesh1.triangle_uvs))
        assert np.all(
            np.asarray(mesh2.triangle_material_ids)
            == np.asarray(mesh1.triangle_material_ids)
        )
        for j in range(len(mesh1.adjacency_list)):
            assert isinstance(mesh2.adjacency_list[j], set)
            assert np.all(
                np.array(list(mesh2.adjacency_list[j]))
                == np.array(list(mesh1.adjacency_list[j]))
            )
        for j in range(len(mesh1.textures)):
            assert isinstance(mesh2.textures[j], o3d.geometry.Image)
            assert np.all(
                np.asarray(mesh2.textures[j]) == np.asarray(mesh1.textures[j])
            )
    elif isinstance(mesh1, o3d.geometry.HalfEdgeTriangleMesh):
        assert np.all(np.asarray(mesh2.triangles) == np.asarray(mesh1.triangles))
        assert np.all(
            np.asarray(mesh2.triangle_normals) == np.asarray(mesh1.triangle_normals)
        )
        for j in range(len(mesh2.ordered_half_edge_from_vertex)):
            assert isinstance(
                mesh2.ordered_half_edge_from_vertex[j], o3d.utility.IntVector
            )
            assert np.all(
                np.asarray(mesh2.ordered_half_edge_from_vertex[j])
                == np.asarray(mesh1.ordered_half_edge_from_vertex[j])
            )
        for j in range(len(mesh2.half_edges)):
            assert isinstance(mesh2.half_edges[j], o3d.geometry.HalfEdge)
            assert mesh2.half_edges[j].next == mesh1.half_edges[j].next
            assert (
                mesh2.half_edges[j].triangle_index == mesh1.half_edges[j].triangle_index
            )
            assert mesh2.half_edges[j].twin == mesh1.half_edges[j].twin
            assert np.all(
                mesh2.half_edges[j].vertex_indices == mesh1.half_edges[j].vertex_indices
            )
    elif isinstance(mesh1, o3d.geometry.TetraMesh):
        assert np.all(np.asarray(mesh2.tetras) == np.asarray(mesh1.tetras))


class TestVisGeometryMaps(object):
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        """
        Random number generator initialized with the global random seed.
        """
        return np.random.default_rng(seed=RANDOM_SEED)

    def test_point_cloud_maps(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading point clouds with the shared memory manager.
        """
        iterations = 3

        write_manager = SharedMemoryManager.with_shm(
            (1000 * np.dtype(np.float64).itemsize * (3 + 3 + 3 + 9)) * iterations
        )

        inputs_pcds = []
        for _ in range(iterations):
            input_pcd = o3d.geometry.PointCloud()
            input_pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
            input_pcd.colors = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
            input_pcd.estimate_normals()
            input_pcd.covariances = input_pcd.estimate_point_covariances(input_pcd)

            assert input_pcd.has_normals(), "PointCloud should have normals"
            assert input_pcd.has_covariances(), "PointCloud should have covariances"

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "PointCloud", input_pcd
            )
            write_manager.write_geometry(input_pcd, memory_map)

            inputs_pcds.append(input_pcd)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_pcd = inputs_pcds[i]
            output_pcd = geometry_dict["geometry"]

            assert (
                output_pcd != input_pcd
            ), "Output and input point clouds should be different objects"

            assert output_pcd.has_normals(), "PointCloud should have normals"
            assert output_pcd.has_covariances(), "PointCloud should have covariances"

            assert np.all(np.asarray(output_pcd.points) == np.asarray(input_pcd.points))
            assert np.all(np.asarray(output_pcd.colors) == np.asarray(input_pcd.colors))
            assert np.all(
                np.asarray(output_pcd.normals) == np.asarray(input_pcd.normals)
            )
            assert np.all(
                np.asarray(output_pcd.covariances) == np.asarray(input_pcd.covariances)
            )

    def test_mesh_base_maps(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading base meshes with the shared memory manager.
        """
        iterations = 3

        mesh_size = (np.dtype(np.float64).itemsize * (3 + 3 + 3)) * 1000
        write_manager = SharedMemoryManager.with_shm(mesh_size * iterations)

        input_meshes = []
        for _ in range(iterations):
            input_mesh = get_random_mesh(o3d.geometry.MeshBase, rng)

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "MeshBase", input_mesh
            )
            write_manager.write_geometry(input_mesh, memory_map)

            input_meshes.append(input_mesh)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_mesh = input_meshes[i]
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert_mesh_equal(output_mesh, input_mesh)

    def test_triangle_mesh_maps(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading triangle meshes with the shared memory manager.
        """
        iterations = 3

        mesh_size = (
            (np.dtype(np.float64).itemsize * (3 + 3 + 3 + 3 + 2))
            + (np.dtype(np.int32).itemsize * (3 + 1))
            + (np.dtype(np.int64).itemsize * (3 + 2 + 3))
        ) * 1000 + ((np.dtype(np.uint8).itemsize * (512 * 512)) * 2)
        write_manager = SharedMemoryManager.with_shm(mesh_size * iterations)

        input_meshes = []
        for _ in range(iterations):
            input_mesh = get_random_mesh(o3d.geometry.TriangleMesh, rng)

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "TriangleMesh", input_mesh
            )
            write_manager.write_geometry(input_mesh, memory_map)

            input_meshes.append(input_mesh)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_mesh = input_meshes[i]
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert_mesh_equal(output_mesh, input_mesh)

    def test_tetra_mesh_maps(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading base meshes with the shared memory manager.
        """
        iterations = 3

        mesh_size = (
            (np.dtype(np.float64).itemsize * (3 + 3 + 3))
            + (np.dtype(np.int64).itemsize * 4)
        ) * 1000
        write_manager = SharedMemoryManager.with_shm(mesh_size * iterations)

        input_meshes = []
        for _ in range(iterations):
            input_mesh = get_random_mesh(o3d.geometry.TetraMesh, rng)

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "TetraMesh", input_mesh
            )
            write_manager.write_geometry(input_mesh, memory_map)

            input_meshes.append(input_mesh)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_mesh = input_meshes[i]
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert_mesh_equal(output_mesh, input_mesh)

    def test_half_edge_mesh_maps(self) -> None:
        """
        Test writing and reading triangle meshes with the shared memory manager.
        """
        iterations = 3

        mesh_size = (
            (np.dtype(np.float64).itemsize * (3 + 3 + 3 + 3 + 2))
            + (np.dtype(np.int32).itemsize * (3 + 1))
            + (np.dtype(np.int64).itemsize * (3 + 2 + 3))
        ) * 1000 + ((np.dtype(np.uint8).itemsize * (512 * 512)) * 2)
        write_manager = SharedMemoryManager.with_shm(mesh_size * iterations)

        input_meshes = []
        for _ in range(iterations):
            t_mesh = o3d.geometry.TriangleMesh.create_cylinder()
            t_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.random.rand(len(t_mesh.vertices), 3)
            )
            t_mesh.compute_vertex_normals()
            t_mesh.compute_triangle_normals()

            assert t_mesh.has_vertex_normals(), "Mesh should have vertex normals"
            assert t_mesh.has_triangle_normals(), "Mesh should have triangle normals"

            input_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(
                t_mesh
            )

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "HalfEdgeTriangleMesh", input_mesh
            )
            write_manager.write_geometry(input_mesh, memory_map)

            input_meshes.append(input_mesh)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_mesh = input_meshes[i]
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert_mesh_equal(output_mesh, input_mesh)

    def test_oriented_bounding_box_maps(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading oriented bounding boxes with the shared memory manager.
        """
        iterations = 3

        bbox_size = (3 + 3 + 3 + 9) * np.dtype(np.float64).itemsize
        write_manager = SharedMemoryManager.with_shm(bbox_size * iterations)

        input_obbs = []
        for _ in range(iterations):
            input_obb = o3d.geometry.OrientedBoundingBox()
            input_obb.center = rng.random(3)
            input_obb.color = rng.random(3)
            input_obb.extent = rng.random(3)
            input_obb.R = rng.random((3, 3))

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "OrientedBoundingBox", input_obb
            )
            write_manager.write_geometry(input_obb, memory_map)

            input_obbs.append(input_obb)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_obb = input_obbs[i]
            output_obb = geometry_dict["geometry"]

            assert (
                output_obb != input_obb
            ), "Output and input boxes should be different objects"
            assert np.all(np.asarray(output_obb.center) == np.asarray(input_obb.center))
            assert np.all(np.asarray(output_obb.color) == np.asarray(input_obb.color))
            assert np.all(np.asarray(output_obb.extent) == np.asarray(input_obb.extent))
            assert np.all(np.asarray(output_obb.R) == np.asarray(input_obb.R))

    def test_axis_aligned_bounding_box_maps(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading axis aligned bounding boxes with the shared memory
        manager.
        """
        iterations = 3

        bbox_size = (3 + 3 + 3) * np.dtype(np.float64).itemsize
        write_manager = SharedMemoryManager.with_shm(bbox_size * iterations)

        input_obbs = []
        for _ in range(iterations):
            input_obb = o3d.geometry.AxisAlignedBoundingBox()
            input_obb.max_bound = rng.random(3)
            input_obb.min_bound = rng.random(3)
            input_obb.color = rng.random(3)

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "AxisAlignedBoundingBox", input_obb
            )
            write_manager.write_geometry(input_obb, memory_map)

            input_obbs.append(input_obb)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_obb = input_obbs[i]
            output_obb = geometry_dict["geometry"]

            assert (
                output_obb != input_obb
            ), "Output and input boxes should be different objects"
            assert np.all(
                np.asarray(output_obb.max_bound) == np.asarray(input_obb.max_bound)
            )
            assert np.all(np.asarray(output_obb.color) == np.asarray(input_obb.color))
            assert np.all(
                np.asarray(output_obb.min_bound) == np.asarray(input_obb.min_bound)
            )

    def test_line_set_maps(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading line sets with the shared memory manager.
        """
        iterations = 3

        lineset_size = (
            ((3 + 3) * np.dtype(np.float64).itemsize)
            + (2 * np.dtype(np.int32).itemsize)
        ) * 1000
        write_manager = SharedMemoryManager.with_shm(lineset_size * iterations)

        input_linesets = []
        for _ in range(iterations):
            input_lineset = o3d.geometry.LineSet()
            input_lineset.points = o3d.utility.Vector3dVector(rng.random((1000, 3)))
            input_lineset.colors = o3d.utility.Vector3dVector(rng.random((1000, 3)))
            input_lineset.lines = o3d.utility.Vector2iVector(rng.random((1000, 2)))

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "LineSet", input_lineset
            )
            write_manager.write_geometry(input_lineset, memory_map)

            input_linesets.append(input_lineset)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_lineset = input_linesets[i]
            output_lineset = geometry_dict["geometry"]

            assert (
                output_lineset != input_lineset
            ), "Output and input line sets should be different objects"

            assert np.all(
                np.asarray(output_lineset.points) == np.asarray(input_lineset.points)
            )
            assert np.all(
                np.asarray(output_lineset.colors) == np.asarray(input_lineset.colors)
            )
            assert np.all(
                np.asarray(output_lineset.lines) == np.asarray(input_lineset.lines)
            )

    def test_maps_from_geometry_dict(self, rng: np.random.Generator) -> None:
        """
        Test writing and reading full geometry dictionaries with the shared memory
        manager.
        """
        m_width, m_height = 16, 9

        iterations = 3

        mesh_size = (
            (np.dtype(np.float64).itemsize * (3 + 3 + 3 + 3 + 2))
            + (np.dtype(np.int32).itemsize * (3 + 1))
            + (np.dtype(np.int64).itemsize * (3 + 2 + 3))
        ) * 1000 + ((np.dtype(np.uint8).itemsize * (512 * 512)) * 2)
        write_manager = SharedMemoryManager.with_shm(mesh_size * iterations)

        input_dicts: List[Dict] = []
        for _ in range(iterations):
            input_mesh = get_random_mesh(o3d.geometry.TriangleMesh, rng)

            material = o3d.visualization.rendering.MaterialRecord()
            material.absorption_color = rng.random(3)
            material.absorption_distance = rng.random()
            material.albedo_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.anisotropy_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.ao_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.ao_rough_metal_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.aspect_ratio = rng.random()
            material.base_anisotropy = rng.random()
            material.base_clearcoat = rng.random()
            material.base_clearcoat_roughness = rng.random()
            material.base_color = rng.random(4)
            material.base_metallic = rng.random()
            material.base_reflectance = rng.random()
            material.base_roughness = rng.random()
            material.clearcoat_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.clearcoat_roughness_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.emissive_color = rng.random(4)
            material.ground_plane_axis = rng.random()
            material.has_alpha = rng.random() > 0.5
            material.metallic_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.normal_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.point_size = rng.random()
            material.reflectance_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.roughness_img = o3d.geometry.Image(
                rng.integers(0, 256, (m_height, m_width, 3), dtype=np.uint8)
            )
            material.sRGB_color = rng.random() > 0.5
            material.scalar_max = rng.random()
            material.scalar_min = rng.random()
            material.shader = "non-default-shader"
            material.thickness = rng.random()
            material.transmission = rng.random()

            input_dict = {
                "name": "TriangleMesh",
                "geometry": input_mesh,
                "material": material,
                "group": "TriangleMesh",
                "time": 698592787,
                "is_visible": True,
            }

            memory_map = Geometry3DMemoryMapFactory.from_geometry_dict(input_dict)
            write_manager.write_geometry(input_dict, memory_map)

            input_dicts.append(input_dict)

        for i, geometry_dict in enumerate(write_manager.read_geometries()):
            input_dict = input_dicts[i]
            input_mesh = input_dict["geometry"]
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert_mesh_equal(output_mesh, input_mesh)

            assert input_dict["name"] == geometry_dict["name"]
            assert input_dict["is_visible"] == geometry_dict["is_visible"]
            assert input_dict["time"] == geometry_dict["time"]
            assert input_dict["is_visible"] == geometry_dict["is_visible"]

            input_material = input_dict["material"]
            output_material = geometry_dict["material"]

            for attribute in [
                "absorption_distance",
                "aspect_ratio",
                "base_anisotropy",
                "base_clearcoat",
                "base_clearcoat_roughness",
                "base_metallic",
                "base_reflectance",
                "base_roughness",
                "ground_plane_axis",
                "has_alpha",
                "point_size",
                "sRGB_color",
                "scalar_max",
                "scalar_min",
                "shader",
                "thickness",
                "transmission",
            ]:
                assert getattr(input_material, attribute) == getattr(
                    output_material, attribute
                )

            for attribute in [
                "absorption_color",
                "base_color",
                "emissive_color",
            ]:
                assert np.all(
                    getattr(input_material, attribute)
                    == getattr(output_material, attribute)
                )

            for attribute in [
                "albedo_img",
                "anisotropy_img",
                "ao_img",
                "ao_rough_metal_img",
                "clearcoat_img",
                "clearcoat_roughness_img",
                "metallic_img",
                "normal_img",
                "reflectance_img",
                "roughness_img",
            ]:
                assert np.all(
                    np.asarray(getattr(input_material, attribute))
                    == np.asarray(getattr(output_material, attribute))
                )
