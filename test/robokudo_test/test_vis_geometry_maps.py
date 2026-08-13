from multiprocessing import shared_memory

import numpy as np
import open3d as o3d
import pytest
from robokudo.vis.multiprocessed_o3d_visualizer import (
    Geometry3DMemoryMapFactory,
    SharedMemoryManager,
)


class TestVisGeometryMaps(object):
    @pytest.fixture
    def write_manager(self) -> SharedMemoryManager:
        return SharedMemoryManager()

    def test_point_cloud_maps(self, write_manager: SharedMemoryManager) -> None:
        """Test writing and reading point clouds with the shared memory manager."""
        iterations = 3
        read_idx = 0

        shm = shared_memory.SharedMemory(
            create=True,
            size=(1000 * np.dtype(np.float64).itemsize * (3 + 3 + 3 + 9)) * iterations,
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
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_pcd)

            inputs_pcds.append(input_pcd)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_pcd = inputs_pcds[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
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

    def test_mesh_base_maps(self, write_manager: SharedMemoryManager) -> None:
        """Test writing and reading base meshes with the shared memory manager."""
        iterations = 3
        read_idx = 0

        mesh_size = (np.dtype(np.float64).itemsize * (3 + 3 + 3)) * 1000

        shm = shared_memory.SharedMemory(
            create=True,
            size=mesh_size * iterations,
        )

        input_meshes = []
        for _ in range(iterations):
            input_mesh = o3d.geometry.MeshBase()
            input_mesh.vertices = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
            input_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.random.rand(1000, 3)
            )
            input_mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.random.rand(1000, 3)
            )

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "MeshBase", input_mesh
            )
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_mesh)

            input_meshes.append(input_mesh)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_mesh = input_meshes[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert np.all(
                np.asarray(output_mesh.vertices) == np.asarray(input_mesh.vertices)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_colors)
                == np.asarray(input_mesh.vertex_colors)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_normals)
                == np.asarray(input_mesh.vertex_normals)
            )

    def test_triangle_mesh_maps(self, write_manager: SharedMemoryManager) -> None:
        """Test writing and reading triangle meshes with the shared memory manager."""
        iterations = 3
        read_idx = 0

        mesh_size = (
            (np.dtype(np.float64).itemsize * (3 + 3 + 3 + 3 + 2))
            + (np.dtype(np.int32).itemsize * (3 + 1))
            + (np.dtype(np.int64).itemsize * (3 + 2 + 3))
        ) * 1000 + ((np.dtype(np.uint8).itemsize * (512 * 512)) * 2)

        shm = shared_memory.SharedMemory(
            create=True,
            size=mesh_size * iterations,
        )

        input_meshes = []
        for _ in range(iterations):
            input_mesh = o3d.geometry.TriangleMesh()
            input_mesh.vertices = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
            input_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.random.rand(1000, 3)
            )
            input_mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.random.rand(1000, 3)
            )
            input_mesh.triangles = o3d.utility.Vector3iVector(np.random.rand(1000, 3))
            input_mesh.triangle_normals = o3d.utility.Vector3dVector(
                np.random.rand(1000, 3)
            )
            input_mesh.triangle_uvs = o3d.utility.Vector2dVector(
                np.random.rand(1000, 2)
            )
            input_mesh.triangle_material_ids = o3d.utility.IntVector(
                np.random.randint(0, 10, size=1000, dtype=np.int32)
            )
            input_mesh.adjacency_list = [{3, 8, 9}, {0, 4}, {0, 2, 4}]
            input_mesh.textures = [
                o3d.geometry.Image(
                    np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
                ),
                o3d.geometry.Image(
                    np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
                ),
            ]

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "TriangleMesh", input_mesh
            )
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_mesh)

            input_meshes.append(input_mesh)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_mesh = input_meshes[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert np.all(
                np.asarray(output_mesh.vertices) == np.asarray(input_mesh.vertices)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_colors)
                == np.asarray(input_mesh.vertex_colors)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_normals)
                == np.asarray(input_mesh.vertex_normals)
            )

            assert np.all(
                np.asarray(output_mesh.triangles) == np.asarray(input_mesh.triangles)
            )
            assert np.all(
                np.asarray(output_mesh.triangle_normals)
                == np.asarray(input_mesh.triangle_normals)
            )
            assert np.all(
                np.asarray(output_mesh.triangle_uvs)
                == np.asarray(input_mesh.triangle_uvs)
            )
            assert np.all(
                np.asarray(output_mesh.triangle_material_ids)
                == np.asarray(input_mesh.triangle_material_ids)
            )
            for j in range(len(input_mesh.adjacency_list)):
                assert isinstance(output_mesh.adjacency_list[j], set)
                assert np.all(
                    np.array(list(output_mesh.adjacency_list[j]))
                    == np.array(list(input_mesh.adjacency_list[j]))
                )
            for j in range(len(input_mesh.textures)):
                assert isinstance(output_mesh.textures[j], o3d.geometry.Image)
                assert np.all(
                    np.asarray(output_mesh.textures[j])
                    == np.asarray(input_mesh.textures[j])
                )

    def test_tetra_mesh_maps(self, write_manager: SharedMemoryManager) -> None:
        """Test writing and reading base meshes with the shared memory manager."""
        iterations = 3
        read_idx = 0

        mesh_size = (
            (np.dtype(np.float64).itemsize * (3 + 3 + 3))
            + (np.dtype(np.int64).itemsize * 4)
        ) * 1000

        shm = shared_memory.SharedMemory(
            create=True,
            size=mesh_size * iterations,
        )

        input_meshes = []
        for _ in range(iterations):
            input_mesh = o3d.geometry.TetraMesh()
            input_mesh.vertices = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
            input_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.random.rand(1000, 3)
            )
            input_mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.random.rand(1000, 3)
            )
            input_mesh.tetras = o3d.utility.Vector4iVector(np.random.rand(1000, 4))

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "TetraMesh", input_mesh
            )
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_mesh)

            input_meshes.append(input_mesh)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_mesh = input_meshes[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert np.all(
                np.asarray(output_mesh.vertices) == np.asarray(input_mesh.vertices)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_colors)
                == np.asarray(input_mesh.vertex_colors)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_normals)
                == np.asarray(input_mesh.vertex_normals)
            )
            assert np.all(
                np.asarray(output_mesh.tetras) == np.asarray(input_mesh.tetras)
            )

    def test_half_edge_mesh_maps(self, write_manager: SharedMemoryManager) -> None:
        """Test writing and reading triangle meshes with the shared memory manager."""
        iterations = 3
        read_idx = 0

        mesh_size = (
            (np.dtype(np.float64).itemsize * (3 + 3 + 3 + 3 + 2))
            + (np.dtype(np.int32).itemsize * (3 + 1))
            + (np.dtype(np.int64).itemsize * (3 + 2 + 3))
        ) * 1000 + ((np.dtype(np.uint8).itemsize * (512 * 512)) * 2)

        shm = shared_memory.SharedMemory(
            create=True,
            size=mesh_size * iterations,
        )

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
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_mesh)

            input_meshes.append(input_mesh)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_mesh = input_meshes[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
            output_mesh = geometry_dict["geometry"]

            assert (
                output_mesh != input_mesh
            ), "Output and input meshes should be different objects"

            assert np.all(
                np.asarray(output_mesh.vertices) == np.asarray(input_mesh.vertices)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_colors)
                == np.asarray(input_mesh.vertex_colors)
            )
            assert np.all(
                np.asarray(output_mesh.vertex_normals)
                == np.asarray(input_mesh.vertex_normals)
            )

            assert np.all(
                np.asarray(output_mesh.triangles) == np.asarray(input_mesh.triangles)
            )
            assert np.all(
                np.asarray(output_mesh.triangle_normals)
                == np.asarray(input_mesh.triangle_normals)
            )
            for j in range(len(output_mesh.ordered_half_edge_from_vertex)):
                assert isinstance(
                    output_mesh.ordered_half_edge_from_vertex[j], o3d.utility.IntVector
                )
                assert np.all(
                    np.asarray(output_mesh.ordered_half_edge_from_vertex[j])
                    == np.asarray(input_mesh.ordered_half_edge_from_vertex[j])
                )
            for j in range(len(output_mesh.half_edges)):
                assert isinstance(output_mesh.half_edges[j], o3d.geometry.HalfEdge)
                assert output_mesh.half_edges[j].next == input_mesh.half_edges[j].next
                assert (
                    output_mesh.half_edges[j].triangle_index
                    == input_mesh.half_edges[j].triangle_index
                )
                assert output_mesh.half_edges[j].twin == input_mesh.half_edges[j].twin
                assert np.all(
                    output_mesh.half_edges[j].vertex_indices
                    == input_mesh.half_edges[j].vertex_indices
                )

    def test_oriented_bounding_box_maps(
        self, write_manager: SharedMemoryManager
    ) -> None:
        """Test writing and reading oriented bounding boxes with the shared memory manager."""
        iterations = 3
        read_idx = 0
        write_idx = 0

        bbox_size = (3 + 3 + 3 + 9) * np.dtype(np.float64).itemsize

        shm = shared_memory.SharedMemory(
            create=True,
            size=bbox_size * iterations,
        )

        input_obbs = []
        for _ in range(iterations):
            input_obb = o3d.geometry.OrientedBoundingBox()
            input_obb.center = np.random.rand(3)
            input_obb.color = np.random.rand(3)
            input_obb.extent = np.random.rand(3)
            input_obb.R = np.random.rand(3, 3)

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "OrientedBoundingBox", input_obb
            )
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_obb)

            input_obbs.append(input_obb)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_obb = input_obbs[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
            output_obb = geometry_dict["geometry"]

            assert (
                output_obb != input_obb
            ), "Output and input boxes should be different objects"
            assert np.all(np.asarray(output_obb.center) == np.asarray(input_obb.center))
            assert np.all(np.asarray(output_obb.color) == np.asarray(input_obb.color))
            assert np.all(np.asarray(output_obb.extent) == np.asarray(input_obb.extent))
            assert np.all(np.asarray(output_obb.R) == np.asarray(input_obb.R))

    def test_axis_aligned_bounding_box_maps(
        self, write_manager: SharedMemoryManager
    ) -> None:
        """Test writing and reading axis aligned bounding boxes with the shared memory manager."""
        iterations = 3
        read_idx = 0

        bbox_size = (3 + 3 + 3) * np.dtype(np.float64).itemsize

        shm = shared_memory.SharedMemory(
            create=True,
            size=bbox_size * iterations,
        )

        input_obbs = []
        for _ in range(iterations):
            input_obb = o3d.geometry.AxisAlignedBoundingBox()
            input_obb.max_bound = np.random.rand(3)
            input_obb.min_bound = np.random.rand(3)
            input_obb.color = np.random.rand(3)

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "AxisAlignedBoundingBox", input_obb
            )
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_obb)

            input_obbs.append(input_obb)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_obb = input_obbs[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
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

    def test_line_set_maps(self, write_manager: SharedMemoryManager) -> None:
        """Test writing and reading line sets with the shared memory manager."""
        iterations = 3
        read_idx = 0

        lineset_size = (
            ((3 + 3) * np.dtype(np.float64).itemsize)
            + (2 * np.dtype(np.int32).itemsize)
        ) * 1000

        shm = shared_memory.SharedMemory(
            create=True,
            size=lineset_size * iterations * 2,
        )

        input_linesets = []
        for _ in range(iterations):
            input_lineset = o3d.geometry.LineSet()
            input_lineset.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
            input_lineset.colors = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
            input_lineset.lines = o3d.utility.Vector2iVector(np.random.rand(1000, 2))

            memory_map = Geometry3DMemoryMapFactory.from_geometry(
                "LineSet", input_lineset
            )
            write_idx = write_manager.append(memory_map)
            memory_map.write_geometry(shm, write_idx, input_lineset)

            input_linesets.append(input_lineset)

        for i, (read_ixd, memory_map) in enumerate(write_manager.read()):
            input_lineset = input_linesets[i]

            geometry_dict, read_idx = memory_map.as_geometry_dict(shm, read_idx)
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
