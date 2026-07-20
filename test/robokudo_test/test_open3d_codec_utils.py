import numpy as np

from robokudo.io import open3d_codec_utils


def _create_point_cloud():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    point_cloud = open3d_codec_utils.o3d.geometry.PointCloud()
    point_cloud.points = open3d_codec_utils.o3d.utility.Vector3dVector(points)
    return point_cloud


def test_open3d_point_cloud_base64_pcd_roundtrip():
    point_cloud = _create_point_cloud()
    payload = open3d_codec_utils.encode_open3d_point_cloud_to_base64_pcd(point_cloud)
    restored = open3d_codec_utils.decode_open3d_point_cloud_from_base64_pcd(payload)

    np.testing.assert_allclose(
        np.asarray(restored.points),
        np.asarray(point_cloud.points),
    )


def test_decode_open3d_point_cloud_invalid_base64_results_in_empty_cloud():
    restored = open3d_codec_utils.decode_open3d_point_cloud_from_base64_pcd("%%%")
    assert len(np.asarray(restored.points)) == 0


def test_is_open3d_point_cloud_detects_type():
    point_cloud = _create_point_cloud()
    assert open3d_codec_utils.is_open3d_point_cloud(point_cloud) is True
    assert open3d_codec_utils.is_open3d_point_cloud({"not": "a point cloud"}) is False
