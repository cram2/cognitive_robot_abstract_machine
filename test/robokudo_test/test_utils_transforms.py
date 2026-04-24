import geometry_msgs.msg
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robokudo.utils.transform import (
    get_pose_from_transform_matrix,
    get_transform_matrix_from_pose,
    get_transform_matrix,
    get_transform_matrix_from_translation,
    get_transform_matrix_from_q,
    get_quaternion_from_rotation_matrix,
    get_quaternion_from_transform_matrix,
    get_translation_from_transform_matrix,
    get_rotation_from_transform_matrix,
    quaternion_about_axis,
    get_transform_matrix_for_rotation_around_axis,
    get_rotation_matrix_from_euler_angles,
    get_transform_from_plane_equation,
    construct_rotation_matrix,
    get_rotation_matrix_from_direction_vector,
)


class TestUtilsTransforms(object):
    @pytest.fixture(scope="function")
    def transform_matrix(self, request) -> np.ndarray:
        seq, angles = request.param
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = Rotation.from_euler(
            seq, angles, degrees=True
        ).as_matrix()
        return transform_matrix

    ROTATION_CASES = [
        (("xyz", [0, 0, 0]), [0, 0, 0, 1]),
        (("x", 90), [0.70710678, 0.0, 0.0, 0.70710678]),
        (("x", 180), [1.0, 0.0, 0.0, 0.0]),
        (("x", 270), [-0.70710678, 0.0, 0.0, 0.70710678]),
        (("y", 90), [0.0, 0.70710678, 0.0, 0.70710678]),
        (("y", 180), [0.0, 1.0, 0.0, 0.0]),
        (("y", 270), [0.0, -0.70710678, 0.0, 0.70710678]),
        (("z", 90), [0.0, 0.0, 0.70710678, 0.70710678]),
        (("z", 180), [0.0, 0.0, 1.0, 0.0]),
        (("z", 270), [0.0, 0.0, -0.70710678, 0.70710678]),
    ]

    TRANSLATION_CASES = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]

    @pytest.mark.parametrize(
        ["transform_matrix", "quaternion"],
        ROTATION_CASES,
        indirect=["transform_matrix"],
    )
    def test_get_pose_from_transform_matrix_valid_rotations(
        self,
        transform_matrix: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ):
        pose = get_pose_from_transform_matrix(transform_matrix)

        pose_quat = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )

        assert np.allclose(pose_quat, quaternion)
        assert np.isclose(np.linalg.norm(pose_quat), 1.0, atol=1e-6)
        assert [pose.position.x, pose.position.y, pose.position.z] == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("translation", TRANSLATION_CASES)
    def test_get_pose_from_transform_matrix_valid_translations(
        self, translation: tuple[float, float, float]
    ):
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        pose = get_pose_from_transform_matrix(matrix)

        pose_quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        assert pose_quat == [0.0, 0.0, 0.0, 1.0]
        assert [pose.position.x, pose.position.y, pose.position.z] == translation

    def test_get_pose_from_transform_matrix_invalid_matrix(self):
        matrix = np.zeros((4, 4))
        # Please note that this exception has been introduced in newer versions of scipy. 1.11.X will fail on this!
        assert pytest.raises(ValueError, get_pose_from_transform_matrix, matrix)

    @pytest.mark.parametrize(
        ["transform_matrix", "quaternion"],
        ROTATION_CASES,
        indirect=["transform_matrix"],
    )
    def test_get_transform_matrix_from_pose_valid_rotations(
        self,
        transform_matrix: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ):
        pose = geometry_msgs.msg.Pose()
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        matrix = get_transform_matrix_from_pose(pose)

        assert np.allclose(matrix, transform_matrix)
        assert np.all(matrix[:3, 3] == [0.0, 0.0, 0.0])

    @pytest.mark.parametrize("translation", TRANSLATION_CASES)
    def test_get_transform_matrix_from_pose_valid_translations(
        self, translation: tuple[float, float, float]
    ):
        pose = geometry_msgs.msg.Pose()
        pose.position.x = translation[0]
        pose.position.y = translation[1]
        pose.position.z = translation[2]

        matrix = get_transform_matrix_from_pose(pose)

        assert np.all(
            matrix[:3, 3] == translation
        ), "translation should not change during conversion"
        assert np.all(
            matrix[:3, :3] == np.eye(3)
        ), "rotation should not change from default during conversion with translation only"

    def test_get_transform_matrix_from_pose_invalid_rotations(self):
        pose = geometry_msgs.msg.Pose()
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0

        assert pytest.raises(ValueError, get_transform_matrix_from_pose, pose)

    @pytest.mark.parametrize(
        ["transform_matrix", "quaternion"],
        ROTATION_CASES,
        indirect=["transform_matrix"],
    )
    def test_get_transform_matrix_valid_rotations(
        self,
        transform_matrix: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ):
        matrix = get_transform_matrix(transform_matrix[:3, :3], transform_matrix[:3, 3])
        assert np.all(matrix == transform_matrix)

    @pytest.mark.parametrize("translation", TRANSLATION_CASES)
    def test_get_transform_matrix_valid_translations(
        self, translation: tuple[float, float, float]
    ):
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = translation
        matrix = get_transform_matrix(transform_matrix[:3, :3], transform_matrix[:3, 3])
        assert np.all(matrix == transform_matrix)

    @pytest.mark.parametrize("translation", TRANSLATION_CASES)
    def test_get_transform_matrix_from_translation(
        self, translation: tuple[float, float, float]
    ):
        matrix = get_transform_matrix_from_translation(translation)
        assert np.all(
            matrix[:3, 3] == translation
        ), "translation should not change during conversion"

    @pytest.mark.parametrize(
        ["transform_matrix", "quaternion"],
        ROTATION_CASES,
        indirect=["transform_matrix"],
    )
    def test_get_transform_matrix_from_q_valid_rotations(
        self, transform_matrix, quaternion
    ):
        matrix = get_transform_matrix_from_q(
            np.array(quaternion), np.array([0.0, 0.0, 0.0])
        )
        assert np.allclose(
            matrix, transform_matrix
        ), "matrix should not change during conversion"

    @pytest.mark.parametrize("translation", TRANSLATION_CASES)
    def test_get_transform_matrix_from_q_valid_translations(
        self, translation: tuple[float, float, float]
    ):
        matrix = get_transform_matrix_from_q(
            np.array([0.0, 0.0, 0.0, 1.0]), translation
        )
        assert np.all(
            matrix[:3, :3] == np.eye(3)
        ), "rotation should not change during conversion with translation only"
        assert np.all(
            matrix[:3, 3] == translation
        ), "translation should not change during conversion"

    def test_get_transform_matrix_from_q_invalid_quaternion(self):
        assert pytest.raises(
            ValueError,
            get_transform_matrix_from_q,
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )

    @pytest.mark.parametrize(
        ["transform_matrix", "quaternion"],
        ROTATION_CASES,
        indirect=["transform_matrix"],
    )
    def test_get_quaternion_from_rotation_matrix(
        self,
        transform_matrix: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ):
        q = get_quaternion_from_rotation_matrix(transform_matrix[:3, :3])
        assert np.allclose(
            q, quaternion
        ), "rotation should not change during conversion"

    def test_get_quaternion_from_rotation_matrix_invalid_rotation_matrix(self):
        assert pytest.raises(
            ValueError, get_quaternion_from_rotation_matrix, np.zeros((3, 3))
        )

    @pytest.mark.parametrize(
        ["transform_matrix", "quaternion"],
        ROTATION_CASES,
        indirect=["transform_matrix"],
    )
    def test_get_quaternion_from_transform_matrix(
        self,
        transform_matrix: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ):
        q = get_quaternion_from_transform_matrix(transform_matrix)
        assert np.allclose(
            q, quaternion
        ), "rotation should not change during conversion"

    def test_get_quaternion_from_transform_matrix_invalid_rotation_matrix(self):
        assert pytest.raises(
            ValueError, get_quaternion_from_rotation_matrix, np.zeros((4, 4))
        )

    @pytest.mark.parametrize("translation", TRANSLATION_CASES)
    def test_get_translation_from_transform_matrix(
        self, translation: tuple[float, float, float]
    ):
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = translation
        tran = get_translation_from_transform_matrix(transform_matrix)
        assert np.all(tran == translation)

    @pytest.mark.parametrize(
        ["transform_matrix", "quaternion"],
        ROTATION_CASES,
        indirect=["transform_matrix"],
    )
    def test_get_rotation_from_transform_matrix(
        self,
        transform_matrix: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ):
        rot = get_rotation_from_transform_matrix(transform_matrix)
        assert np.all(rot == transform_matrix[:3, :3])
        assert np.allclose(Rotation.from_matrix(rot).as_quat(True), quaternion)

    @pytest.mark.parametrize(
        ["rotation", "quaternion"],
        [
            (("x", 0), [0, 0, 0, 1]),  # 0 radians: identity quaternion
            (("x", np.pi / 2), [0.70710678, 0, 0, 0.70710678]),  # 90° in radians
            (("x", np.pi), [1, 0, 0, 0]),  # 180° in radians
            (("x", 3 * np.pi / 2), [-0.70710678, 0, 0, 0.70710678]),  # 270° in radians
            (("y", np.pi / 2), [0, 0.70710678, 0, 0.70710678]),
            (("y", np.pi), [0, 1, 0, 0]),
            (("y", 3 * np.pi / 2), [0, -0.70710678, 0, 0.70710678]),
            (("z", np.pi / 2), [0, 0, 0.70710678, 0.70710678]),
            (("z", np.pi), [0, 0, 1, 0]),
            (("z", 3 * np.pi / 2), [0, 0, -0.70710678, 0.70710678]),
        ],
    )
    def test_quaternion_about_axis(
        self, rotation: tuple[str, float], quaternion: tuple[float, float, float, float]
    ):
        axis = (
            1 if rotation[0] == "x" else 0,
            1 if rotation[0] == "y" else 0,
            1 if rotation[0] == "z" else 0,
        )
        result = quaternion_about_axis(rotation[1], axis)
        assert np.allclose(result, quaternion, atol=1e-6)

    @pytest.mark.parametrize(
        ["rotation", "transform_matrix"],
        [
            (("x", 0), np.eye(4)),  # 0 radians: identity matrix
            (
                ("x", np.pi / 2),
                np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
            ),  # 90° in radians
            (
                ("x", np.pi),
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            ),  # 180° in radians
            (
                ("x", 3 * np.pi / 2),
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            ),  # 270° in radians
            (
                ("y", np.pi / 2),
                [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
            ),
            (("y", np.pi), [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
            (
                ("y", 3 * np.pi / 2),
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            ),
            (
                ("z", np.pi / 2),
                [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
            (("z", np.pi), [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            (
                ("z", 3 * np.pi / 2),
                [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ),
        ],
    )
    def test_get_transform_matrix_for_rotation_around_axis(
        self, rotation: tuple[str, float], transform_matrix: list[list[float]]
    ):
        axis = (
            1 if rotation[0] == "x" else 0,
            1 if rotation[0] == "y" else 0,
            1 if rotation[0] == "z" else 0,
        )
        matrix = get_transform_matrix_for_rotation_around_axis(rotation[1], axis)

        assert np.allclose(matrix, transform_matrix, atol=1e-6)

    @pytest.mark.parametrize(
        ["rotation", "rotation_matrix"],
        [
            ({"x": 0}, np.eye(3)),  # 0 radians: identity matrix
            (
                {"x": np.pi / 2},
                np.array(
                    [
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0],
                    ]
                ),
            ),  # 90° in radians
            (
                {"x": np.pi},
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ],
            ),  # 180° in radians
            (
                {"x": 3 * np.pi / 2},
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
            ),  # 270° in radians
            (
                {"y": np.pi / 2},
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0],
                ],
            ),
            (
                {"y": np.pi},
                [
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                ],
            ),
            (
                {"y": 3 * np.pi / 2},
                [
                    [0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0],
                ],
            ),
            (
                {"z": np.pi / 2},
                [
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                ],
            ),
            (
                {"z": np.pi},
                [
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                ],
            ),
            (
                {"z": 3 * np.pi / 2},
                [
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1],
                ],
            ),
            (
                {"x": 0, "y": np.pi, "z": np.pi},
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ],
            ),
            (
                {"x": np.pi, "y": 0, "z": np.pi},
                [
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                ],
            ),
            (
                {"x": np.pi, "y": np.pi, "z": 0},
                [
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                ],
            ),
        ],
    )
    def test_get_rotation_matrix_from_euler_angles(
        self, rotation: dict[str, float], rotation_matrix: list[list[float]]
    ):
        rotation = {"x": 0, "y": 0, "z": 0} | rotation
        assert np.allclose(
            get_rotation_matrix_from_euler_angles(**rotation), rotation_matrix
        )

    @pytest.mark.parametrize(
        ["plane_equation", "expected_transform"],
        [
            # Base plane
            (np.array([0, 0, 1, 0]), np.eye(4)),
            # Test with translation
            (
                np.array([0, 0, 1, -1]),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
            ),
            # Test with rotation
            (
                np.array([1, 0, 0, 0]),
                np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
            ),
        ],
    )
    def test_get_transform_from_plane_equation(
        self, plane_equation: np.ndarray, expected_transform: np.ndarray
    ):
        transform = get_transform_from_plane_equation(plane_equation)
        assert np.allclose(transform, expected_transform)

    @pytest.mark.parametrize(
        ["rotation", "axis_order", "new_rotation"],
        [
            (np.eye(3), (0, 1, 2), np.eye(3)),  # Axis order unchanged
            (
                np.eye(3),
                (1, 0, 2),
                np.array(
                    [
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, -1],
                    ]
                ),
            ),  # Swap x and y
            (
                np.eye(3),
                (2, 1, 0),
                np.array(
                    [
                        [0, 0, -1],
                        [0, 1, 0],
                        [1, 0, 0],
                    ]
                ),
            ),  # Swap x and z
            (
                np.eye(3),
                (0, 2, 1),
                np.array(
                    [
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0],
                    ]
                ),
            ),  # Swap y and z
            (
                np.eye(3),
                (2, 1, 0),
                np.array(
                    [
                        [0, 0, -1],
                        [0, 1, 0],
                        [1, 0, 0],
                    ]
                ),
            ),  # Reverse
        ],
    )
    def test_construct_rotation_matrix(
        self,
        rotation: np.ndarray,
        axis_order: tuple[int, int, int],
        new_rotation: np.ndarray,
    ):
        assert np.allclose(
            construct_rotation_matrix(rotation, axis_order), new_rotation
        )

    @pytest.mark.parametrize(
        "axis_order",
        [
            [0, 1],  # Not enough axis
            [0, 1, 2, 3],  # Too many axis
            [0, 1, 3],  # Invalid axis index
        ],
    )
    def test_construct_rotation_matrix_invalid_axis_order(self, axis_order: list[int]):
        assert pytest.raises(
            ValueError, construct_rotation_matrix, np.eye(3), axis_order
        )

    @pytest.mark.parametrize(
        ["direction", "up_hint"],
        [
            (np.array([1, 0, 0]), np.array([0, 0, 1])),
            (np.array([0, 0, 1]), np.array([0, 0, 1])),
            (np.random.rand(3), np.array([0, 0, 1])),
            (np.array([1, 0, 0]), np.array([0, 1, 0])),
        ],
    )
    def test_get_rotation_matrix_from_direction_vector(
        self, direction: np.ndarray, up_hint: np.ndarray
    ):
        rot = get_rotation_matrix_from_direction_vector(direction, up_hint=up_hint)

        assert np.allclose(rot[:, 0], direction / np.linalg.norm(direction))
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(rot), 1, atol=1e-6)

    def test_get_rotation_matrix_from_direction_vector_invalid_vector(self):
        assert pytest.raises(
            ValueError, get_rotation_matrix_from_direction_vector, np.array([0, 0, 0])
        )
