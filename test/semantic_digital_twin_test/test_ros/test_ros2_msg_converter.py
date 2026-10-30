import os
import shutil
import subprocess
import sys

import geometry_msgs.msg as geometry_msgs
import pytest
from importlib.resources import files
from pathlib import Path

from krrood.utils import get_full_class_name
from semantic_digital_twin.adapters.ros.exceptions import (
    CannotConvertRos2ToSemDTError,
    CannotConvertSemDTToRos2Error,
)
from semantic_digital_twin.adapters.ros.msg_converter import (
    Ros2ToSemDTConverter,
    SemDTToRos2Converter,
)
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    LaserScanToSemDTConverter,
)
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    BoxToRos2Converter,
    PoseToRos2Converter,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Scale,
    Cylinder,
    Sphere,
    Mesh,
)


def test_convert_transform(cylinder_bot_world):
    transform = geometry_msgs.TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "bot"
    transform.transform.translation.x = 1.0
    transform.transform.rotation.x = 1.0
    transform.transform.rotation.w = 0.0
    transformation_matrix = Ros2ToSemDTConverter.convert(
        transform, world=cylinder_bot_world
    )
    position = transformation_matrix.position.evaluate()
    rotation = transformation_matrix.quaternion.evaluate()
    assert position[0] == transform.transform.translation.x
    assert position[1] == transform.transform.translation.y
    assert position[2] == transform.transform.translation.z

    assert rotation[0] == transform.transform.rotation.x
    assert rotation[1] == transform.transform.rotation.y
    assert rotation[2] == transform.transform.rotation.z
    assert rotation[3] == transform.transform.rotation.w
    assert transformation_matrix.child_frame == cylinder_bot_world.get_body_by_name(
        "bot"
    )
    assert transformation_matrix.reference_frame == cylinder_bot_world.get_body_by_name(
        "map"
    )

    transform2 = SemDTToRos2Converter.convert(transformation_matrix)
    assert transform == transform2


def test_convert_point_stamped(cylinder_bot_world):
    point_msg = geometry_msgs.PointStamped()
    point_msg.header.frame_id = "map"
    point_msg.point.x = 1.0
    point_msg.point.y = 2.0
    point_msg.point.z = 3.0

    point = Ros2ToSemDTConverter.convert(point_msg, world=cylinder_bot_world)
    coords = point.evaluate()

    assert coords[0] == point_msg.point.x
    assert coords[1] == point_msg.point.y
    assert coords[2] == point_msg.point.z

    assert point.reference_frame == cylinder_bot_world.get_body_by_name("map")

    point_msg2 = SemDTToRos2Converter.convert(point)
    assert point_msg == point_msg2


def test_convert_quaternion(cylinder_bot_world):
    quat_msg = geometry_msgs.QuaternionStamped()
    quat_msg.header.frame_id = "map"
    quat_msg.quaternion.x = 1.0
    quat_msg.quaternion.y = 0.0
    quat_msg.quaternion.z = 0.0
    quat_msg.quaternion.w = 0.0

    quat = Ros2ToSemDTConverter.convert(quat_msg, world=cylinder_bot_world)
    values = quat.evaluate()

    assert values[0] == quat_msg.quaternion.x
    assert values[1] == quat_msg.quaternion.y
    assert values[2] == quat_msg.quaternion.z
    assert values[3] == quat_msg.quaternion.w

    assert quat.reference_frame == cylinder_bot_world.get_body_by_name("map")

    quat_msg2 = SemDTToRos2Converter.convert(quat)
    assert quat_msg == quat_msg2


def test_convert_vector3(cylinder_bot_world):
    vec_msg = geometry_msgs.Vector3Stamped()
    vec_msg.header.frame_id = "map"
    vec_msg.vector.x = -1.0
    vec_msg.vector.y = 0.5
    vec_msg.vector.z = 2.5

    vec = Ros2ToSemDTConverter.convert(vec_msg, world=cylinder_bot_world)
    values = vec.evaluate()

    assert values[0] == vec_msg.vector.x
    assert values[1] == vec_msg.vector.y
    assert values[2] == vec_msg.vector.z

    assert vec.reference_frame == cylinder_bot_world.get_body_by_name("map")

    vec_msg2 = SemDTToRos2Converter.convert(vec)
    assert vec_msg == vec_msg2


def test_convert_pose_stamped(cylinder_bot_world):
    pose_msg = geometry_msgs.PoseStamped()
    pose_msg.header.frame_id = "map"
    pose_msg.pose.position.x = 1.2
    pose_msg.pose.position.y = -0.4
    pose_msg.pose.position.z = 0.7
    pose_msg.pose.orientation.x = 0.0
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 1.0

    pose = Ros2ToSemDTConverter.convert(pose_msg, world=cylinder_bot_world)

    pos = pose.position.evaluate()
    quat = pose.quaternion.evaluate()

    assert pos[0] == pose_msg.pose.position.x
    assert pos[1] == pose_msg.pose.position.y
    assert pos[2] == pose_msg.pose.position.z

    assert quat[0] == pose_msg.pose.orientation.x
    assert quat[1] == pose_msg.pose.orientation.y
    assert quat[2] == pose_msg.pose.orientation.z
    assert quat[3] == pose_msg.pose.orientation.w

    assert pose.reference_frame == cylinder_bot_world.get_body_by_name("map")

    pose_msg2 = SemDTToRos2Converter.convert(pose)
    assert pose_msg == pose_msg2


def test_convert_pose(cylinder_bot_world):
    pose_msg = geometry_msgs.Pose()
    pose_msg.position.x = 1.2
    pose_msg.position.y = -0.4
    pose_msg.position.z = 0.7
    pose_msg.orientation.x = 0.0
    pose_msg.orientation.y = 0.0
    pose_msg.orientation.z = 0.0
    pose_msg.orientation.w = 1.0

    pose = Ros2ToSemDTConverter.convert(pose_msg, world=cylinder_bot_world)

    pos = pose.position.evaluate()
    quat = pose.quaternion.evaluate()

    assert pos[0] == pose_msg.position.x
    assert pos[1] == pose_msg.position.y
    assert pos[2] == pose_msg.position.z

    assert quat[0] == pose_msg.orientation.x
    assert quat[1] == pose_msg.orientation.y
    assert quat[2] == pose_msg.orientation.z
    assert quat[3] == pose_msg.orientation.w

    pose_msg2 = PoseToRos2Converter.convert(pose)
    assert pose_msg == pose_msg2


def test_no_frame_id(cylinder_bot_world):
    SemDTToRos2Converter.convert(HomogeneousTransformationMatrix())
    SemDTToRos2Converter.convert(Pose())
    SemDTToRos2Converter.convert(Point3())
    SemDTToRos2Converter.convert(Vector3())
    SemDTToRos2Converter.convert(Quaternion())

    Ros2ToSemDTConverter.convert(geometry_msgs.TransformStamped(), cylinder_bot_world)
    Ros2ToSemDTConverter.convert(geometry_msgs.PointStamped(), cylinder_bot_world)
    Ros2ToSemDTConverter.convert(geometry_msgs.QuaternionStamped(), cylinder_bot_world)
    Ros2ToSemDTConverter.convert(geometry_msgs.Vector3Stamped(), cylinder_bot_world)
    Ros2ToSemDTConverter.convert(geometry_msgs.PoseStamped(), cylinder_bot_world)


def test_convert_box_shape(cylinder_bot_world):
    box = Box(
        scale=Scale(1, 2, 3),
        color=Color(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 2, 3, reference_frame=cylinder_bot_world.get_body_by_name("map")
        ),
    )
    shape = SemDTToRos2Converter.convert(box)

    box2 = Ros2ToSemDTConverter.convert(shape, cylinder_bot_world)
    assert box == box2

    shape2 = SemDTToRos2Converter.convert(box2)
    assert shape == shape2


def test_convert_cylinder_shape(cylinder_bot_world):
    cylinder = Cylinder(
        width=1.0,
        height=2.0,
        color=Color(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            1.0, 2.0, 3.0, reference_frame=cylinder_bot_world.get_body_by_name("map")
        ),
    )
    shape = SemDTToRos2Converter.convert(cylinder)

    cylinder2 = Ros2ToSemDTConverter.convert(shape, cylinder_bot_world)
    assert cylinder == cylinder2

    shape2 = SemDTToRos2Converter.convert(cylinder2)
    assert shape == shape2


def test_convert_sphere_shape(cylinder_bot_world):
    sphere = Sphere(
        radius=1.0,
        color=Color(),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            1.0, 2.0, 3.0, reference_frame=cylinder_bot_world.get_body_by_name("map")
        ),
    )
    shape = SemDTToRos2Converter.convert(sphere)

    sphere2 = Ros2ToSemDTConverter.convert(shape, cylinder_bot_world)
    assert sphere == sphere2

    shape2 = SemDTToRos2Converter.convert(sphere2)
    assert shape == shape2


def test_convert_mesh_shape(cylinder_bot_world):
    stl_file = os.path.join(
        Path(files("semantic_digital_twin")).parent.parent,
        "resources",
        "stl",
        "jeroen_cup.stl",
    )
    mesh = Mesh(
        filename=stl_file,
        color=Color.from_rgba(red=0.0, green=0.0, blue=0.0, opacity=0.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 2, 3, reference_frame=cylinder_bot_world.get_body_by_name("map")
        ),
    )
    shape = SemDTToRos2Converter.convert(mesh)

    mesh2 = Ros2ToSemDTConverter.convert(shape, cylinder_bot_world)
    assert mesh == mesh2

    shape2 = SemDTToRos2Converter.convert(mesh2)
    assert shape == shape2


# %% drawing a mesh from its file


def textured_chair(cylinder_bot_world) -> Mesh:
    """
    :return: A mesh whose file paints it with a texture, placed in the map of
        `cylinder_bot_world`.
    """
    ply_directory = os.path.join(
        Path(files("semantic_digital_twin")).parent.parent, "resources", "ply"
    )
    return Mesh.from_ply_file(
        ply_file_path=os.path.join(ply_directory, "chair.ply"),
        texture_file_path=os.path.join(ply_directory, "chair_texture.png"),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=cylinder_bot_world.get_body_by_name("map")
        ),
    )


def test_a_textured_mesh_is_drawn_with_the_materials_of_its_file(cylinder_bot_world):
    marker = SemDTToRos2Converter.convert(textured_chair(cylinder_bot_world))

    assert marker.mesh_use_embedded_materials


def test_drawing_a_mesh_leaves_its_geometry_unloaded(cylinder_bot_world):
    """
    The marker refers to the mesh's file, so converting it must not keep the mesh's
    geometry and textures in memory.
    """
    mesh = textured_chair(cylinder_bot_world)

    SemDTToRos2Converter.convert(mesh)

    assert "mesh" not in vars(mesh)


# %% mesh files that move between machines


def map_origin(world) -> HomogeneousTransformationMatrix:
    """
    :return: The identity pose in the map of the given world.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        reference_frame=world.get_body_by_name("map")
    )


@pytest.fixture
def installed_mesh_file(tmp_path, monkeypatch) -> Path:
    """
    :return: A mesh file in the share directory of the package ``mesh_package``, which
        is registered in an ament prefix.
    """
    share = tmp_path / "share"
    (share / "ament_index" / "resource_index" / "packages").mkdir(parents=True)
    (share / "ament_index" / "resource_index" / "packages" / "mesh_package").touch()
    mesh_file = share / "mesh_package" / "meshes" / "jeroen_cup.stl"
    mesh_file.parent.mkdir(parents=True)
    shutil.copy(
        Path(files("semantic_digital_twin")).parent.parent
        / "resources"
        / "stl"
        / "jeroen_cup.stl",
        mesh_file,
    )
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path))
    return mesh_file


def test_a_mesh_in_a_package_is_published_by_package_uri(
    cylinder_bot_world, installed_mesh_file
):
    mesh = Mesh(
        filename=str(installed_mesh_file), origin=map_origin(cylinder_bot_world)
    )

    marker = SemDTToRos2Converter.convert(mesh)

    assert marker.mesh_resource == "package://mesh_package/meshes/jeroen_cup.stl"


def test_a_mesh_outside_any_package_is_published_by_file_uri(
    cylinder_bot_world, tmp_path
):
    mesh_file = tmp_path / "jeroen_cup.stl"
    shutil.copy(
        Path(files("semantic_digital_twin")).parent.parent
        / "resources"
        / "stl"
        / "jeroen_cup.stl",
        mesh_file,
    )
    mesh = Mesh(filename=str(mesh_file), origin=map_origin(cylinder_bot_world))

    marker = SemDTToRos2Converter.convert(mesh)

    assert marker.mesh_resource == f"file://{mesh_file}"


def test_a_package_uri_mesh_is_found_in_the_local_install(
    cylinder_bot_world, installed_mesh_file
):
    marker = SemDTToRos2Converter.convert(
        Mesh(filename=str(installed_mesh_file), origin=map_origin(cylinder_bot_world))
    )

    mesh = Ros2ToSemDTConverter.convert(marker, cylinder_bot_world)

    assert mesh.filename == str(installed_mesh_file)


# %% nothing converts the given object


def test_a_message_no_converter_handles_is_reported_as_unconvertible_to_semdt():
    with pytest.raises(CannotConvertRos2ToSemDTError):
        Ros2ToSemDTConverter.get_to_converter(object())


def test_an_object_no_converter_handles_is_reported_as_unconvertible_to_ros2():
    with pytest.raises(CannotConvertSemDTToRos2Error):
        SemDTToRos2Converter.get_to_converter(object())


# %% discovery in a process that imported nothing else


def run_cold_import_script(script_name: str) -> str:
    """
    :param script_name: Name of the script beside this module to run.
    :return: What the script printed, with surrounding whitespace removed.
    """
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / script_name)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_converters_are_found_without_anything_having_imported_them():
    found = run_cold_import_script("converter_lookup_from_cold_import.py")

    assert found.splitlines() == [
        LaserScanToSemDTConverter.__name__,
        BoxToRos2Converter.__name__,
    ]


def test_a_message_is_serializable_once_its_serializer_module_is_imported():
    serialized_type = run_cold_import_script(
        "message_serialization_from_cold_import.py"
    )

    assert serialized_type == get_full_class_name(geometry_msgs.Point)
