import pytest

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import OmniDrive


@pytest.fixture()
def armar7() -> Armar7:
    world = URDFParser.from_file(Armar7.get_ros_file_path()).parse()
    return Armar7.from_world(world)


# %% mobile base bounding box


def test_armar7_mobile_base_bounding_box(armar7):
    """
    ``Dummy_Platform_link``, the Armar7 mobile base root, carries no geometry of its own
    in the URDF, which used to make :meth:`MobileBase.bounding_box` fail with a
    `ValueError` on an empty shape collection.
    """
    bounding_box = armar7.mobile_base.bounding_box

    assert bounding_box.min_x == pytest.approx(-0.375)
    assert bounding_box.max_x == pytest.approx(0.375)
    assert bounding_box.min_y == pytest.approx(-0.3725)
    assert bounding_box.max_y == pytest.approx(0.3675)
    assert bounding_box.min_z == pytest.approx(-0.029)
    assert bounding_box.max_z == pytest.approx(0.411)


# %% root connection type


def test_armar7_root_attaches_via_drive_connection(armar7_world_state_reset):
    """
    Armar7's root used to be ``Dummy_Platform_link``, one fixed joint below the URDF's
    actual root, which attached the robot to the world via a :class:`FixedConnection`
    instead of its :class:`OmniDrive`.

    Location sampling relies on setting
    ``robot.root.parent_connection.origin``, which only a drive connection supports.
    """
    armar7 = armar7_world_state_reset.get_semantic_annotations_by_type(Armar7)[0]

    assert isinstance(armar7.root.parent_connection, OmniDrive)

    armar7.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.0, 2.0, 0.0, yaw=0.5
    )
