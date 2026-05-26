from robokudo.descriptors.camera_configs.config_kinect_robot import KinectCameraConfig
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
import pytest

from robokudo.descriptors.camera_configs.registry import CameraConfigRegistry


class TestCameraConfigRegistry(object):
    def test_camera_registry_valid_camera_name(self) -> None:
        """Test the creation of a valid camera config."""
        config = CameraConfigRegistry.create_config("mongo")
        assert type(config) == MongoCameraConfig

    def test_camera_registry_invalid_camera_name(self) -> None:
        """Test the creation of an invalid camera config."""
        with pytest.raises(ValueError):
            CameraConfigRegistry.create_config("invalid_camera_name")

    def test_camera_registry_valid_additional_config(self) -> None:
        """Test passing of valid additional config parameters to the camera config."""
        loop = False
        db_name = "other_db_name"

        config: MongoCameraConfig = CameraConfigRegistry.create_config(
            "mongo",
            loop=loop,
            db_name=db_name,
        )

        assert type(config) == MongoCameraConfig
        assert config.loop == loop
        assert config.db_name == db_name

    def test_camera_registry_invalid_additional_config(self) -> None:
        """Test handling of invalid additional config parameters to the camera config."""
        with pytest.raises(TypeError):
            CameraConfigRegistry.create_config("mongo", invalid_config_key="any_value")

    def test_camera_registry_register_multiple_times(self) -> None:
        """Test whether registration can be called multiple times without errors if there are no type conflicts."""
        CameraConfigRegistry.register_all()
        CameraConfigRegistry.register_all()

    def test_camera_registry_registration_conflict(self) -> None:
        """Test handling of conflicts in camera config registration."""
        type_before = CameraConfigRegistry._registry["mongo"]

        # Change type in registry
        CameraConfigRegistry._registry["mongo"] = KinectCameraConfig

        # Try to register MongoCameraConfig again
        with pytest.raises(ValueError):
            CameraConfigRegistry.register_all()

        CameraConfigRegistry._registry["mongo"] = type_before
