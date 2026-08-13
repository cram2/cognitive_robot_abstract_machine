import os

import pytest
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.io.storage_reader_interface import StorageReaderInterface
from robokudo.descriptors import CrDescriptorFactory


class TestCrDescriptorFactory(object):

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="module temporarily disabled until storage functionality is migrated to ormatic",
    )
    def test_cr_descriptor_factory_valid_interface_name(self) -> None:
        """Test the creation of a valid cr descriptor config."""
        descriptor = CrDescriptorFactory.create_descriptor("mongo")

        assert type(descriptor.parameters.camera_config) == MongoCameraConfig
        assert type(descriptor.parameters.camera_interface) == StorageReaderInterface

    def test_cr_descriptor_factory_invalid_interface_name(self) -> None:
        """Test handling of an invalid cr descriptor config."""
        with pytest.raises(ValueError):
            CrDescriptorFactory.create_descriptor("invalid_interface_name")

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="module temporarily disabled until storage functionality is migrated to ormatic",
    )
    def test_cr_descriptor_factory_valid_additional_config(self) -> None:
        """Test passing of valid additional config parameters to the camera config through descriptor factory."""
        loop = False
        db_name = "other_db_name"

        descriptor = CrDescriptorFactory.create_descriptor(
            "mongo",
            loop=loop,
            db_name=db_name,
        )

        assert type(descriptor.parameters.camera_config) == MongoCameraConfig
        assert type(descriptor.parameters.camera_interface) == StorageReaderInterface

        assert descriptor.parameters.camera_config.loop == loop
        assert descriptor.parameters.camera_config.db_name == db_name

    def test_cr_descriptor_factory_invalid_additional_config(self) -> None:
        """Test handling of invalid additional config parameters to the camera config through descriptor factory."""
        with pytest.raises(TypeError):
            CrDescriptorFactory.create_descriptor(
                "mongo", invalid_config_key="any_value"
            )
