# %% Imports
import pytest
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.io.storage_reader_interface import StorageReaderInterface


class TestCrDescriptorFactory(object):
    def test_cr_descriptor_factory_valid_interface_name(self) -> None:
        """
        Test the creation of a valid cr descriptor config.
        """
        descriptor = CollectionReaderDescriptorFactory.create_descriptor("mongo")

        assert type(descriptor.parameters.camera_config) == MongoCameraConfig
        assert descriptor.parameters.camera_interface_type is StorageReaderInterface

    def test_cr_descriptor_factory_invalid_interface_name(self) -> None:
        """
        Test handling of an invalid cr descriptor config.
        """
        with pytest.raises(ValueError):
            CollectionReaderDescriptorFactory.create_descriptor(
                "invalid_interface_name"
            )

    def test_cr_descriptor_factory_valid_additional_config(self) -> None:
        """
        Test passing of valid additional config parameters to the camera config through
        descriptor factory.
        """
        loop = False
        db_name = "other_db_name"

        descriptor = CollectionReaderDescriptorFactory.create_descriptor(
            "mongo",
            loop=loop,
            db_name=db_name,
        )

        assert type(descriptor.parameters.camera_config) == MongoCameraConfig
        assert descriptor.parameters.camera_interface_type is StorageReaderInterface

        assert descriptor.parameters.camera_config.loop == loop
        assert descriptor.parameters.camera_config.db_name == db_name

    def test_cr_descriptor_factory_invalid_additional_config(self) -> None:
        """
        Test handling of invalid additional config parameters to the camera config
        through descriptor factory.
        """
        with pytest.raises(TypeError):
            CollectionReaderDescriptorFactory.create_descriptor(
                "mongo", invalid_config_key="any_value"
            )
