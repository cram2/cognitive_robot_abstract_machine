"""
Tests for the analysis engine that perceives recorded files into a shared world.
"""

from __future__ import annotations

import pytest

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.analysis_engines.semdt_shared_world_from_files import (
    WORLD_T_CAMERA,
    AnalysisEngine,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.io.camera_interface import CameraInterface

# %% camera configuration


class TestCameraConfiguration:
    def test_recorded_frames_are_seen_from_the_fixed_camera_pose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Keeps the camera configuration real while nothing reads the recorded files.
        monkeypatch.setitem(
            CollectionReaderDescriptorFactory._camera_interface_types,
            "file_reader",
            CameraInterface,
        )

        pipeline = AnalysisEngine().implementation()

        [reader] = [
            child
            for child in pipeline.children
            if isinstance(child, CollectionReaderAnnotator)
        ]
        camera_config = reader.descriptor.parameters.camera_config
        assert camera_config.static_camera_transform_enabled
        assert camera_config.static_world_T_camera is WORLD_T_CAMERA
