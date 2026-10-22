"""
Tests for the example analysis engine that perceives recorded files into a shared world.
"""

from __future__ import annotations

import pytest

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.examples.descriptors.analysis_engines.shared_world_from_files import (
    AnalysisEngine,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.utils.module_loader import ModuleLoader

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

        analysis_engine = AnalysisEngine()

        pipeline = analysis_engine.implementation()

        [reader] = [
            child
            for child in pipeline.children
            if isinstance(child, CollectionReaderAnnotator)
        ]
        camera_config = reader.descriptor.parameters.camera_config
        assert camera_config.static_camera_transform_enabled
        assert camera_config.static_world_T_camera is analysis_engine.world_T_camera


# %% loading


class TestLoading:
    def test_main_loads_the_example_from_the_examples_package(self) -> None:
        analysis_engine = ModuleLoader().load_ae(
            ros_pkg_name="robokudo.examples", module_name="shared_world_from_files"
        )

        # The loader reloads an imported module, so the class is a fresh copy.
        assert (
            type(analysis_engine).__module__,
            type(analysis_engine).__qualname__,
        ) == (AnalysisEngine.__module__, AnalysisEngine.__qualname__)
