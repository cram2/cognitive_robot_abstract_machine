from __future__ import annotations

import numpy as np

from robokudo.annotators.static_camera_transform import (
    StaticCameraTransformAnnotator,
)
from robokudo.pipeline import Pipeline

# %% static camera pose


class TestStaticCameraPose:
    def test_camera_is_placed_at_the_configured_pose_in_the_world_of_its_pipeline(
        self,
    ) -> None:
        annotator = StaticCameraTransformAnnotator()
        pipeline = Pipeline("Pipeline")
        pipeline.add_child(annotator)
        parameters = annotator.descriptor.parameters

        annotator.update()

        camera_to_world_transform = pipeline.cas.camera_to_world_transform
        assert camera_to_world_transform.child_frame is pipeline.world.get_body_by_name(
            parameters.camera_frame
        )
        assert np.allclose(
            camera_to_world_transform.to_np(), parameters.world_T_camera.to_np()
        )
