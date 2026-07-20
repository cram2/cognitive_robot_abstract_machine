from __future__ import annotations

import numpy as np
import py_trees
import pytest
from py_trees.blackboard import Blackboard

from robokudo.annotators.semantic_world_connector import (
    SemanticDigitalTwinConnector,
)
from robokudo.cas import CAS, CASViews
from robokudo.pipeline import Pipeline
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    PoseAnnotation,
    StampedPoseAnnotation,
)
from robokudo.types.scene import ObjectHypothesis
from robokudo import world as rk_world
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def _make_hypothesis(
    *,
    roi: tuple[int, int, int, int] = (10, 15, 20, 25),
    translation: tuple[float, float, float] = (0.0, 0.0, 1.0),
    source: str = "test_source",
) -> ObjectHypothesis:
    hypothesis = ObjectHypothesis()

    x, y, width, height = roi
    hypothesis.roi.roi.pos.x = x
    hypothesis.roi.roi.pos.y = y
    hypothesis.roi.roi.width = width
    hypothesis.roi.roi.height = height
    hypothesis.roi.mask = np.ones((height, width), dtype=np.uint8)

    pose = PoseAnnotation()
    pose.source = source
    pose.translation = list(translation)
    pose.rotation = [0.0, 0.0, 0.0, 1.0]
    hypothesis.annotations.append(pose)

    bbox = BoundingBox3DAnnotation()
    bbox.source = source
    bbox.pose.translation = list(translation)
    bbox.pose.rotation = [0.0, 0.0, 0.0, 1.0]
    bbox.x_length = 0.1
    bbox.y_length = 0.2
    bbox.z_length = 0.3
    hypothesis.annotations.append(bbox)

    return hypothesis


def _attach_connector_to_pipeline(
    connector: SemanticDigitalTwinConnector, cas: CAS
) -> Pipeline:
    pipeline = Pipeline("SemanticWorldConnectorTestPipeline")
    pipeline.add_child(connector)
    pipeline.cas = cas
    return pipeline


def _set_camera_to_world_transform(
    cas: CAS,
    *,
    pos_x: float = 0.0,
    pos_y: float = 0.0,
    pos_z: float = 0.0,
) -> None:
    sem_world = rk_world.world_instance()
    world_body = sem_world.get_body_by_name(cas.world_frame)
    camera_body = sem_world.get_body_by_name(cas.camera_frame)
    cas.camera_to_world_transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=pos_z,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
        reference_frame=world_body,
        child_frame=camera_body,
    )


def _stamped_world_poses(
    hypothesis: ObjectHypothesis,
) -> list[StampedPoseAnnotation]:
    return [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, StampedPoseAnnotation)
        and annotation.source == SemanticDigitalTwinConnector.__name__
    ]


@pytest.fixture
def connector() -> SemanticDigitalTwinConnector:
    return SemanticDigitalTwinConnector()


@pytest.fixture
def semdt_cas() -> CAS:
    rk_world.get_object_belief_states().clear()
    rk_world.init_world_with_entity_tracker()
    rk_world.setup_world_for_camera_frame(world_frame="map", camera_frame="camera")

    cas = CAS()
    cas.world_frame = "map"
    cas.camera_frame = "camera"
    _set_camera_to_world_transform(cas)

    yield cas

    rk_world.get_object_belief_states().clear()
    rk_world.init_world_with_entity_tracker()


class TestSemanticDigitalTwinConnector:
    def test_no_existing_beliefs_creates_belief_for_each_hypothesis(
        self,
        connector: SemanticDigitalTwinConnector,
        semdt_cas: CAS,
    ) -> None:
        hypotheses = [
            _make_hypothesis(roi=(5, 5, 10, 10)),
            _make_hypothesis(roi=(25, 25, 10, 10)),
        ]

        associated_hypotheses = connector.associate_hypotheses_with_beliefs(
            hypotheses, [], semdt_cas
        )

        assert [hypothesis for hypothesis, _ in associated_hypotheses] == hypotheses
        assert len(rk_world.get_object_belief_states()) == 2
        for hypothesis, belief in associated_hypotheses:
            assert belief.latest_hypothesis is hypothesis
            assert belief.body.parent_connection is not None
            assert len(belief.body.visual.shapes) == 1
            stamped_poses = _stamped_world_poses(hypothesis)
            assert len(stamped_poses) == 1
            assert stamped_poses[0].frame == "map"

    def test_high_similarity_updates_match_and_creates_beliefs_for_unmatched_hypotheses(
        self,
        connector: SemanticDigitalTwinConnector,
        semdt_cas: CAS,
    ) -> None:
        previous_hypothesis = _make_hypothesis(translation=(0.0, 0.0, 1.0))
        existing_belief = connector.associate_hypotheses_with_beliefs(
            [previous_hypothesis], [], semdt_cas
        )[0][1]
        assert len(_stamped_world_poses(previous_hypothesis)) == 1

        _set_camera_to_world_transform(semdt_cas, pos_x=1.0)
        # Same world pose as the previous hypothesis, seen from a translated camera.
        matched_hypothesis = _make_hypothesis(
            roi=(10, 10, 12, 12), translation=(-1.0, 0.0, 1.0)
        )
        unmatched_hypothesis = _make_hypothesis(
            roi=(80, 80, 12, 12), translation=(2.0, 0.0, 1.0)
        )
        _attach_connector_to_pipeline(connector, semdt_cas)

        associated_hypotheses = connector.associate_hypotheses_with_beliefs(
            [matched_hypothesis, unmatched_hypothesis],
            [existing_belief],
            semdt_cas,
        )

        assert len(rk_world.get_object_belief_states()) == 2
        assert existing_belief.latest_hypothesis is matched_hypothesis
        assert associated_hypotheses[0] == (matched_hypothesis, existing_belief)
        assert associated_hypotheses[1][0] is unmatched_hypothesis
        assert associated_hypotheses[1][1] is not existing_belief

    def test_low_similarity_assignment_creates_new_belief_instead_of_updating(
        self,
        connector: SemanticDigitalTwinConnector,
        semdt_cas: CAS,
    ) -> None:
        previous_hypothesis = _make_hypothesis()
        existing_belief = connector.associate_hypotheses_with_beliefs(
            [previous_hypothesis], [], semdt_cas
        )[0][1]
        hypothesis = _make_hypothesis(translation=(5.0, 0.0, 1.0))
        connector.descriptor.parameters.confidence_threshold = 0.99
        _attach_connector_to_pipeline(connector, semdt_cas)

        associated_hypotheses = connector.associate_hypotheses_with_beliefs(
            [hypothesis], [existing_belief], semdt_cas
        )

        assert len(rk_world.get_object_belief_states()) == 2
        assert existing_belief.latest_hypothesis is previous_hypothesis
        assert associated_hypotheses[0][0] is hypothesis
        assert associated_hypotheses[0][1] is not existing_belief
        assert associated_hypotheses[0][1].latest_hypothesis is hypothesis

    def test_add_world_pose_annotations_skips_hypotheses_without_transform(
        self,
        connector: SemanticDigitalTwinConnector,
    ) -> None:
        cas = CAS()
        hypothesis = _make_hypothesis()

        connector.add_world_pose_annotations([hypothesis], cas)

        assert _stamped_world_poses(hypothesis) == []

    def test_update_reads_cas_associates_beliefs_and_publishes_visualization(
        self,
        connector: SemanticDigitalTwinConnector,
        semdt_cas: CAS,
    ) -> None:
        blackboard = Blackboard()
        blackboard.set(
            "annotator_output_pipeline_map_buffer", AnnotatorOutputPerPipelineMap()
        )

        pipeline = Pipeline("SemanticWorldConnectorTestPipeline")
        pipeline.add_child(connector)
        pipeline.setup()
        pipeline.cas = semdt_cas

        color_image = np.zeros((80, 100, 3), dtype=np.uint8)
        hypothesis = _make_hypothesis(roi=(20, 25, 30, 20))
        pipeline.cas.set(CASViews.COLOR_IMAGE, color_image)
        pipeline.cas.annotations.append(hypothesis)

        status = connector.compute()

        output_image = connector.get_annotator_output_struct().image
        assert status is py_trees.common.Status.SUCCESS
        assert len(rk_world.get_object_belief_states()) == 1
        assert len(_stamped_world_poses(hypothesis)) == 1
        assert not np.array_equal(output_image, color_image)
        assert np.array_equal(pipeline.cas.get(CASViews.COLOR_IMAGE), color_image)
