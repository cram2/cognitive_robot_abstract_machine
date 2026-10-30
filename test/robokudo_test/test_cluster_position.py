# %% imports

from __future__ import annotations

import numpy
import open3d
import pytest
from py_trees.common import Status

from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import CloudAnnotation, PositionAnnotation
from robokudo.types.scene import ObjectHypothesis

# %% centroid estimation


@pytest.mark.parametrize("analysis_scope", [ObjectHypothesis, CloudAnnotation])
@pytest.mark.parametrize("point_count", [0, 1])
def test_position_requires_nonempty_cloud(
    analysis_scope: type[ObjectHypothesis] | type[CloudAnnotation],
    point_count: int,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Create a position exactly when the selected cloud contains a point.
    """
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(numpy.ones((point_count, 3)))
    hypothesis = ObjectHypothesis(points=cloud)
    if analysis_scope is CloudAnnotation:
        hypothesis.points = open3d.geometry.PointCloud()
        hypothesis.annotations.append(CloudAnnotation(points=cloud))
    pipeline = Pipeline()
    annotator = ClusterPositionAnnotator()
    annotator.descriptor.parameters.analysis_scope = analysis_scope
    pipeline.add_child(annotator)
    pipeline.setup()
    pipeline.cas.set(CASViews.CLOUD, cloud)
    pipeline.cas.annotations.append(hypothesis)

    assert annotator.update() == Status.SUCCESS

    positions = [
        position
        for position in hypothesis.annotations
        if isinstance(position, PositionAnnotation)
    ]
    assert len(positions) == point_count
    if point_count:
        numpy.testing.assert_array_equal(positions[0].translation, cloud.points[0])
