"""Point cloud validation and threshold checking.

This module provides an annotator for:

* Validating point cloud size
* Configurable threshold checking
* Customizable status responses
* Optional error raising

The annotator supports:

* Configurable point count threshold
* Different status returns for above/below threshold
* Optional exception raising on failure
* Detailed logging of check results

.. note::
   Point cloud must be available in CAS under CASViews.CLOUD.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from timeit import default_timer

import open3d as o3d
from krrood.ormatic.utils import classproperty
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters
from robokudo.exceptions import PointCloudThresholdError, PointCloudThresholdRelation
from robokudo.utils.error_handling import catch_and_raise_to_blackboard

# %% PointcloudCheck parameters


@dataclass(kw_only=True)
class PointcloudCheckParameters(AnnotatorParameters):
    """Parameters for configuring point cloud validation."""

    point_threshold: int = 100
    """Decision boundary for: if CASViews.CLOUD has less than this amount of points"""

    status_below_threshold: Status = Status.FAILURE
    """Status when below threshold"""

    status_above_threshold: Status = Status.SUCCESS
    """Status when above threshold"""

    raise_on_failure: bool = True
    """Whether to raise exception on failure"""


# %% PointcloudCheck descriptor


@dataclass(kw_only=True)
class PointcloudCheckDescriptor(AnnotatorDescriptor[PointcloudCheckParameters]):
    """Declarative configuration of :class:`PointcloudCheckAnnotator`."""

    parameters: PointcloudCheckParameters = field(
        default_factory=PointcloudCheckParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% PointcloudCheck annotator


@dataclass(eq=False)
class PointcloudCheckAnnotator(BaseAnnotator[PointcloudCheckDescriptor]):
    """Check if the CASViews.Cloud contains more than X points.

    .. warning::
       Will raise exception if point cloud not found in CAS.
    """

    name: str = "PointcloudcloudCheckAnnotator"
    """Name of this node in the behaviour tree."""

    descriptor: PointcloudCheckDescriptor = field(
        default_factory=PointcloudCheckDescriptor
    )
    """Configuration supplied to this annotator instance."""

    @classproperty
    def Descriptor(cls) -> type[PointcloudCheckDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return PointcloudCheckDescriptor

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Check point cloud size against threshold.

        The method:

        * Loads point cloud from CAS
        * Counts number of points
        * Compares against threshold:
          * Below: Returns status_below_threshold
          * Above: Returns status_above_threshold
        * Optionally raises exceptions on failure
        * Logs detailed check results

        :return: Configured status based on point count
        :raises PointCloudThresholdError: If point count fails threshold and raise_on_failure is True
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        assert isinstance(cloud, o3d.geometry.PointCloud)
        point_count = len(cloud.points)
        if point_count < self.descriptor.parameters.point_threshold:
            if (
                self.descriptor.parameters.status_below_threshold == Status.FAILURE
                and self.descriptor.parameters.raise_on_failure
            ):
                raise PointCloudThresholdError(
                    point_count=point_count,
                    threshold=self.descriptor.parameters.point_threshold,
                    relation=PointCloudThresholdRelation.BELOW,
                )

            self.rk_logger.info(
                f"Scene Pointcloud size({point_count}) is below "
                f"threshold of {self.descriptor.parameters.point_threshold}"
            )
            end_timer = default_timer()
            self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
            return self.descriptor.parameters.status_below_threshold
        else:
            if (
                self.descriptor.parameters.status_above_threshold == Status.FAILURE
                and self.descriptor.parameters.raise_on_failure
            ):
                raise PointCloudThresholdError(
                    point_count=point_count,
                    threshold=self.descriptor.parameters.point_threshold,
                    relation=PointCloudThresholdRelation.ABOVE,
                )

            self.rk_logger.info(
                f"Scene Pointcloud size({point_count}) is above "
                f"threshold of {self.descriptor.parameters.point_threshold}"
            )
            end_timer = default_timer()
            self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
            return self.descriptor.parameters.status_above_threshold

        #
        #
        # end_timer = default_timer()
        # self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        # return py_trees.common.Status.SUCCESS
