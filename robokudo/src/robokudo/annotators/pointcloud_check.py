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

from timeit import default_timer

import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class PointcloudCheckAnnotator(BaseAnnotator):
    """Check if the CASViews.Cloud contains more than X points.

    .. warning::
       Will raise exception if point cloud not found in CAS.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for point cloud checking."""

        class Parameters:
            """Parameters for configuring point cloud validation."""

            def __init__(self) -> None:
                #
                self.point_threshold: int = 100
                """Decision boundary for: if CASViews.CLOUD has less than this amount of points"""

                self.status_below_threshold: Status = Status.FAILURE
                """Status when below threshold"""

                self.status_above_threshold: Status = Status.SUCCESS
                """Status when above threshold"""

                self.raise_on_failure: bool = True
                """Whether to raise exception on failure"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "PointcloudcloudCheckAnnotator",
        descriptor: "PointcloudCheckAnnotator.Descriptor" = Descriptor(),
    ):
        """Initialize the point cloud checker.

        :param name: Name of this annotator instance, defaults to "PointcloudcloudCheckAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)

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
        :raises Exception: If point count fails threshold and raise_on_failure is True
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        assert isinstance(cloud, o3d.geometry.PointCloud)
        point_count = len(cloud.points)
        if point_count < self.descriptor.parameters.point_threshold:
            if (
                self.descriptor.parameters.status_below_threshold
                == py_trees.Status.FAILURE
                and self.descriptor.parameters.raise_on_failure
            ):
                raise Exception(
                    f"Scene Pointcloud size({point_count}) is below "
                    f"threshold of {self.descriptor.parameters.point_threshold}"
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
                self.descriptor.parameters.status_above_threshold
                == py_trees.Status.FAILURE
                and self.descriptor.parameters.raise_on_failure
            ):
                raise Exception(
                    f"Scene Pointcloud size({point_count}) is above "
                    f"threshold of {self.descriptor.parameters.point_threshold}"
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
