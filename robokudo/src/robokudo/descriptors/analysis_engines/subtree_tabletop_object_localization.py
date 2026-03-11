import py_trees.composites

import robokudo.analysis_engine
import robokudo.pipeline
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def implementation(self) -> py_trees.composites.Sequence:
        """
        Returns a sequence which will find objects on the biggest plane in available 3d data.
        """
        seq = py_trees.composites.Sequence("TT Object Localization", memory=True)
        seq.add_children(
            [
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
            ]
        )
        return seq
