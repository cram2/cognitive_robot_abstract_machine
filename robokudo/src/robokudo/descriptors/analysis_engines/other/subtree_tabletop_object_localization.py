from py_trees.composites import Sequence

from robokudo.analysis_engine import SubtreeInterface
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor


class Subtree(SubtreeInterface):
    def implementation(self) -> Sequence:
        """
        Returns a sequence which will find objects on the biggest plane in available 3d data.
        """
        seq = Sequence("TT Object Localization", memory=True)
        seq.add_children(
            [
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
            ]
        )
        return seq
