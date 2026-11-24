"""
Validate CAS ownership and annotator traversal across nested behaviour trees.
"""

# %% Imports
import py_trees
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.testing import (
    FailingAnnotator,
    FakeCollectionReaderAnnotator,
    SlowAnnotator,
)
from robokudo.pipeline import Pipeline


def create_test_tree():
    root = py_trees.composites.Sequence("Sequence", memory=True)

    # for annotator in [CollectionReader(), ThreadedAnnotator("ImagePreprocessor"), CaffeAnnotator("CaffeAnnotator")]:
    for annotator in [
        FakeCollectionReaderAnnotator(),
        FailingAnnotator("FailingAnnotator"),
        SlowAnnotator("CaffeAnnotator"),
    ]:
        root.add_child(annotator)

    return root


# %% Pipeline traversal
class TestPipeline(object):
    """
    Check parent lookup and traversal boundaries for pipeline-owned annotators.
    """

    def test_find_cas_direct(self) -> None:
        """
        Direct children access the CAS belonging to their parent pipeline.
        """
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator(name="1")
        annotator2 = BaseAnnotator(name="2")
        root.add_child(annotator1)
        root.add_child(annotator2)

        assert root.cas == annotator1.get_cas()

    def test_find_cas_from_deeper_leaf(self) -> None:
        """
        Annotators below ordinary composites still resolve their owning pipeline.
        """
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator(name="1")
        annotator2 = BaseAnnotator(name="2")
        annotatorInSelector = BaseAnnotator(name="annotatorInSelector")

        selector = py_trees.composites.Selector("Selector", memory=True)
        success_after_two = py_trees.behaviours.Dummy(
            name="After Two",
        )
        always_running = py_trees.behaviours.Running(name="Running")
        selector.add_children([success_after_two, always_running, annotatorInSelector])

        root.add_child(annotator1)
        root.add_child(annotator2)
        root.add_child(selector)

        assert root.cas == annotatorInSelector.get_cas()

    def test_get_annotators_nested(self) -> None:
        """
        Annotator traversal includes leaves inside ordinary nested sequences.
        """
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator(name="1")
        annotator3 = BaseAnnotator(name="3")

        sub_sequence = py_trees.composites.Sequence("SubSequence", memory=True)
        annotator2 = BaseAnnotator(name="2")
        sub_sequence.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(sub_sequence)
        root.add_child(annotator3)

        assert root.get_annotators() == [annotator1, annotator2, annotator3]

    def test_pipeline_children1(self) -> None:
        """
        Child traversal includes nested composites and their annotator leaves.
        """
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator(name="1")
        annotator3 = BaseAnnotator(name="3")

        sub_sequence = py_trees.composites.Sequence("SubSequence", memory=True)
        annotator2 = BaseAnnotator(name="2")
        sub_sequence.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(sub_sequence)
        root.add_child(annotator3)

        assert root.pipeline_children() == [
            annotator1,
            annotator2,
            sub_sequence,
            annotator3,
        ]

    def test_pipeline_children2(self) -> None:
        """
        Child traversal excludes annotators owned by another pipeline.
        """
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator(name="1")
        annotator3 = BaseAnnotator(name="3")

        sub_pipeline = Pipeline("Subsequence")
        annotator2 = BaseAnnotator(name="2")
        sub_pipeline.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(sub_pipeline)
        root.add_child(annotator3)

        assert root.pipeline_children() == [annotator1, annotator3]

    def test_pipeline_children3(self) -> None:
        """
        A nested pipeline is omitted regardless of its position among siblings.
        """
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator(name="1")
        annotator3 = BaseAnnotator(name="3")

        # Pipeline children will not be fetched in nested RK Pipelines
        sub_pipeline = Pipeline("Subsequence")
        annotator2 = BaseAnnotator(name="2")
        sub_pipeline.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(annotator3)
        root.add_child(sub_pipeline)

        assert root.pipeline_children() == [annotator1, annotator3]
