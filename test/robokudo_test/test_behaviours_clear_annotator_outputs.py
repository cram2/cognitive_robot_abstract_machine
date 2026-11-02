import py_trees
import pytest

from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.behaviours.clear_annotator_outputs import ClearAnnotatorOutputs
from robokudo.exceptions import ParentPipelineMissing


class TestClearAnnotatorOutputs:
    def test_raises_without_parent_pipeline(
        self, annotator_output_buffer: AnnotatorOutputPerPipelineMap
    ):
        root = py_trees.composites.Sequence("Sequence", memory=True)
        clear_outputs = ClearAnnotatorOutputs()
        root.add_child(clear_outputs)

        with pytest.raises(ParentPipelineMissing):
            clear_outputs.update()

    def test_removing_one_of_two_instances_keeps_the_other(self):
        """
        Instances with the same name stay distinct children of a composite.
        """
        root = py_trees.composites.Sequence("Sequence", memory=True)
        first = ClearAnnotatorOutputs()
        second = ClearAnnotatorOutputs()
        root.add_children([first, second])

        root.remove_child(second)

        assert [child.id for child in root.children] == [first.id]
