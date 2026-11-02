"""
Fixtures shared by the tests of the YOLO annotators.
"""

from __future__ import annotations

import pytest

from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap

# %% visualization output


@pytest.fixture(autouse=True)
def visualization_output(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> AnnotatorOutputPerPipelineMap:
    """
    Give every annotator under test a buffer to write its visualization output to.
    """
    return annotator_output_buffer
