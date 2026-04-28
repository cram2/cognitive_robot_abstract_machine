from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from py_trees.behaviour import Behaviour

"""
Analysis Engine interface for RoboKudo.

This module provides the interface that all Analysis Engines must implement.
Analysis Engines are responsible for defining behavior trees that implement
specific robotic tasks or pipelines.
"""


class AnalysisEngineInterface(Protocol):
    """Protocol defining the interface for Analysis Engines.

    This interface must be implemented by all Analysis Engines in RoboKudo.
    Each Analysis Engine provides a behavior tree implementation for a specific
    task or pipeline. The RoboKudo runner will load and execute these
    implementations.

    .. warning::
        Do not instantiate this class directly. Instead, create a subclass
        that implements this interface.

    See :mod:`robokudo.descriptors.analysis_engines` for example implementations.
    """

    def name(self) -> str:
        """Get the name of this Analysis Engine.

        :return: The unique name identifying this Analysis Engine
        """
        pass

    def implementation(self) -> Behaviour:
        """Get the behavior tree implementation for this Analysis Engine.

        :return: The root node of the behavior tree implementing this engine
        """
        pass


class SubtreeInterface(AnalysisEngineInterface):
    """
    Analysis Engines often contain reusable components which are used across different use cases.
    To represent these parts, use the SubtreeInterface and create a subclass.
    Similarly to an AnalysisEngine, return the py_trees.Behaviour in the implementation method.

    To provide semantic difference,
    """

    top_level: bool = False
