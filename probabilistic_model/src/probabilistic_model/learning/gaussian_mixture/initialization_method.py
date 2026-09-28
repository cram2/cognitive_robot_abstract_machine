"""
The ways StepMix can start fitting a mixture.
"""

import enum


class InitializationMethod(enum.StrEnum):
    """
    How StepMix initializes the responsibilities, with StepMix's names as values.
    """

    K_MEANS = "kmeans"
    """
    From the clusters of k-means.
    """

    RANDOM = "random"
    """
    At random.
    """
