"""Turn semantic objects into a feature dataframe for confidence-aware evaluation.

The out-of-distribution check needs the features of an object as a row of a
dataframe. This module bridges the semantic objects of a world to that
dataframe: it converts each object to its data access object, hands the
collection to the :class:`FeatureExtractor`, and keeps the mass and the object
class as the features the confidence model is learned on.
"""

from __future__ import annotations

import enum

import pandas as pd
from krrood.ormatic.data_access_objects.dao import to_dao
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from typing_extensions import Any, List


class ObjectClass(enum.Enum):
    """The semantic-annotation classes the confidence model distinguishes.

    Enum members carry the class name as their value, so an instance's class is
    looked up with ``ObjectClass(type(instance).__name__)``.
    """

    CUP = "Cup"
    POT = "Pot"


def extract_feature_dataframe(objects: List[Any]) -> pd.DataFrame:
    """Extract the mass and class of each object as a feature dataframe.

    Each object is converted to its data access object so that the
    :class:`FeatureExtractor` can read its mapped attributes, and its own
    ``preprocess_dataframe`` is run on that extracted dataframe before any column
    is selected, converting any boolean or enum-typed extracted attribute into a
    JPT-compatible column. The mass is kept under a stable column name and the
    object class is added as an :class:`ObjectClass` column, while the remaining
    extracted attributes are dropped.

    :param objects: The semantic objects whose features are extracted.
    :return: One row per object with a mass and a class column.
    """
    data_access_objects = [to_dao(instance) for instance in objects]
    extractor = FeatureExtractor.from_instances(data_access_objects)
    extracted = extractor.create_dataframe(data_access_objects)
    extracted = extractor.preprocess_dataframe(extracted)

    mass_column = next(name for name in extracted.columns if name.endswith(".mass"))
    dataframe = pd.DataFrame(
        {
            "mass": extracted[mass_column].to_numpy(),
            "class": [ObjectClass(type(instance).__name__) for instance in objects],
        }
    )
    return dataframe
