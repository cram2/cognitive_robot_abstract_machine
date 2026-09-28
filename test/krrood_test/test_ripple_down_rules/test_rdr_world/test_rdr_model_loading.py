from __future__ import annotations

import pytest

from krrood.ripple_down_rules.exceptions import RDRLoadError
from krrood.ripple_down_rules.rdr import GeneralRDR

# %% the cause of a failed load

MISSING_MODEL_NAME = "a_model_that_was_never_saved"
"""
The name of a model nothing saves, so loading it always fails.
"""


def test_load_chains_the_error_that_prevented_it(tmp_path):
    """
    A failed load carries the error that caused it, so a caller is told why it failed.
    """
    with pytest.raises(Exception) as reading_the_model:
        GeneralRDR.from_python(str(tmp_path / MISSING_MODEL_NAME))
    with pytest.raises(RDRLoadError) as loading_the_model:
        GeneralRDR.load(str(tmp_path), model_name=MISSING_MODEL_NAME)
    assert type(loading_the_model.value.__cause__) is type(reading_the_model.value)


# %% where a saved model may live

SAVE_DIRECTORY_NAME = "outside_any_package"
"""
The directory a model is saved into, chosen so that no ancestor of it is a package.
"""


def test_model_saved_outside_a_package_is_loadable(
    drawer_cabinet_rdr, handles_and_containers_world, tmp_path
):
    """
    A model is loadable from wherever it was saved, not only from inside a package.
    """
    save_directory = str(tmp_path / SAVE_DIRECTORY_NAME)
    model_name = drawer_cabinet_rdr.save(save_directory)
    loaded_rdr = GeneralRDR.load(save_directory, model_name=model_name)
    assert loaded_rdr.classify(
        handles_and_containers_world
    ) == drawer_cabinet_rdr.classify(handles_and_containers_world)
