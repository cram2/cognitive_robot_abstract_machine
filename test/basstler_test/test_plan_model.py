"""
Tests for plan_model.py: the statuses a plan's items carry.
"""

import pytest

from basstler.plan_model import ItemStatus

# %% stored spelling


@pytest.mark.parametrize("status", list(ItemStatus))
def test_a_status_value_reads_as_words(status):
    """
    The value is what a page shows, so it is spelled as words rather than as an
    identifier.
    """
    assert "_" not in status.value


@pytest.mark.parametrize(
    "underscore_spelling, status",
    [
        ("not_started", ItemStatus.NOT_STARTED),
        ("in_progress", ItemStatus.IN_PROGRESS),
    ],
)
def test_a_manifest_written_with_underscores_still_reads(underscore_spelling, status):
    """
    The literals are the spelling manifests were written in before the values took
    spaces, which is the data this reading exists for.
    """
    assert ItemStatus(underscore_spelling) is status


def test_an_unknown_status_is_still_refused():
    with pytest.raises(ValueError):
        ItemStatus("in-review")
