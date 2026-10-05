"""
Tests for repository.py: reading a GitHub repository out of a reference or a remote URL.
"""

import pytest

from basstler.repository import MalformedRepositoryError, Repository
from basstler.setup_steps import Host

REPOSITORY = Repository("an-owner", "a-repository")
"""
The repository every well-formed reference and remote URL in these tests names.
"""

# %% repository references


def test_repository_splits_a_reference_into_owner_and_name():
    assert Repository.parse(REPOSITORY.full_name) == REPOSITORY


def test_repository_round_trips_through_the_form_github_uses():
    assert str(Repository.parse(REPOSITORY.full_name)) == REPOSITORY.full_name


@pytest.mark.parametrize("malformed", ["no-separator", "/no-owner", "no-name/"])
def test_repository_rejects_a_reference_that_is_not_owner_and_name(malformed: str):
    """
    A half-parsed reference would silently target the wrong repository.
    """
    with pytest.raises(MalformedRepositoryError):
        Repository.parse(malformed)


@pytest.mark.parametrize(
    "url",
    [
        f"{Host.GITHUB.url}/{REPOSITORY.full_name}.git",
        f"{Host.GITHUB.url}/{REPOSITORY.full_name}",
        f"git@{Host.GITHUB}:{REPOSITORY.full_name}.git",
        f"http://127.0.0.1:41729/git/{REPOSITORY.full_name}",
    ],
)
def test_repository_reads_the_owner_and_name_from_a_remote_url(url: str):
    """
    Every shape a fork remote takes names the same repository.

    A cloud session reaches GitHub through a local proxy, so the URL it sees shares
    neither host nor scheme with the one a laptop clone has.
    """
    assert Repository.from_remote_url(url) == REPOSITORY


@pytest.mark.parametrize("malformed", ["", f"{Host.GITHUB.url}/only-one-segment"])
def test_repository_rejects_a_remote_url_naming_no_repository(malformed: str):
    with pytest.raises(MalformedRepositoryError):
        Repository.from_remote_url(malformed)
