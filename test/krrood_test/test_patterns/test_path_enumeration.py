from __future__ import annotations

from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration

# %% enumeration under test


class ConfigurationLocation(PathEnumeration):
    """
    Paths of a configuration directory and a file in it.
    """

    DIRECTORY = Path("configuration")
    """
    The configuration directory.
    """

    SETTINGS = Path("configuration/settings.toml")
    """
    The settings file inside it.
    """


# %% a member is its path


def test_member_equals_its_path() -> None:
    assert ConfigurationLocation.SETTINGS == ConfigurationLocation.SETTINGS.value


def test_member_and_its_path_are_one_key() -> None:
    assert {ConfigurationLocation.SETTINGS.value: True}[ConfigurationLocation.SETTINGS]


def test_name_is_the_last_component_of_the_path() -> None:
    assert (
        ConfigurationLocation.SETTINGS.name == ConfigurationLocation.SETTINGS.value.name
    )


def test_member_name_is_read_as_underscore_name() -> None:
    assert ConfigurationLocation.SETTINGS._name_ == "SETTINGS"


# %% derived paths


def test_joined_path_is_a_plain_path() -> None:
    joined = ConfigurationLocation.DIRECTORY / "other.toml"
    assert type(joined) is type(Path())
    assert joined == ConfigurationLocation.DIRECTORY.value / "other.toml"


def test_parent_is_a_plain_path_even_when_it_names_a_member() -> None:
    parent = ConfigurationLocation.SETTINGS.parent
    assert type(parent) is type(Path())
    assert parent == ConfigurationLocation.DIRECTORY.value
