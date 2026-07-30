"""Tests for naming-convention helpers in ``krrood.code_generation.naming``."""

from __future__ import annotations

import pytest

from krrood.code_generation.naming import (
    camel_case_to_lower_camel_case,
    to_camel_case,
    to_snake_case,
)


class TestToCamelCase:
    """Tests for :func:`to_camel_case`."""

    @pytest.mark.parametrize(
        "input_string, expected",
        [
            ("distance", "Distance"),
            ("my_func", "MyFunc"),
            ("my_distance_rdr", "MyDistanceRdr"),
            ("already_camel", "AlreadyCamel"),
            ("hello_world_test", "HelloWorldTest"),
            ("a", "A"),
            ("x", "X"),
            ("a_b_c", "ABC"),
            ("", ""),
        ],
    )
    def test_converts_snake_to_camel(self, input_string: str, expected: str) -> None:
        """Every underscore-separated segment is title-cased."""
        assert to_camel_case(input_string) == expected

    def test_returns_str(self) -> None:
        """The result is always a ``str``."""
        assert isinstance(to_camel_case("distance"), str)

    def test_result_is_nonempty_for_nonempty_input(self) -> None:
        """A non-empty input never yields an empty string."""
        assert to_camel_case("x") != ""


class TestToSnakeCase:
    """Tests for :func:`to_snake_case`."""

    @pytest.mark.parametrize(
        "input_string, expected",
        [
            ("MyFunc", "my_func"),
            ("Distance", "distance"),
            ("HelloWorldTest", "hello_world_test"),
            ("XMLParser", "xml_parser"),
        ],
    )
    def test_converts_camel_to_snake(self, input_string: str, expected: str) -> None:
        assert to_snake_case(input_string) == expected


class TestCamelCaseToLowerCamelCase:
    """Tests for :func:`camel_case_to_lower_camel_case`."""

    @pytest.mark.parametrize(
        "input_string, expected",
        [
            ("Distance", "distance"),
            ("MyDistance", "myDistance"),
            ("ABC", "aBC"),
            ("", ""),
        ],
    )
    def test_lowercases_first_character(self, input_string: str, expected: str) -> None:
        assert camel_case_to_lower_camel_case(input_string) == expected
