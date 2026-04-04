import os
from PySide6.QtGui import QColor
from typing import Optional


def get_theme_color(name: str, default: str) -> str:
    """
    Gets a color from the qt-material theme environment variables.

    :param name: The variable name (e.g. 'primaryColor', 'secondaryColor').
    :param default: The fallback color if the theme variable is not set.
    :return: The hex color string.
    """
    env_name = f"QTMATERIAL_{name.upper()}"
    return os.environ.get(env_name, default)


def get_primary_color(default: str = "#1de9b6") -> str:
    """Gets the primary theme color."""
    return get_theme_color("primaryColor", default)


def get_primary_light_color(default: str = "#6effe8") -> str:
    """Gets the primary light theme color."""
    return get_theme_color("primaryLightColor", default)


def get_secondary_color(default: str = "#232629") -> str:
    """Gets the secondary theme color."""
    return get_theme_color("secondaryColor", default)


def get_secondary_dark_color(default: str = "#31363b") -> str:
    """Gets the secondary dark theme color."""
    return get_theme_color("secondaryDarkColor", default)


def get_secondary_light_color(default: str = "#4f5b62") -> str:
    """Gets the secondary light theme color."""
    return get_theme_color("secondaryLightColor", default)


def get_primary_text_color(default: str = "#ffffff") -> str:
    """Gets the primary text theme color."""
    return get_theme_color("primaryTextColor", default)


def get_secondary_text_color(default: str = "#ffffff") -> str:
    """Gets the secondary text theme color."""
    return get_theme_color("secondaryTextColor", default)


def is_dark_theme() -> bool:
    """Checks if the current theme is a dark theme."""
    theme = os.environ.get("QTMATERIAL_THEME", "dark")
    return "dark" in theme.lower()
