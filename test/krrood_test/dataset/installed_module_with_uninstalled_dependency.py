"""
A module that is installed but cannot be imported, because a package it imports at
module level is not installed.

Mirrors a module of an optional part of a distribution, installed without the extra that
provides its dependencies.
"""

import uninstalled_optional_dependency
