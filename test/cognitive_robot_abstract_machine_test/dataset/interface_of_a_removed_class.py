"""
Stand-in for an ORM interface generated from a class its package no longer holds.

A generated interface reaches every class it maps as an attribute of the module holding
it, so a class that was renamed or removed after the build leaves the interface raising
:class:`AttributeError` on import.
"""

from __future__ import annotations

import importlib

MAPPED_MODULE_NAME = "mapped_module"
"""
Module of this interface's package whose classes it maps.
"""

package_name = __name__.split(".")[0]
"""
Package this interface belongs to, which is the one it maps.
"""

mapped_module = importlib.import_module(f"{package_name}.{MAPPED_MODULE_NAME}")
"""
The module this interface maps.
"""

MAPPED_CLASS = mapped_module.Drawer
"""
The class this interface maps, which its package no longer holds.
"""
