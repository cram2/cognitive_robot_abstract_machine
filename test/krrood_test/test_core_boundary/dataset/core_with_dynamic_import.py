"""
A core module that loads a package by name, which no static import graph can see.
"""

import importlib

importlib.import_module("dynamically_imported")
