"""Code generation utilities for the CRAM krrood package.

This package provides general infrastructure for generating Python source code:
naming conventions, import extraction, type-hint serialisation, source-code
formatting, and a :class:`~krrood.code_generation.generator.CodeGenerator` base
class for Jinja2-based generation.

Import from the specific submodule (for example
:mod:`krrood.code_generation.naming` or :mod:`krrood.code_generation.imports`);
the package root deliberately exposes no symbols.
"""
