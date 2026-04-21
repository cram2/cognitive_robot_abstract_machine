"""Shared test fixtures for llmr test suite.

This package centralises reusable domain objects used across bridge, world,
reasoning, and backend tests. Tests should import from the submodules rather
than redefining Symbol/dataclass stand-ins inline.

Layout:
  actions.py — PyCRAM-free action dataclasses (MockPickUpAction et al.).
  symbols.py — Symbol subclasses modelling bodies, annotations, manipulators.
  worlds.py  — pytest fixtures that populate SymbolGraph for grounding tests.
"""
