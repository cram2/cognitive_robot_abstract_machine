"""Tests for the :mod:`llmr.bridge` gateway package.

The bridge is the only llmr layer allowed to import krrood internals —
these tests exercise each gateway (``introspect``, ``match_reader``,
``world_reader``) against lightweight PyCRAM-free fixtures so downstream
code can trust the plain-data snapshots returned from here.
"""
