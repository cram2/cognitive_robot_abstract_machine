"""
Tests for the ConclusionAid base: both optional hooks default to a no-op (return ``None``),
and subclasses may override either or both.
"""

from __future__ import annotations

import unittest

from krrood.entity_query_language.rdr.aid import ConclusionAid


class TestConclusionAidDefaults(unittest.TestCase):
    def test_base_hooks_return_none(self):
        aid = ConclusionAid()
        self.assertIsNone(aid.present(context=None))
        self.assertIsNone(aid.suggest(context=None))

    def test_present_only_subclass(self):
        class InfoAid(ConclusionAid):
            def present(self, context):
                return "see the picture"

        aid = InfoAid()
        self.assertEqual(aid.present(context=None), "see the picture")
        self.assertIsNone(aid.suggest(context=None))

    def test_suggest_only_subclass(self):
        class Suggester(ConclusionAid):
            def suggest(self, context):
                return "guess"

        aid = Suggester()
        self.assertIsNone(aid.present(context=None))
        self.assertEqual(aid.suggest(context=None), "guess")


if __name__ == "__main__":
    unittest.main()
