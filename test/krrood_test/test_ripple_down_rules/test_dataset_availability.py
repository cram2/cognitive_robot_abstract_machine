import unittest
from unittest.mock import patch

from ucimlrepo import DatasetNotFoundError

from . import datasets


class DatasetServerUnavailableTestCase(unittest.TestCase):
    """
    Verifies that a dataset-server outage is treated as an unavailable
    dataset instead of crashing the caller, so dependent test modules can
    skip themselves via ``unittest.skipIf(len(zoo_cases) == 0, ...)``
    instead of failing to collect.
    """

    def test_get_dataset_returns_none_when_server_reports_dataset_not_found(self):
        with patch.object(
            datasets, "fetch_ucirepo", side_effect=DatasetNotFoundError("boom")
        ):
            self.assertIsNone(datasets.get_dataset(111))

    def test_get_dataset_returns_none_when_server_is_unreachable(self):
        with patch.object(
            datasets, "fetch_ucirepo", side_effect=ConnectionError("boom")
        ):
            self.assertIsNone(datasets.get_dataset(111))

    def test_load_zoo_dataset_skips_gracefully_when_server_unavailable(self):
        with patch.object(
            datasets, "fetch_ucirepo", side_effect=DatasetNotFoundError("boom")
        ):
            cases, targets = datasets.load_zoo_dataset()
        self.assertEqual(cases, [])
        self.assertEqual(targets, [])
