"""
Verifies that a dataset-server outage is treated as an unavailable
dataset instead of crashing the caller, so dependent test modules can
skip themselves via ``@pytest.mark.skipif(len(zoo_cases) == 0, ...)``
instead of failing to collect.
"""

from ucimlrepo import DatasetNotFoundError

from . import datasets


def test_get_dataset_returns_none_when_server_reports_dataset_not_found(monkeypatch):
    def mock_fetch(*args, **kwargs):
        raise DatasetNotFoundError("boom")

    monkeypatch.setattr(datasets, "fetch_ucirepo", mock_fetch)
    assert datasets.get_dataset(111) is None


def test_get_dataset_returns_none_when_server_is_unreachable(monkeypatch):
    def mock_fetch(*args, **kwargs):
        raise ConnectionError("boom")

    monkeypatch.setattr(datasets, "fetch_ucirepo", mock_fetch)
    assert datasets.get_dataset(111) is None


def test_load_zoo_dataset_skips_gracefully_when_server_unavailable(monkeypatch):
    def mock_fetch(*args, **kwargs):
        raise DatasetNotFoundError("boom")

    monkeypatch.setattr(datasets, "fetch_ucirepo", mock_fetch)
    cases, targets = datasets.load_zoo_dataset()
    assert cases == []
    assert targets == []
