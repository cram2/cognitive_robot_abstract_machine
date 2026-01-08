import pytest
from pycram.filter import Butterworth


def test_initialization_with_default_values():
    filter = Butterworth()
    assert filter.order == 4
    assert filter.cutoff == 10
    assert filter.fs == 60


def test_initialization_with_custom_values():
    filter = Butterworth(order=2, cutoff=5, fs=30)
    assert filter.order == 2
    assert filter.cutoff == 5
    assert filter.fs == 30


def test_filter_data_with_default_values():
    filter = Butterworth()
    data = [1, 2, 3, 4, 5]
    filtered_data = filter.filter(data)
    assert len(filtered_data) == len(data)


def test_filter_data_with_custom_values():
    filter = Butterworth(order=2, cutoff=5, fs=30)
    data = [1, 2, 3, 4, 5]
    filtered_data = filter.filter(data)
    assert len(filtered_data) == len(data)


def test_filter_empty_data():
    filter = Butterworth()
    data = []
    filtered_data = filter.filter(data)
    assert filtered_data.tolist() == data


def test_filter_single_value_data():
    filter = Butterworth()
    data = [1]
    filtered_data = filter.filter(data)
    expected_filtered_data = 0.026077721701092293  # The expected filtered value
    assert filtered_data.tolist()[0] == pytest.approx(expected_filtered_data)
