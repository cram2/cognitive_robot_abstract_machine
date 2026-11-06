"""
Verifies that a reader of the dataset cache never sees a part file the writer has not
finished, so parallel test workers that load the cache while another one fills it do not
fail with :class:`EOFError`.
"""

from __future__ import annotations

import pickle

import pandas as pd

from . import datasets

# %% concurrent cache access


def test_cache_read_while_saving_sees_no_partial_part(tmp_path, monkeypatch):
    cache_file = str(tmp_path / "zoo_dataset.pkl")
    dataset = datasets.ZooDataset(
        features=pd.DataFrame({"hair": [1, 0]}),
        targets=pd.DataFrame({"type": [1, 4]}),
        ids=pd.DataFrame({"animal_name": ["aardvark", "bass"]}),
    )
    datasets_read_during_save = []
    dump = pickle.dump

    def dump_while_another_worker_reads(obj, file) -> None:
        datasets_read_during_save.append(datasets.load_cached_dataset(cache_file))
        dump(obj, file)

    monkeypatch.setattr(datasets.pickle, "dump", dump_while_another_worker_reads)
    datasets.save_dataset_to_cache(dataset, cache_file)

    assert datasets_read_during_save == [None, None, None]
