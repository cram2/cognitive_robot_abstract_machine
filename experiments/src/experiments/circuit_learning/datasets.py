"""
The fourteen datasets of the circuit learning benchmark: binary datasets with their
published splits, continuous datasets as the masked autoregressive flow paper prepared
them, and mixed datasets from OpenML, downloaded on first use and kept in a cache.
"""

from __future__ import annotations

# %% imports
import io
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from random_events.set import Set
from random_events.variable import Continuous, Symbolic, Variable
from sklearn.datasets import fetch_openml
from sortedcontainers import SortedSet

# %% names


class DatasetName(StrEnum):
    """
    The datasets of the benchmark.
    """

    NLTCS = "nltcs"
    MSNBC = "msnbc"
    PLANTS = "plants"
    AD = "ad"
    POWER = "power"
    GAS = "gas"
    HEPMASS = "hepmass"
    MINIBOONE = "miniboone"
    BSDS300 = "bsds300"
    DIAMONDS = "diamonds"
    ALLSTATE_CLAIMS_SEVERITY = "allstate_claims_severity"
    NYC_GREEN_TAXI = "nyc_green_taxi"
    KDD_CUP_99 = "kdd_cup_99"
    ZURICH_TRANSPORT_DELAYS = "zurich_transport_delays"


# %% splits


@dataclass(kw_only=True)
class BenchmarkDataset:
    """
    A dataset split into training, validation and test rows, with symbolic columns as
    strings and every other column as floats.
    """

    name: DatasetName
    """
    The name of the dataset.
    """

    train: pd.DataFrame
    """
    The training rows.
    """

    validation: pd.DataFrame
    """
    The rows methods that stop on held-out rows stop on.
    """

    test: pd.DataFrame
    """
    The test rows.
    """

    variables: SortedSet = field(init=False)
    """
    One variable per column: symbolic over the values of all splits, or continuous.
    """

    def __post_init__(self) -> None:
        """
        Infer :attr:`variables` from the values of all splits.
        """
        variables: list[Variable] = []
        for column in self.train.columns:
            if pd.api.types.is_float_dtype(self.train[column]):
                variables.append(Continuous(column))
                continue
            values = sorted(
                set(self.train[column])
                | set(self.validation[column])
                | set(self.test[column])
            )
            variables.append(Symbolic(name=column, domain=Set.from_iterable(values)))
        self.variables = SortedSet(variables)

    @property
    def is_symbolic(self) -> bool:
        """
        :return: Whether every variable is symbolic.
        """
        return all(isinstance(variable, Symbolic) for variable in self.variables)

    def encoded(self, *, frame: pd.DataFrame) -> np.ndarray:
        """
        :param frame: Rows of one of the splits.
        :return: The rows with every symbolic value replaced by the hash of its domain
            element, the columns in the order of :attr:`variables`.
        """
        columns = []
        for variable in self.variables:
            values = frame[variable.name]
            if isinstance(variable, Symbolic):
                columns.append(values.map(hash).to_numpy(dtype=float))
            else:
                columns.append(values.to_numpy(dtype=float))
        return np.stack(columns, axis=1)


# %% sources


@dataclass(kw_only=True)
class DatasetSource(ABC):
    """
    Where the rows of a dataset come from.
    """

    cache_directory: Path
    """
    The directory downloaded files are kept in.
    """

    @abstractmethod
    def load(self) -> BenchmarkDataset:
        """
        :return: The dataset, downloaded if it is not in the cache yet.
        """

    def cached_bytes(self, *, url: str, filename: str) -> bytes:
        """
        :param url: The address of the file.
        :param filename: The name of the file in the cache.
        :return: The content of the file, downloaded on first use.
        """
        path = self.cache_directory / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        return path.read_bytes()

    @staticmethod
    def float_frame(*, values: np.ndarray) -> pd.DataFrame:
        """
        :param values: Rows of continuous values.
        :return: The rows as a frame with one named float column per variable.
        """
        return pd.DataFrame(
            values.astype(float),
            columns=[f"x{index:03d}" for index in range(values.shape[1])],
        )


# %% binary


class BinaryDatasetFile(StrEnum):
    """
    The names of the binary datasets in the repository of :cite:`liu2021tractable`.
    """

    NLTCS = "nltcs"
    MSNBC = "msnbc"
    PLANTS = "plants"
    AD = "ad"


class BinarySplit(StrEnum):
    """
    The splits of a binary dataset, as their files are named.
    """

    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


@dataclass(kw_only=True)
class BinaryDatasetSource(DatasetSource):
    """
    A binary dataset of the Density-Estimation-Datasets repository, with its own splits.
    """

    dataset: BinaryDatasetFile
    """
    The dataset.
    """

    repository_url: str = (
        "https://raw.githubusercontent.com/UCLA-StarAI/Density-Estimation-Datasets/"
        "master/datasets"
    )
    """
    The address of the directory of the datasets.
    """

    def load(self) -> BenchmarkDataset:
        frames = {}
        for split in BinarySplit:
            filename = f"{self.dataset}.{split}.data"
            content = self.cached_bytes(
                url=f"{self.repository_url}/{self.dataset}/{filename}",
                filename=f"binary/{filename}",
            )
            values = np.loadtxt(io.BytesIO(content), delimiter=",", dtype=int)
            frames[split] = pd.DataFrame(
                values.astype(str),
                columns=[f"x{index:04d}" for index in range(values.shape[1])],
            )
        return BenchmarkDataset(
            name=DatasetName(self.dataset.value),
            train=frames[BinarySplit.TRAIN],
            validation=frames[BinarySplit.VALIDATION],
            test=frames[BinarySplit.TEST],
        )


# %% continuous


@dataclass(kw_only=True)
class MaskedAutoregressiveFlowSource(DatasetSource, ABC):
    """
    A continuous dataset prepared as :cite:`papamakarios2017masked` prepares it, read
    from the ``data`` directory of their archive at :attr:`archive_url`.
    """

    archive_url: str = "https://zenodo.org/record/1161203/files/data.tar.gz"
    """
    The address of the archive of the prepared datasets.
    """

    @property
    def data_directory(self) -> Path:
        """
        :return: The directory the archive was unpacked into.
        """
        return self.cache_directory / "maf" / "data"

    @abstractmethod
    def splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: The training, validation and test rows, as the paper's code makes them.
        """

    @property
    @abstractmethod
    def name(self) -> DatasetName:
        """
        :return: The name of the dataset.
        """

    def load(self) -> BenchmarkDataset:
        train, validation, test = self.splits()
        return BenchmarkDataset(
            name=self.name,
            train=self.float_frame(values=train.astype(np.float32)),
            validation=self.float_frame(values=validation.astype(np.float32)),
            test=self.float_frame(values=test.astype(np.float32)),
        )

    @staticmethod
    def split_off_tenths(
        *, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param data: All rows.
        :return: The last tenth as test rows, the last tenth of the rest as validation
            rows and the remaining rows for training, in that order reversed.
        """
        number_of_test_rows = int(0.1 * data.shape[0])
        test = data[-number_of_test_rows:]
        data = data[0:-number_of_test_rows]
        number_of_validation_rows = int(0.1 * data.shape[0])
        validation = data[-number_of_validation_rows:]
        train = data[0:-number_of_validation_rows]
        return train, validation, test

    @staticmethod
    def normalized_by_training_and_validation(
        *, train: np.ndarray, validation: np.ndarray, test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: The splits standardized by the mean and deviation of the training and
            validation rows together.
        """
        data = np.vstack((train, validation))
        mean = data.mean(axis=0)
        deviation = data.std(axis=0)
        return (
            (train - mean) / deviation,
            (validation - mean) / deviation,
            (test - mean) / deviation,
        )


@dataclass(kw_only=True)
class PowerSource(MaskedAutoregressiveFlowSource):
    """
    POWER, ``datasets/power.py`` of the paper's code.
    """

    @property
    def name(self) -> DatasetName:
        return DatasetName.POWER

    def splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        random_state = np.random.RandomState(42)
        data = np.load(self.data_directory / "power" / "data.npy")
        random_state.shuffle(data)
        number_of_rows = data.shape[0]
        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        voltage_noise = 0.01 * random_state.rand(number_of_rows, 1)
        gap_noise = 0.001 * random_state.rand(number_of_rows, 1)
        sub_metering_noise = random_state.rand(number_of_rows, 3)
        time_noise = np.zeros((number_of_rows, 1))
        data = data + np.hstack(
            (gap_noise, voltage_noise, sub_metering_noise, time_noise)
        )
        train, validation, test = self.split_off_tenths(data=data)
        return self.normalized_by_training_and_validation(
            train=train, validation=validation, test=test
        )


@dataclass(kw_only=True)
class GasSource(MaskedAutoregressiveFlowSource):
    """
    GAS, ``datasets/gas.py`` of the paper's code, from its pickle converted to
    ``ethylene_CO.csv``, since current pandas no longer reads that pickle.
    """

    correlation_threshold: float = 0.98
    """
    The correlation above which two columns count as duplicates.
    """

    @property
    def name(self) -> DatasetName:
        return DatasetName.GAS

    def splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = pd.read_csv(self.data_directory / "gas" / "ethylene_CO.csv")
        data = data.drop(columns=["Meth", "Eth", "Time"])
        correlated = self.correlated_counts(data=data)
        while np.any(correlated > 1):
            column = data.columns[np.where(correlated > 1)[0][0]]
            data = data.drop(columns=[column])
            correlated = self.correlated_counts(data=data)
        data = (data - data.mean()) / data.std()
        return self.split_off_tenths(data=data.to_numpy())

    def correlated_counts(self, *, data: pd.DataFrame) -> np.ndarray:
        """
        :param data: The columns.
        :return: For every column, the number of columns it is correlated with above
            :attr:`correlation_threshold`, itself included.
        """
        return (data.corr() > self.correlation_threshold).to_numpy().sum(axis=1)


@dataclass(kw_only=True)
class HepmassSource(MaskedAutoregressiveFlowSource):
    """
    HEPMASS, ``datasets/hepmass.py`` of the paper's code.
    """

    maximum_count_of_smallest_value: int = 5
    """
    How often the smallest value of a column may occur before the column is removed.
    """

    @property
    def name(self) -> DatasetName:
        return DatasetName.HEPMASS

    def splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        directory = self.data_directory / "hepmass"
        train = pd.read_csv(directory / "1000_train.csv", index_col=False)
        test = pd.read_csv(directory / "1000_test.csv", index_col=False)
        train = train[train[train.columns[0]] == 1].drop(columns=train.columns[0])
        test = test[test[test.columns[0]] == 1].drop(columns=test.columns[0])
        test = test.drop(columns=test.columns[-1])
        mean = train.mean()
        deviation = train.std()
        train = ((train - mean) / deviation).to_numpy()
        test = ((test - mean) / deviation).to_numpy()
        kept = [
            index
            for index, feature in enumerate(train.T)
            if Counter(feature)[feature.min()] <= self.maximum_count_of_smallest_value
        ]
        train = train[:, kept]
        test = test[:, kept]
        number_of_validation_rows = int(train.shape[0] * 0.1)
        return (
            train[0:-number_of_validation_rows],
            train[-number_of_validation_rows:],
            test,
        )


@dataclass(kw_only=True)
class MiniBooNESource(MaskedAutoregressiveFlowSource):
    """
    MINIBOONE, ``datasets/miniboone.py`` of the paper's code.
    """

    @property
    def name(self) -> DatasetName:
        return DatasetName.MINIBOONE

    def splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.load(self.data_directory / "miniboone" / "data.npy")
        train, validation, test = self.split_off_tenths(data=data)
        return self.normalized_by_training_and_validation(
            train=train, validation=validation, test=test
        )


@dataclass(kw_only=True)
class BSDS300Source(MaskedAutoregressiveFlowSource):
    """
    BSDS300, ``datasets/bsds300.py`` of the paper's code, from its splits converted to
    ``train.npy``, ``validation.npy`` and ``test.npy``.
    """

    @property
    def name(self) -> DatasetName:
        return DatasetName.BSDS300

    def splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        directory = self.data_directory / "BSDS300"
        return (
            np.load(directory / "train.npy"),
            np.load(directory / "validation.npy"),
            np.load(directory / "test.npy"),
        )


# %% mixed


class OpenMLIdentifier(IntEnum):
    """
    The OpenML data identifiers of the mixed datasets.
    """

    DIAMONDS = 42225
    ALLSTATE_CLAIMS_SEVERITY = 42571
    NYC_GREEN_TAXI = 42729
    KDD_CUP_99 = 42746
    ZURICH_TRANSPORT_DELAYS = 45045


@dataclass(kw_only=True)
class OpenMLSource(DatasetSource):
    """
    A dataset of OpenML with symbolic and numeric columns, split at random into training
    and test rows, the validation rows taken from the training rows.
    """

    dataset: OpenMLIdentifier
    """
    The dataset.
    """

    test_fraction: float = 0.2
    """
    The fraction of the rows that become test rows.
    """

    validation_fraction: float = 0.1
    """
    The fraction of the training rows that become validation rows.
    """

    seed: int = 69
    """
    The seed of the split.
    """

    def load(self) -> BenchmarkDataset:
        frame = self.typed_frame()
        order = np.random.default_rng(self.seed).permutation(len(frame))
        number_of_test_rows = int(self.test_fraction * len(frame))
        test = frame.iloc[order[:number_of_test_rows]]
        rest = order[number_of_test_rows:]
        number_of_validation_rows = int(self.validation_fraction * len(rest))
        validation = frame.iloc[rest[:number_of_validation_rows]]
        train = frame.iloc[rest[number_of_validation_rows:]]
        return BenchmarkDataset(
            name=DatasetName[self.dataset.name],
            train=train.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
            test=test.reset_index(drop=True),
        )

    def typed_frame(self) -> pd.DataFrame:
        """
        :return: All rows as published, with numeric columns as floats, every other
            column as strings, and rows with a missing value left out, since a circuit
            has no value for them.
        """
        frame = fetch_openml(
            data_id=int(self.dataset),
            as_frame=True,
            data_home=str(self.cache_directory / "openml"),
            parser="liac-arff",
        ).frame
        frame = frame.dropna(axis=1, how="all").dropna().reset_index(drop=True)
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].astype("int64").astype(float)
            elif pd.api.types.is_numeric_dtype(
                frame[column]
            ) and not pd.api.types.is_bool_dtype(frame[column]):
                frame[column] = frame[column].astype(float)
            else:
                frame[column] = frame[column].astype(str)
        return frame


# %% all sources


@dataclass(kw_only=True)
class DatasetSources:
    """
    The source of every dataset of the benchmark.
    """

    cache_directory: Path
    """
    The directory downloaded files are kept in.
    """

    def find(self, *, name: DatasetName) -> DatasetSource:
        """
        :param name: The name of a dataset.
        :return: Its source.
        """
        if name.name in BinaryDatasetFile.__members__:
            return BinaryDatasetSource(
                cache_directory=self.cache_directory,
                dataset=BinaryDatasetFile[name.name],
            )
        if name.name in OpenMLIdentifier.__members__:
            return OpenMLSource(
                cache_directory=self.cache_directory,
                dataset=OpenMLIdentifier[name.name],
            )
        continuous: dict[DatasetName, type[MaskedAutoregressiveFlowSource]] = {
            DatasetName.POWER: PowerSource,
            DatasetName.GAS: GasSource,
            DatasetName.HEPMASS: HepmassSource,
            DatasetName.MINIBOONE: MiniBooNESource,
            DatasetName.BSDS300: BSDS300Source,
        }
        return continuous[name](cache_directory=self.cache_directory)
