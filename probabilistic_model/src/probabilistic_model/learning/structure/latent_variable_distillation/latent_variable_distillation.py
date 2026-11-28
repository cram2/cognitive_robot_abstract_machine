"""
Latent variable distillation of :cite:`liu2023scaling` for hidden Chow-Liu tree
circuits: a masked autoencoder trained on the rows embeds every variable, the embeddings
of every variable are clustered into the states of its latent variable, and the
circuit's initial parameters are counted from the completed rows.
"""

from __future__ import annotations

# %% imports
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from krrood_compute_resources.devices import Device
from random_events.variable import Symbolic
from sklearn.cluster import KMeans

from probabilistic_model.learning.parameters.torch_circuit_parameter_learning import (
    TorchDeviceType,
)
from probabilistic_model.learning.structure.hidden_chow_liu_tree.hidden_chow_liu_tree import (
    HiddenChowLiuTree,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)


# %% masked autoencoder


@dataclass(kw_only=True)
class TransformerArchitecture:
    """
    The size of the transformer encoder of a :class:`TabularMaskedAutoencoder`.
    """

    width: int = 64
    """
    The size of a token.
    """

    layers: int = 2
    """
    The number of transformer layers.
    """

    heads: int = 4
    """
    The number of attention heads.
    """

    feedforward_width: int = 128
    """
    The size of the hidden layer of every transformer layer's feed-forward network.
    """


@dataclass(kw_only=True, eq=False, repr=False)
class TabularMaskedAutoencoder(torch.nn.Module):
    """
    A masked autoencoder over the variables of a table: every variable is one token, a
    transformer encodes the visible tokens together with a shared token for the hidden
    ones, and one head per variable reconstructs its value.
    """

    number_of_states: list[int]
    """
    The number of states of every symbolic variable, zero for a continuous one.
    """

    architecture: TransformerArchitecture
    """
    The size of the transformer encoder.
    """

    initial_position_scale: float = 0.02
    """
    The standard deviation of the random initial position embeddings.
    """

    def __post_init__(self) -> None:
        """
        Creates the embeddings, the encoder and the reconstruction heads.
        """
        super().__init__()
        width = self.architecture.width
        self.value_embeddings = torch.nn.ModuleList(
            (torch.nn.Embedding(states, width) if states else torch.nn.Linear(1, width))
            for states in self.number_of_states
        )
        self.position_embedding = torch.nn.Parameter(
            torch.randn(len(self.number_of_states), width) * self.initial_position_scale
        )
        self.mask_embedding = torch.nn.Parameter(torch.zeros(width))
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=width,
                nhead=self.architecture.heads,
                dim_feedforward=self.architecture.feedforward_width,
                batch_first=True,
            ),
            num_layers=self.architecture.layers,
            enable_nested_tensor=False,
        )
        self.reconstruction_heads = torch.nn.ModuleList(
            torch.nn.Linear(width, states if states else 1)
            for states in self.number_of_states
        )

    def embeddings(self, *, values: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        """
        :param values: The state index or the standardized value of every variable,
            shape (#rows, #variables).
        :param masked: Whether each value is hidden from the encoder.
        :return: The encoded token of every variable, shape (#rows, #variables, width).
        """
        tokens = torch.stack(
            [
                (
                    embedding(values[:, index].long())
                    if states
                    else embedding(values[:, index : index + 1])
                )
                for index, (embedding, states) in enumerate(
                    zip(self.value_embeddings, self.number_of_states)
                )
            ],
            dim=1,
        )
        tokens = torch.where(masked[..., None], self.mask_embedding, tokens)
        return self.encoder(tokens + self.position_embedding)

    def reconstruction_loss(
        self, *, values: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor:
        """
        :param values: The state index or the standardized value of every variable.
        :param masked: Whether each value is hidden from the encoder.
        :return: The mean loss of reconstructing the hidden values: cross entropy for
            symbolic and squared error for continuous variables.
        """
        encoded = self.embeddings(values=values, masked=masked)
        losses = []
        for index, (head, states) in enumerate(
            zip(self.reconstruction_heads, self.number_of_states)
        ):
            prediction = head(encoded[:, index])
            if states:
                loss = torch.nn.functional.cross_entropy(
                    prediction, values[:, index].long(), reduction="none"
                )
            else:
                loss = (prediction[:, 0] - values[:, index]) ** 2
            losses.append((loss * masked[:, index]).sum())
        return torch.stack(losses).sum() / masked.sum().clamp(min=1)


@dataclass(kw_only=True)
class MaskedAutoencoderTraining:
    """
    The architecture of a :class:`TabularMaskedAutoencoder` and how it is trained to
    reconstruct randomly hidden values.
    """

    architecture: TransformerArchitecture = field(
        default_factory=TransformerArchitecture
    )
    """
    The size of the autoencoder's transformer encoder.
    """

    mask_probability: float = 0.5
    """
    The probability that a value is hidden from the encoder during training.
    """

    epochs: int = 10
    """
    The number of passes over the rows.
    """

    batch_size: int = 256
    """
    The number of rows per step, also when the trained autoencoder embeds rows.
    """

    learning_rate: float = 1e-3
    """
    The learning rate of the Adam optimizer.
    """

    def trained_autoencoder(
        self, *, values: torch.Tensor, number_of_states: list[int], seed: int
    ) -> TabularMaskedAutoencoder:
        """
        :param values: The state index or the standardized value of every variable of
            every row, on the device to train on.
        :param number_of_states: The number of states of every symbolic variable, zero
            for a continuous one.
        :param seed: The seed of the initial weights, the dropout, the order of the rows
            and the hidden values; torch's global random state is left as it was.
        :return: The trained autoencoder, on the device of the values.
        """
        with torch.random.fork_rng(devices=[values.device] if values.is_cuda else []):
            torch.manual_seed(seed)
            model = TabularMaskedAutoencoder(
                number_of_states=number_of_states, architecture=self.architecture
            ).to(values.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            model.train()
            for _ in range(self.epochs):
                order = torch.randperm(len(values), device=values.device)
                for start in range(0, len(values), self.batch_size):
                    batch = values[order[start : start + self.batch_size]]
                    masked = (
                        torch.rand(batch.shape, device=values.device)
                        < self.mask_probability
                    )
                    optimizer.zero_grad()
                    model.reconstruction_loss(values=batch, masked=masked).backward()
                    optimizer.step()
        return model


@dataclass(kw_only=True)
class EmbeddingBatch:
    """
    The embeddings of consecutive rows with all values visible.
    """

    first_row: int
    """
    The index of the batch's first row among all rows.
    """

    embeddings: npt.NDArray[np.float32]
    """
    The encoded token of every variable of every row in the batch, shape (#rows in the
    batch, #variables, width).
    """


# %% latent variable distillation


@dataclass(kw_only=True)
class LatentVariableDistillation:
    """
    Builds a hidden Chow-Liu tree circuit whose initial parameters are counted from
    latent states that a masked autoencoder trained on the same rows assigns.
    """

    structure: HiddenChowLiuTree
    """
    The hidden Chow-Liu tree, whose variables, hidden states and seed the distillation
    shares.
    """

    autoencoder: MaskedAutoencoderTraining = field(
        default_factory=MaskedAutoencoderTraining
    )
    """
    The autoencoder and its training.
    """

    clustering_rows: int = 10000
    """
    The most rows, sampled without replacement, whose embeddings the clusters are fitted
    on; every row is assigned to the nearest cluster.
    """

    device: Device = Device.CENTRAL_PROCESSOR
    """
    The device the autoencoder trains and embeds on.
    """

    @property
    def number_of_states(self) -> list[int]:
        """
        :return: The number of states of every symbolic variable, zero for a continuous
            one.
        """
        return [
            len(variable.domain.simple_sets) if isinstance(variable, Symbolic) else 0
            for variable in self.structure.variables
        ]

    def build_circuit(
        self, *, data: npt.NDArray[np.float64]
    ) -> LayeredProbabilisticCircuit:
        """
        :param data: The rows, encoded as the circuits read them.
        :return: The hidden Chow-Liu tree circuit with the initial numpy parameters
            counted from the distilled latent states.
        """
        return self.structure.build_circuit(
            data=data, latent_states=self.latent_states(data=data)
        )

    def autoencoder_inputs(self, *, data: npt.NDArray[np.float64]) -> torch.Tensor:
        """
        :param data: The rows, encoded as the circuits read them.
        :return: The rows with every symbolic value replaced by its state index and
            every continuous value by its standard score, on :attr:`device`.
        """
        values = np.array(data, dtype=float)
        for index, variable in enumerate(self.structure.variables):
            column = values[:, index]
            if isinstance(variable, Symbolic):
                values[:, index] = SymbolicEncoding(
                    variable=variable
                ).indices_of_hashes(column)
            else:
                values[:, index] = (column - column.mean()) / max(
                    float(column.std()), GaussianLayer.minimum_scale
                )
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=torch.device(TorchDeviceType.from_device(device=self.device)),
        )

    def latent_states(self, *, data: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """
        :param data: The rows, encoded as the circuits read them.
        :return: The cluster of the embedding of every variable in every row, with all
            values visible, shape (#rows, #variables).
        """
        seed = self.structure.seed
        values = self.autoencoder_inputs(data=data)
        model = self.autoencoder.trained_autoencoder(
            values=values, number_of_states=self.number_of_states, seed=seed
        )
        model.eval()
        sample = np.random.default_rng(seed).choice(
            len(values), size=min(self.clustering_rows, len(values)), replace=False
        )
        sample_embeddings = np.concatenate(
            [
                batch.embeddings
                for batch in self.batch_embeddings(
                    model=model, values=values[torch.as_tensor(sample)]
                )
            ]
        )
        clusterings = [
            KMeans(n_clusters=self.structure.hidden_states, random_state=seed).fit(
                sample_embeddings[:, index]
            )
            for index in range(len(self.structure.variables))
        ]
        latent_states = np.empty(values.shape, dtype=np.int64)
        for batch in self.batch_embeddings(model=model, values=values):
            rows = slice(batch.first_row, batch.first_row + len(batch.embeddings))
            for index, clustering in enumerate(clusterings):
                latent_states[rows, index] = clustering.predict(
                    batch.embeddings[:, index]
                )
        return latent_states

    def batch_embeddings(
        self, *, model: TabularMaskedAutoencoder, values: torch.Tensor
    ) -> Iterator[EmbeddingBatch]:
        """
        :param model: The trained autoencoder.
        :param values: The state index or the standardized value of every variable of
            every row.
        :return: The embeddings of the rows, in batches of the autoencoder's batch size.
        """
        with torch.no_grad():
            for start in range(0, len(values), self.autoencoder.batch_size):
                batch = values[start : start + self.autoencoder.batch_size]
                yield EmbeddingBatch(
                    first_row=start,
                    embeddings=model.embeddings(
                        values=batch, masked=torch.zeros_like(batch, dtype=torch.bool)
                    )
                    .cpu()
                    .numpy(),
                )
