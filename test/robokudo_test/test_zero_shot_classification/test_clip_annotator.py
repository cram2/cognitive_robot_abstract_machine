"""
Tests for classifying images by their CLIP similarity to the vocabulary.

The CLIP model is replaced by a stand-in that embeds an image as its mean color and a
text as the color it names, so the expected classification of every input is known
without downloading model weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing_extensions import Sequence

import robokudo.annotators.clip_annotator
from robokudo.annotators.clip_annotator import CLIPAnnotator
from robokudo.types.annotation import Classification

from ._color_scenes import COLOR_BY_WORD, COLOR_VOCABULARY, rgb, solid_image

# %% color embedding stand-in for CLIP

CLIP_LOGIT_SCALE = 100.0
"""
The temperature CLIP's similarity logits are scaled with.
"""


UNNAMED_COLOR = np.array([1, 1, 1], dtype=np.uint8)
"""
The color a text naming none of the vocabulary colors is embedded as.
"""


def _named_color(text: str) -> np.ndarray:
    """
    :return: The color named by the vocabulary word contained in the given text, in RGB
        channel order.
    """
    return next(
        (rgb(color) for word, color in COLOR_BY_WORD.items() if word in text),
        UNNAMED_COLOR,
    )


class DeviceRecordingEncoding(BatchEncoding):
    """
    Model input stand-in that records the device it is moved to instead of moving its
    tensors, so any device can be requested without the hardware being present.

    The recorded device reaches the model as the ``device`` keyword.
    """

    def __init__(self, colors: torch.Tensor) -> None:
        super().__init__({"colors": colors, "device": torch.device("cpu")})

    def to(
        self, device: str | torch.device | int, *, non_blocking: bool = False
    ) -> DeviceRecordingEncoding:
        self["device"] = torch.device(device)
        return self


@dataclass
class ColorTokenizer:
    """
    Tokenizer stand-in that encodes each text as the color it names.
    """

    received_texts: list[str] = field(default_factory=list)
    """
    The texts this tokenizer was asked to encode.
    """

    def __call__(
        self, texts: list[str], padding: bool, return_tensors: str
    ) -> BatchEncoding:
        self.received_texts.extend(texts)
        colors = np.stack([_named_color(text) for text in texts])
        return DeviceRecordingEncoding(torch.tensor(colors, dtype=torch.float32))


@dataclass
class ColorProcessor:
    """
    CLIP processor stand-in that encodes each image as its mean color.
    """

    tokenizer: ColorTokenizer = field(default_factory=ColorTokenizer)
    """
    The tokenizer used for vocabulary texts.
    """

    def __call__(
        self,
        images: np.ndarray | Image.Image | Sequence[Image.Image],
        return_tensors: str,
    ) -> DeviceRecordingEncoding:
        image_list = images if isinstance(images, list) else [images]
        mean_colors = np.stack(
            [np.asarray(image).reshape(-1, 3).mean(axis=0) for image in image_list]
        )
        return DeviceRecordingEncoding(torch.tensor(mean_colors, dtype=torch.float32))


@dataclass
class ColorModelConfiguration:
    """
    Model configuration stand-in exposing the embedding size.
    """

    projection_dim: int = 3
    """
    Size of the shared image and text embedding: one dimension per color channel.
    """


@dataclass
class ColorEmbeddingModel:
    """
    CLIP model stand-in whose image and text embeddings are the encoded colors.
    """

    config: ColorModelConfiguration = field(default_factory=ColorModelConfiguration)
    """
    The model configuration.
    """

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    """
    The device the model runs on.
    """

    logit_scale: torch.Tensor = field(
        default_factory=lambda: torch.tensor(math.log(CLIP_LOGIT_SCALE))
    )
    """
    Logarithm of the temperature the similarity logits are scaled with.
    """

    def to(self, device: torch.device) -> ColorEmbeddingModel:
        self.device = device
        return self

    def get_text_features(
        self, colors: torch.Tensor, device: torch.device
    ) -> BaseModelOutputWithPooling:
        return self._embed(colors, device)

    def get_image_features(
        self, colors: torch.Tensor, device: torch.device
    ) -> BaseModelOutputWithPooling:
        return self._embed(colors, device)

    def _embed(
        self, colors: torch.Tensor, device: torch.device
    ) -> BaseModelOutputWithPooling:
        """
        :return: The encoded colors as embeddings.
        :raises RuntimeError: As torch does, if the inputs are on another device than
            the model.
        """
        if device != self.device:
            raise RuntimeError(
                f"Inputs on {device} cannot run on a model on {self.device}"
            )
        return BaseModelOutputWithPooling(pooler_output=colors.clone())


@dataclass
class ColorEmbeddingClip:
    """
    The stand-in model and processor handed out in place of pretrained CLIP weights.
    """

    model: ColorEmbeddingModel = field(default_factory=ColorEmbeddingModel)
    """
    The stand-in CLIP model.
    """

    processor: ColorProcessor = field(default_factory=ColorProcessor)
    """
    The stand-in CLIP processor.
    """


@pytest.fixture
def color_embedding_clip(monkeypatch) -> ColorEmbeddingClip:
    """
    Make the annotator load the color embedding stand-in instead of pretrained CLIP.
    """
    clip = ColorEmbeddingClip()
    monkeypatch.setattr(
        robokudo.annotators.clip_annotator.CLIPModel,
        "from_pretrained",
        lambda model_name: clip.model,
    )
    monkeypatch.setattr(
        robokudo.annotators.clip_annotator.CLIPProcessor,
        "from_pretrained",
        lambda model_name: clip.processor,
    )
    return clip


# %% classification setup


def _color_descriptor() -> CLIPAnnotator.Descriptor:
    """
    :return: A descriptor classifying against the color vocabulary on the CPU.
    """
    descriptor = CLIPAnnotator.Descriptor()
    descriptor.parameters.device = torch.device("cpu")
    descriptor.parameters.vocabulary = COLOR_VOCABULARY
    descriptor.parameters.save_top_k = len(COLOR_VOCABULARY)
    return descriptor


def _classify_solid_image(
    annotator: CLIPAnnotator, color: np.ndarray
) -> list[Classification]:
    """
    :return: The classifications of an image filled with the given CAS color.
    """
    return annotator.classify([Image.fromarray(solid_image(rgb(color)))])[0]


# %% construction


def test_annotator_without_descriptor_indexes_default_vocabulary(
    color_embedding_clip,
):
    """
    An annotator built without a descriptor falls back to the default parameters.
    """
    default_vocabulary = CLIPAnnotator.Descriptor().parameters.vocabulary

    annotator = CLIPAnnotator()

    assert annotator.vocabulary_index.ntotal == len(default_vocabulary)


def test_vocabulary_is_encoded_through_the_template(color_embedding_clip):
    descriptor = _color_descriptor()
    descriptor.parameters.vocabulary_template = "A photo of something {}"

    CLIPAnnotator(descriptor=descriptor)

    assert color_embedding_clip.processor.tokenizer.received_texts == [
        descriptor.parameters.vocabulary_template.format(word)
        for word in COLOR_VOCABULARY
    ]


def test_model_runs_on_the_configured_device(color_embedding_clip):
    descriptor = _color_descriptor()
    descriptor.parameters.device = torch.device("cuda", 0)

    annotator = CLIPAnnotator(descriptor=descriptor)
    _classify_solid_image(annotator, COLOR_BY_WORD["green"])

    assert color_embedding_clip.model.device == descriptor.parameters.device


# %% classification


def test_word_naming_image_color_ranks_first(color_embedding_clip):
    annotator = CLIPAnnotator(descriptor=_color_descriptor())

    classifications = _classify_solid_image(annotator, COLOR_BY_WORD["green"])

    assert classifications[0].classname == "green"


def test_each_image_of_a_batch_is_classified_by_its_own_color(color_embedding_clip):
    annotator = CLIPAnnotator(descriptor=_color_descriptor())
    images = [
        Image.fromarray(solid_image(rgb(COLOR_BY_WORD["red"]))),
        Image.fromarray(solid_image(rgb(COLOR_BY_WORD["blue"]))),
    ]

    classifications = annotator.classify(images)

    assert [
        image_classifications[0].classname for image_classifications in classifications
    ] == ["red", "blue"]


def test_classification_keeps_save_top_k_classifications(color_embedding_clip):
    descriptor = _color_descriptor()
    descriptor.parameters.save_top_k = 2
    annotator = CLIPAnnotator(descriptor=descriptor)

    classifications = _classify_solid_image(annotator, COLOR_BY_WORD["red"])

    assert len(classifications) == descriptor.parameters.save_top_k


def test_classifications_name_the_annotator_as_source(color_embedding_clip):
    annotator = CLIPAnnotator(descriptor=_color_descriptor())

    classifications = _classify_solid_image(annotator, COLOR_BY_WORD["red"])

    assert {classification.source for classification in classifications} == {
        annotator.name
    }


def test_softmax_confidences_over_whole_vocabulary_sum_to_one(color_embedding_clip):
    annotator = CLIPAnnotator(descriptor=_color_descriptor())

    classifications = _classify_solid_image(annotator, COLOR_BY_WORD["blue"])

    confidences = [classification.confidence for classification in classifications]
    assert sum(confidences) == pytest.approx(1.0)


def test_confidence_without_softmax_is_rescaled_cosine_similarity(
    color_embedding_clip,
):
    """
    Without softmax, a cosine similarity in [-1, 1] is reported as a confidence in
    [0, 1]: the matching color has similarity 1, every other color similarity 0.
    """
    descriptor = _color_descriptor()
    descriptor.parameters.use_softmax = False
    annotator = CLIPAnnotator(descriptor=descriptor)

    classifications = _classify_solid_image(annotator, COLOR_BY_WORD["blue"])

    matching_similarity, orthogonal_similarity = 1.0, 0.0
    confidence_by_word = {
        classification.classname: classification.confidence
        for classification in classifications
    }
    assert confidence_by_word == pytest.approx(
        {
            word: (
                (matching_similarity if word == "blue" else orthogonal_similarity) + 1.0
            )
            / 2.0
            for word in COLOR_VOCABULARY
        }
    )


def test_save_top_k_beyond_vocabulary_without_softmax_reports_each_word_once(
    color_embedding_clip,
):
    descriptor = _color_descriptor()
    descriptor.parameters.use_softmax = False
    descriptor.parameters.save_top_k = len(COLOR_VOCABULARY) + 2
    annotator = CLIPAnnotator(descriptor=descriptor)

    classifications = _classify_solid_image(annotator, COLOR_BY_WORD["blue"])

    assert sorted(
        classification.classname for classification in classifications
    ) == sorted(COLOR_VOCABULARY)
