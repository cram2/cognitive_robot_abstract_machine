"""
The models behind
:class:`~semantic_digital_twin.part_finding.vision_language.VisionLanguagePartLocator`:
Qwen3-VL puts boxes around a part, SAM 2 masks it inside them.

Needs the ``vision`` extra of this package. The weights are downloaded from Hugging Face
on first use, at the revisions pinned here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from functools import cached_property

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
    ProcessorMixin,
    Qwen3VLForConditionalGeneration,
    Sam2Model,
    Sam2Processor,
)

from semantic_digital_twin.part_finding.vision_language import (
    BoxSegmenter,
    PartBoxDetector,
    PartBoxPrompt,
)
from semantic_digital_twin.spatial_computations.mesh_rendering import RenderedViews

logger = logging.getLogger(__name__)

# %% which models


@dataclass(frozen=True)
class PretrainedWeights:
    """
    A model's weights on Hugging Face, at a fixed revision.
    """

    repository: str
    """
    The Hugging Face repository holding the weights.
    """

    revision: str
    """
    The commit of the repository the weights are taken from.
    """


class QwenModel(PretrainedWeights, Enum):
    """
    The sizes of Qwen3-VL the handle finder benchmark measured.
    """

    QWEN3_VL_2B = (
        "Qwen/Qwen3-VL-2B-Instruct",
        "89644892e4d85e24eaac8bacfd4f463576704203",
    )
    QWEN3_VL_4B = (
        "Qwen/Qwen3-VL-4B-Instruct",
        "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    )
    QWEN3_VL_8B = (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    )


class SegmentationModel(PretrainedWeights, Enum):
    """
    The SAM 2 models a box can be segmented with.
    """

    SAM2_1_HIERA_LARGE = (
        "facebook/sam2.1-hiera-large",
        "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    )


class Quantization(StrEnum):
    """
    How a model's weights are stored on the device.
    """

    NONE = "none"
    """
    In bfloat16 on a GPU and in float32 on a CPU.
    """

    FOUR_BIT = "4bit"
    """
    In 4 bits, which fits the 4B model next to SAM 2 into 8 GB of GPU memory.
    """


class Device(StrEnum):
    """
    Where a model runs.
    """

    CUDA = "cuda"
    CPU = "cpu"

    @property
    def number_type(self) -> torch.dtype:
        """
        :return: The floating point type models compute in on this device; bfloat16 is
            much slower than float32 on a CPU.
        """
        return torch.bfloat16 if self is Device.CUDA else torch.float32


# %% the detector


@dataclass
class QwenBoxDetector(PartBoxDetector):
    """
    Asks Qwen3-VL for boxes around a part, about all views in one batch.
    """

    model: QwenModel = QwenModel.QWEN3_VL_8B
    """
    The size of Qwen3-VL to ask.
    """

    quantization: Quantization = Quantization.NONE
    """
    How the model's weights are stored.
    """

    device: Device = Device.CUDA
    """
    Where the model runs.
    """

    prompt: PartBoxPrompt = field(default_factory=PartBoxPrompt)
    """
    The question asked about each view.
    """

    maximum_answer_tokens: int = 120
    """
    How long an answer may get.
    """

    coordinate_range: float = 1000.0
    """
    The width and height of an image in the coordinates Qwen3-VL answers in.
    """

    @cached_property
    def processor(self) -> ProcessorMixin:
        """
        :return: Turns views and the question into the model's input, and its output
            into text.
        """
        processor = AutoProcessor.from_pretrained(
            self.model.repository, revision=self.model.revision
        )
        processor.tokenizer.padding_side = "left"
        return processor

    @cached_property
    def network(self) -> PreTrainedModel:
        """
        :return: The model, loaded onto :attr:`device`.
        """
        weights = self.model
        if self.quantization is Quantization.FOUR_BIT:
            return Qwen3VLForConditionalGeneration.from_pretrained(
                weights.repository,
                revision=weights.revision,
                device_map=self.device.value,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.device.number_type,
                ),
            ).eval()
        return (
            Qwen3VLForConditionalGeneration.from_pretrained(
                weights.repository,
                revision=weights.revision,
                dtype=self.device.number_type,
            )
            .to(self.device.value)
            .eval()
        )

    def detect(
        self, views: RenderedViews, part_name: str, whole_name: str
    ) -> list[NDArray[np.float64]]:
        question = self.prompt.question(part_name, whole_name)
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.fromarray(image).convert("RGB"),
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for image in views.images
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.network.device)
        with torch.no_grad():
            generated = self.network.generate(
                **inputs, max_new_tokens=self.maximum_answer_tokens, do_sample=False
            )
        answers = self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        _, height, width = views.images.shape
        image_size = np.array([width, height, width, height], dtype=np.float64)
        return [
            self._boxes_in(answer) / self.coordinate_range * image_size
            for answer in answers
        ]

    def load(self) -> None:
        logger.info("Loaded %s on %s", self.model.repository, self.network.device)

    @staticmethod
    def _boxes_in(answer: str) -> NDArray[np.float64]:
        """
        :param answer: What the model answered.
        :return: Every group of four numbers in brackets in the answer, as rows.
        """
        number = r"\s*([\d.]+)\s*"
        groups = re.findall(rf"\[{number},{number},{number},{number}\]", answer)
        return np.array(groups, dtype=np.float64).reshape(-1, 4)


# %% the segmenter


@dataclass
class Sam2BoxSegmenter(BoxSegmenter):
    """
    Masks the part inside each box with SAM 2, one view at a time.
    """

    model: SegmentationModel = SegmentationModel.SAM2_1_HIERA_LARGE
    """
    The SAM 2 model to segment with.
    """

    device: Device = Device.CUDA
    """
    Where the model runs.
    """

    @cached_property
    def processor(self) -> Sam2Processor:
        """
        :return: Turns a view and its boxes into the model's input, and its output
            into masks.
        """
        return Sam2Processor.from_pretrained(
            self.model.repository, revision=self.model.revision
        )

    @cached_property
    def network(self) -> Sam2Model:
        """
        :return: The model, loaded onto :attr:`device`.
        """
        return (
            Sam2Model.from_pretrained(
                self.model.repository, revision=self.model.revision
            )
            .to(self.device.value)
            .eval()
        )

    def segment(
        self, views: RenderedViews, boxes: list[NDArray[np.float64]]
    ) -> NDArray[np.bool_]:
        masks = np.zeros(views.faces.shape, dtype=bool)
        for mask, image, view_boxes in zip(masks, views.images, boxes):
            if len(view_boxes) == 0:
                continue
            inputs = self.processor(
                images=Image.fromarray(image).convert("RGB"),
                input_boxes=[view_boxes.tolist()],
                return_tensors="pt",
            ).to(self.device.value)
            with torch.no_grad():
                outputs = self.network(**inputs, multimask_output=False)
            box_masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]
            mask[:] = box_masks[:, 0].numpy().any(axis=0)
        return masks

    def load(self) -> None:
        logger.info("Loaded %s on %s", self.model.repository, self.network.device)
