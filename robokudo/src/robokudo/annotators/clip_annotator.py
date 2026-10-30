from __future__ import annotations

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.zero_shot_classification_annotator import (
    ZeroShotClassificationAnnotator,
)
from robokudo.types.annotation import Classification

if TYPE_CHECKING:
    import numpy.typing as npt


class CLIPAnnotator(ZeroShotClassificationAnnotator):
    """
    Zero-shot classification by the similarity of CLIP's image embedding to the CLIP
    text embedding of each vocabulary entry.
    """

    class Descriptor(ZeroShotClassificationAnnotator.Descriptor):
        """
        Descriptor for the CLIPAnnotator.
        """

        class Parameters(ZeroShotClassificationAnnotator.Descriptor.Parameters):
            """
            Parameters for the Descriptor.
            """

            def __init__(self) -> None:
                super().__init__()
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
                """
                Torch device to be used by the annotator.
                """
                self.model_name = "openai/clip-vit-large-patch14"
                """
                Name of the CLIP to use for feature extraction.
                """
                self.vocabulary_template: str = "A photo of a {}"
                """
                A template string that each vocabulary word is inserted into.
                """
                self.use_softmax: bool = True
                """
                Whether to apply softmax to the results of the analyzed image or
                hypothesis or not.
                """

    def __init__(
        self,
        name: str = "CLIPAnnotator",
        descriptor: CLIPAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name=name, descriptor=descriptor)

        self.rk_logger.debug(f"Starting to init {self.__class__.__name__}")

        self.clip_model = CLIPModel.from_pretrained(
            self.descriptor.parameters.model_name
        ).to(self.descriptor.parameters.device)
        """
        CLIP model used for feature extraction.
        """
        self.clip_preprocess = CLIPProcessor.from_pretrained(
            self.descriptor.parameters.model_name
        )
        """
        CLIP preprocessor used for input preprocessing.
        """
        self.clip_tokenizer = self.clip_preprocess.tokenizer
        """
        CLIP tokenizer used for text preprocessing.
        """
        self.vocabulary_index: faiss.IndexFlatIP = faiss.IndexFlatIP(
            self.clip_model.config.projection_dim
        )
        """
        A faiss index for text features used in similarity search.
        """
        with torch.no_grad():
            templated_texts = [
                self.descriptor.parameters.vocabulary_template.format(name)
                for name in self.descriptor.parameters.vocabulary
            ]
            inputs = self.clip_tokenizer(
                templated_texts,
                padding=True,
                return_tensors="pt",
            ).to(self.descriptor.parameters.device)
            text_features = self.clip_model.get_text_features(**inputs).pooler_output
            text_features /= torch.norm(text_features, p=2, dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
        self.vocabulary_index.add(text_features)

        self.rk_logger.debug(f"Initialized {self.__class__.__name__}")

    def _get_image_features(self, images: list[Image.Image]) -> npt.NDArray[np.float32]:
        """
        Get normalized image features for the given images.

        :param images: The images to get features from.
        :return: The features of each image.
        """
        with torch.inference_mode():
            inputs = self.clip_preprocess(images=images, return_tensors="pt").to(
                self.clip_model.device
            )
            crop_features = self.clip_model.get_image_features(**inputs).pooler_output
            crop_features /= torch.norm(crop_features, p=2, dim=-1, keepdim=True)
            crop_features = crop_features.cpu().numpy()
        return crop_features

    def _candidate_count(self) -> int:
        """
        Number of vocabulary entries to retrieve from the index per analyzed image.

        With softmax, every entry is needed to normalize the confidences; without it,
        only the kept ones, but never more than the vocabulary holds.
        """
        vocabulary_size = len(self.descriptor.parameters.vocabulary)
        if self.descriptor.parameters.use_softmax:
            return vocabulary_size
        return min(self.descriptor.parameters.save_top_k, vocabulary_size)

    def _search_index(
        self,
        features: npt.NDArray[np.float32],
        k: int,
        use_softmax: bool,
        save_top_k: int,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """
        Search the faiss index using the given features.

        :param features: The features to search the index with.
        :param k: The number of candidates to retrieve from the index for each feature.
        :param use_softmax: Whether to apply a softmax to the results of each feature.
        :param save_top_k: How many results to keep after applying softmax.
        :return: A tuple of (similarities, class indices)
        """
        class_similarities, class_indices = self.vocabulary_index.search(features, k=k)

        if use_softmax:
            logit_scale = self.clip_model.logit_scale.exp().item()
            logits = class_similarities * logit_scale

            class_similarities = torch.softmax(torch.tensor(logits), dim=1).numpy()

            class_similarities = class_similarities[:, :save_top_k]
            class_indices = class_indices[:, :save_top_k]
        else:
            # Move [-1.0, 1.0] to [0.0, 1.0]
            class_similarities += 1.0
            class_similarities /= 2.0

        return class_similarities, class_indices

    def classify(self, images: list[Image.Image]) -> list[list[Classification]]:
        class_similarities, class_indices = self._search_index(
            features=self._get_image_features(images),
            k=self._candidate_count(),
            use_softmax=self.descriptor.parameters.use_softmax,
            save_top_k=self.descriptor.parameters.save_top_k,
        )
        return [
            [
                Classification(
                    source=self.name,
                    classname=self.descriptor.parameters.vocabulary[index],
                    confidence=similarity,
                )
                for index, similarity in zip(image_indices, image_similarities)
            ]
            for image_indices, image_similarities in zip(
                class_indices, class_similarities
            )
        ]
