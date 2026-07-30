from __future__ import annotations

import time

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from py_trees.common import Status
from transformers import CLIPModel, CLIPProcessor
from typing_extensions import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt

FONT = cv2.FONT_HERSHEY_COMPLEX


class ClipAnnotator(ThreadedAnnotator):
    """
    A simple CLIP annotator that uses a given vocabulary to annotate all incoming object
    hypothesis.
    """

    class Descriptor(ThreadedAnnotator.Descriptor):
        """
        Descriptor for the ClipAnnotator.
        """

        class Parameters:
            """
            Parameters for the Descriptor.
            """

            def __init__(self) -> None:
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
                self.vocabulary: List[str] = [
                    "Milk Container",
                    "Box of Cornflakes",
                    "Pancake Batter",
                    "Instant Coffee",
                    "Crepe Pan",
                    "Dish Soap",
                ]
                """
                The vocabulary used for object classification.
                """
                self.analysis_scope = CASViews.COLOR_IMAGE
                """
                The analysis scope of the annotator.

                Possible values:

                * CASViews.COLOR_IMAGE: Analyze the entire color image at once.
                * ObjectHypothesis: Analyze each object hypothesis individually.
                """
                self.filter_fn: Optional[Callable[[ObjectHypothesis], bool]] = None
                """
                An optional filter function used to filter object hypotheses before
                analyzing them.
                """
                self.save_top_k: int = min(5, len(self.vocabulary))
                """
                How many annotations to create per analyzed image or hypothesis.
                """
                self.use_softmax: bool = True
                """
                Whether to apply softmax to the results of the analyzed image or
                hypothesis or not.
                """

    def __init__(
        self,
        name: str = "ClipAnnotator",
        descriptor: ClipAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name=name, descriptor=descriptor)

        self.rk_logger.debug(f"Starting to init {self.__class__.__name__}")

        self.parameters = descriptor.parameters
        """
        The parameters used by this annotator.
        """
        self.clip_model = CLIPModel.from_pretrained(self.parameters.model_name)
        """
        CLIP model used for feature extraction.
        """
        self.clip_preprocess = CLIPProcessor.from_pretrained(self.parameters.model_name)
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
                self.parameters.vocabulary_template.format(name)
                for name in self.parameters.vocabulary
            ]
            inputs = self.clip_tokenizer(
                templated_texts,
                padding=True,
                return_tensors="pt",
            ).to(self.parameters.device)
            text_features = self.clip_model.get_text_features(**inputs).pooler_output
            text_features /= torch.norm(text_features, p=2, dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
        self.vocabulary_index.add(text_features)

        self.rk_logger.debug(f"Initialized {self.__class__.__name__}")

    def _get_image_features(
        self,
        images: Union[Image.Image, List[Image.Image], npt.NDArray, List[npt.NDArray]],
    ) -> npt.NDArray[np.float32]:
        """
        Get normalized image features for the given image(s).

        :param images: The image(s) to get features from.
        :return: The image(s) features.
        """
        with torch.inference_mode():
            inputs = self.clip_preprocess(images=images, return_tensors="pt").to(
                self.clip_model.device
            )
            crop_features = self.clip_model.get_image_features(**inputs).pooler_output
            crop_features /= torch.norm(crop_features, p=2, dim=-1, keepdim=True)
            crop_features = crop_features.cpu().numpy()
        return crop_features

    def _search_index(
        self,
        features: npt.NDArray[np.float32],
        k: int,
        use_softmax: bool,
        save_top_k: int,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
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

    def _analyze_scene(self, color_image: npt.NDArray[np.uint8]) -> Status:
        """
        Analyze the entire color image.

        :param color_image: The color image to analyze.
        :return: The py_trees status.
        """
        cas = self.get_cas()
        parameters: ClipAnnotator.Descriptor.Parameters = self.parameters

        vocabulary: List[str] = parameters.vocabulary
        use_softmax: bool = parameters.use_softmax
        save_top_k: int = parameters.save_top_k

        image_features = self._get_image_features(color_image)

        class_similarities, class_indices = self._search_index(
            features=image_features,
            k=len(vocabulary) if use_softmax else save_top_k,
            use_softmax=use_softmax,
            save_top_k=save_top_k,
        )

        class_indices = class_indices[0]
        class_similarities = class_similarities[0]

        class_names = [vocabulary[i] for i in class_indices]

        visualization_img = color_image.copy()

        self.rk_logger.info("Image Classification Results:")
        self.rk_logger.info("Label \t\t Score")
        for i, (class_name, similarity) in enumerate(
            zip(class_names, class_similarities)
        ):
            classification = Classification(
                source=self.name, classname=class_name, confidence=similarity
            )
            cas.annotations.append(classification)

            text = f"{similarity:.2f} /{class_name}"
            visualization_img = cv2.putText(
                visualization_img, text, (10, 40 + 20 * i), FONT, 0.5, (0, 0, 255), 1, 2
            )

            self.rk_logger.info(
                f"Top {i} prediction: Class {class_name} with confidence\t {similarity:.4f}"
            )

        self.get_annotator_output_struct().set_image(visualization_img)
        return Status.SUCCESS

    def _analyze_object_hypotheses(self, color_image: npt.NDArray[np.uint8]) -> Status:
        """
        Analyze the object hypothesis using the given color image.

        :param color_image: The color image to use for analyzation.
        :return: The py_trees status.
        """
        ohs: List[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )
        if len(ohs) == 0:
            self.rk_logger.info("Found 0 object hypotheses to analyze")
            return Status.SUCCESS

        parameters: ClipAnnotator.Descriptor.Parameters = self.parameters
        if parameters.filter_fn is not None:
            filter_fn = parameters.filter_fn
            ohs = list(filter(filter_fn, ohs))

            if len(ohs) == 0:
                self.rk_logger.info("All object hypotheses were filtered out")
                return Status.SUCCESS

        vocabulary: List[str] = parameters.vocabulary
        use_softmax: bool = parameters.use_softmax
        save_top_k: int = parameters.save_top_k

        # Create per-object image crop
        pil_image = Image.fromarray(color_image)
        cropped_images = [pil_image.crop(oh.roi.roi.get_corner_points()) for oh in ohs]

        crop_features = self._get_image_features(cropped_images)

        class_similarities, class_indices = self._search_index(
            features=crop_features,
            k=len(vocabulary) if use_softmax else save_top_k,
            use_softmax=use_softmax,
            save_top_k=save_top_k,
        )

        class_names = [[vocabulary[idx] for idx in idxs] for idxs in class_indices]

        visualization_img = color_image.copy()

        for object_hypothesis, oh_class_names, oh_class_similarities in zip(
            ohs, class_names, class_similarities
        ):
            self.rk_logger.info(
                f"Classification results for Object hypothesis ID: '{object_hypothesis.id}'"
            )

            text = f"{oh_class_similarities[0]:.4f} / {oh_class_names[0]}"
            x1, y1, x2, y2 = object_hypothesis.roi.roi.get_corner_points()
            visualization_img = cv2.rectangle(
                visualization_img, (x1, y1), (x2, y2), (255, 255, 255), 2
            )
            visualization_img = cv2.putText(
                visualization_img, text, (x1, y1 - 20), FONT, 0.5, (0, 0, 255), 1, 2
            )

            for i, (class_name, similarity) in enumerate(
                zip(oh_class_names, oh_class_similarities)
            ):
                classification = Classification(
                    source=self.name, classname=class_name, confidence=similarity
                )
                object_hypothesis.annotations.append(classification)

                self.rk_logger.info(
                    f"Top {i}: Classified as {class_name} "
                    f"with confidence {similarity:.4f}"
                )

        self.get_annotator_output_struct().set_image(visualization_img)
        return Status.SUCCESS

    def compute(self) -> Status:
        """
        Compute CLIP features depending on the analysis scope.

        :return: Whether the computation was successfull or not.
        """
        start_time = time.perf_counter()
        analysis_scope = self.parameters.analysis_scope

        color_image = self.get_cas().get(CASViews.COLOR_IMAGE).copy()

        if analysis_scope == CASViews.COLOR_IMAGE:
            status = self._analyze_scene(color_image)
        elif analysis_scope == ObjectHypothesis:
            status = self._analyze_object_hypotheses(color_image)
        else:
            self.feedback_message = f"Unknown analysis scope: {analysis_scope}"
            return Status.FAILURE

        self.feedback_message = (
            f"Processing took {time.perf_counter() - start_time:.2f}s"
        )
        return status
