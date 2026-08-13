import copy

import clip
import faiss
import torch
from PIL import Image
from py_trees.common import Status
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from typing_extensions import List


class ClipAnnotator(BaseAnnotator):
    """A simple CLIP annotator that uses a given vocabulary to annotate all incoming object hypothesis."""

    def __init__(self, name: str = "ClipAnnotator") -> None:
        super().__init__(name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """Torch device to be used by the annotator."""

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)

        self.clip_model = clip_model
        """CLIP model used for feature extraction."""

        self.clip_preprocess = clip_preprocess
        """CLIP preprocessor used for feature extraction."""

        self.vocabulary: List[str] = [
            "Milk Container",
            "Conflakes",
            "Pancake Batter",
            "Instant Coffee",
            "Crepe Pan",
            "Dish Soap",
        ]
        """The vocabulary used for object classification"""

        self.vocabulary_index: faiss.IndexFlatIP = faiss.IndexFlatIP(512)
        """A faiss index for text features used in similarity search."""

        with torch.no_grad():
            tokenized_vocabulary = [
                clip.tokenize([name]).to(self.device) for name in self.vocabulary
            ]
            text_features = torch.stack(
                [self.clip_model.encode_text(t) for t in tokenized_vocabulary]
            )
            text_features /= torch.norm(text_features, p=2, dim=1, keepdim=True)
            text_features = text_features.cpu().numpy()

        for feat in text_features:
            self.vocabulary_index.add(feat)

    def update(self) -> Status:
        ohs: List[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )
        if len(ohs) == 0:
            return Status.SUCCESS

        color_pil = Image.fromarray(
            copy.deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))
        )

        # Create per-object image crop
        cropped_images = (color_pil.crop(oh.roi.roi.get_corner_points()) for oh in ohs)

        with torch.no_grad():
            preprocessed_images = torch.stack(
                [self.clip_preprocess(crop) for crop in cropped_images]
            ).to(self.device)
            crop_features = self.clip_model.encode_image(preprocessed_images)
            crop_features /= torch.norm(crop_features, p=2, dim=1, keepdim=True)
            crop_features = crop_features.cpu().numpy()

        data, idx = self.vocabulary_index.search(crop_features, k=1)
        best_indices = idx[:, 0].astype(int)
        best_classes = [self.vocabulary[i] for i in best_indices]
        best_similarities = data[:, 0] * 100.0

        for oh, best_idx, best_class, best_similarity in zip(
            ohs, best_indices, best_classes, best_similarities
        ):
            oh.annotations.append(
                Classification(
                    source=self.name,
                    class_id=best_idx,
                    classname=best_class,
                    confidence=best_similarity,
                )
            )

            self.rk_logger.debug(
                f"Classified oh {oh.id} as {best_class} ({best_similarity:.2f})"
            )

        return Status.SUCCESS
