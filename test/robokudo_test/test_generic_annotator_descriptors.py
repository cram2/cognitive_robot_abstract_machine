"""
Generic bindings determine annotator defaults and concrete field types.
"""

from __future__ import annotations

# %% Imports
from dataclasses import fields, replace

import pytest
from robokudo.annotators.blur import BlurAnnotator, BlurDescriptor
from robokudo.annotators.cas_check import (
    CASCheckAnnotationTypeExists,
    CASCheckObjectHypothesisExists,
)
from robokudo.annotators.clip_annotator import CLIPAnnotator, CLIPDescriptor
from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.annotators.image_cluster_extractor import (
    ImageClusterExtractor,
    ImageClusterExtractorDescriptor,
)
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneDescriptor
from robokudo.annotators.pointcloud_check import PointcloudCheckAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import (
    NaivePointCloudClusterExtractor,
    NaivePointCloudClusterExtractorDescriptor,
)
from robokudo.datastructures.annotator_descriptor import BaseAnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorPredefinedParameters
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from typing_extensions import get_type_hints

from .dataset.generic_annotators import (
    AnnotatorNamingScope,
    DescriptorBindingAnnotator,
    InheritedAnnotationChecker,
    InheritedDescriptorBindingAnnotator,
    InheritedObjectHypothesisChecker,
    ThreadedDescriptorBindingAnnotator,
)


# %% Default construction and generic bindings
@pytest.mark.parametrize(
    ("annotator_type", "descriptor_type"),
    [
        (CLIPAnnotator, CLIPDescriptor),
        (ImageClusterExtractor, ImageClusterExtractorDescriptor),
        (NaivePointCloudClusterExtractor, NaivePointCloudClusterExtractorDescriptor),
    ],
)
def test_descriptor_discovery_handles_classifier_and_runtime_field_annotations(
    annotator_type: type[BaseAnnotator],
    descriptor_type: type,
) -> None:
    """
    Descriptor bindings resolve independently of runtime resource annotations.
    """
    assert annotator_type.descriptor_type is descriptor_type


@pytest.mark.parametrize(
    ("annotator_type", "descriptor_type"),
    [
        (DescriptorBindingAnnotator, BlurDescriptor),
        (InheritedDescriptorBindingAnnotator, BlurDescriptor),
        (ThreadedDescriptorBindingAnnotator, PlaneDescriptor),
    ],
)
def test_generic_binding_supplies_descriptor_without_child_declarations(
    annotator_type: type[BaseAnnotator],
    descriptor_type: type,
) -> None:
    """
    Defaults and field annotations agree on the concrete binding.
    """
    annotator = annotator_type()

    assert type(annotator.descriptor) is descriptor_type
    assert annotator_type.descriptor_type is descriptor_type
    assert get_type_hints(annotator_type)["descriptor"] is descriptor_type
    assert (
        next(item for item in fields(annotator) if item.name == "descriptor").type
        is descriptor_type
    )


@pytest.mark.parametrize("annotator_type", [BaseAnnotator, ThreadedAnnotator])
def test_unspecialized_base_annotator_has_common_descriptor(
    annotator_type: type[BaseAnnotator],
) -> None:
    """
    Direct construction keeps the common configuration available.
    """
    assert type(annotator_type().descriptor) is BaseAnnotatorDescriptor


@pytest.mark.parametrize("annotator_type", [BaseAnnotator, ThreadedAnnotator])
def test_replacing_base_annotator_retains_configuration_and_creates_new_tree_identity(
    annotator_type: type[BaseAnnotator],
) -> None:
    """
    Dataclass copies of either base accept a new name and retain their descriptor.
    """
    descriptor = BaseAnnotatorDescriptor()
    original = annotator_type(name="Original", descriptor=descriptor)

    replacement = replace(original, name="Replacement")

    assert replacement.name == "Replacement"
    assert replacement.descriptor is descriptor
    assert replacement.id != original.id
    assert replacement.parent is None
    assert replacement.rk_logger is original.rk_logger


def test_generic_defaults_are_independent_and_capture_application_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A new annotator captures changed defaults without mutating existing nodes.
    """
    first = DescriptorBindingAnnotator()
    monkeypatch.setattr(
        AnnotatorPredefinedParameters(),
        "global_with_visualization",
        not first.descriptor.parameters.global_with_visualization,
    )
    second = DescriptorBindingAnnotator()

    assert first.descriptor is not second.descriptor
    assert (
        second.descriptor.parameters.global_with_visualization
        is not first.descriptor.parameters.global_with_visualization
    )


def test_explicit_descriptor_is_preserved_by_construction_and_replacement() -> None:
    """
    Supplying configuration keeps its identity across dataclass copies.
    """
    descriptor = BlurDescriptor()
    original = DescriptorBindingAnnotator(descriptor=descriptor)
    replacement = replace(original, name="Replacement")

    assert original.descriptor is descriptor
    assert replacement.descriptor is descriptor
    assert replacement.id != original.id


def test_explicit_absence_uses_default_descriptor() -> None:
    """
    The existing constructor convention accepts absence as a default request.
    """
    assert (
        type(DescriptorBindingAnnotator(descriptor=None).descriptor) is BlurDescriptor
    )


# %% Specialized initialization
def test_query_scheduler_preserves_supplied_descriptor() -> None:
    """
    Scheduler initialization preserves configuration supplied to its annotator base.
    """
    descriptor = BaseAnnotatorDescriptor()

    scheduler = QueryBasedScheduler(descriptor=descriptor)

    assert scheduler.descriptor is descriptor


@pytest.mark.parametrize(
    "annotator_type", [CASCheckAnnotationTypeExists, CASCheckObjectHypothesisExists]
)
def test_checker_initialization_preserves_supplied_descriptor(
    annotator_type: type[BaseAnnotator],
) -> None:
    """
    Explicit configuration survives specialized checker initialization and copying.
    """
    descriptor = BaseAnnotatorDescriptor()
    checker = annotator_type(descriptor=descriptor)

    assert checker.descriptor is descriptor
    assert replace(checker, name="ReplacementChecker").descriptor is descriptor


# %% Concrete annotator names
@pytest.mark.parametrize(
    "annotator_type",
    [
        BaseAnnotator,
        ThreadedAnnotator,
        BlurAnnotator,
        InheritedDescriptorBindingAnnotator,
        ThreadedDescriptorBindingAnnotator,
        ImagePreprocessorAnnotator,
        PointcloudCheckAnnotator,
        CASCheckAnnotationTypeExists,
        CASCheckObjectHypothesisExists,
        InheritedAnnotationChecker,
        InheritedObjectHypothesisChecker,
        QueryBasedScheduler,
    ],
)
def test_default_name_identifies_the_concrete_annotator(
    annotator_type: type[BaseAnnotator],
) -> None:
    """
    Inherited and custom constructors use the concrete class for tree labels.
    """
    annotator = annotator_type()

    assert annotator.name == annotator_type.__qualname__
    assert (
        annotator.qualified_name
        == f"{annotator_type.__qualname__}/{annotator_type.__qualname__}"
    )


@pytest.mark.parametrize(
    "annotator_type",
    [
        BaseAnnotator,
        ThreadedDescriptorBindingAnnotator,
        CASCheckAnnotationTypeExists,
        CASCheckObjectHypothesisExists,
        InheritedAnnotationChecker,
        InheritedObjectHypothesisChecker,
        QueryBasedScheduler,
    ],
)
@pytest.mark.parametrize("configured_name", ["AEConfiguredName", ""])
def test_explicit_name_is_preserved(
    annotator_type: type[BaseAnnotator],
    configured_name: str,
) -> None:
    """
    Explicit labels, including empty strings, remain under the caller's control.
    """
    annotator = annotator_type(name=configured_name)

    assert annotator.name == configured_name
    assert (
        annotator.qualified_name == f"{annotator_type.__qualname__}/{configured_name}"
    )


def test_default_name_uses_the_qualified_class_name() -> None:
    """
    A qualified class label is preserved instead of shortening it to the simple name.
    """
    annotator = AnnotatorNamingScope.Annotator()

    assert annotator.name == type(annotator).__qualname__


def test_dataclass_copies_preserve_resolved_names_and_accept_overrides() -> None:
    """
    Copies retain resolved labels while obtaining independent tree identities.
    """
    original = InheritedDescriptorBindingAnnotator()
    copied = replace(original)
    renamed = replace(original, name="RenamedForAE")

    assert copied.name == type(original).__qualname__
    assert copied.qualified_name == original.qualified_name
    assert copied.id != original.id
    assert renamed.name == "RenamedForAE"
    assert renamed.qualified_name == f"{type(original).__qualname__}/{renamed.name}"
    assert renamed.descriptor is original.descriptor
    assert get_type_hints(type(original))["name"] is str
