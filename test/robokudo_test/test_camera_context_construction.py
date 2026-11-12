"""
Dataclass construction of camera context resolvers.
"""

from __future__ import annotations

# %% Imports
from robokudo.descriptors.camera_configs.config_ray_tracing import (
    RayTracingCameraConfiguration,
)
from robokudo.io.ray_tracing_context import RayTracingContextResolver
from robokudo.utils.module_loader import ModuleLoader

# %% Resolver construction


def test_camera_context_resolver_preserves_injected_loader() -> None:
    """
    Construction retains the caller's configuration and loader.
    """
    configuration = RayTracingCameraConfiguration()
    loader = ModuleLoader()
    resolver = RayTracingContextResolver(configuration, module_loader=loader)
    assert resolver.camera_config is configuration
    assert resolver.module_loader is loader


def test_camera_context_resolvers_have_separate_default_loaders() -> None:
    """
    Default loaders belong to individual resolver instances.
    """
    configuration = RayTracingCameraConfiguration()
    first_resolver = RayTracingContextResolver(configuration)
    second_resolver = RayTracingContextResolver(configuration)
    assert isinstance(first_resolver.module_loader, ModuleLoader)
    assert isinstance(second_resolver.module_loader, ModuleLoader)
    assert first_resolver.module_loader is not second_resolver.module_loader
