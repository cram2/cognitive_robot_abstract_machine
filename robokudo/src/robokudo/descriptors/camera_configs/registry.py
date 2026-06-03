import importlib
import pkgutil

from typing_extensions import Type, Dict, Any

from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig


class CameraConfigRegistry:
    """Registry for camera config classes."""

    _registry: Dict[str, Type[BaseCameraConfig]] = {}
    """Map of camera config registry name to their class."""

    @classmethod
    def register_all(
        cls, package_name: str = "robokudo.descriptors.camera_configs"
    ) -> None:
        """Register all camera configs of the given package to the camera config registry.

        :param package_name: The name of the package to register camera configs from.
        :raises ImportError: If the given package cannot be imported.
        :raises ValueError: If there is a duplicate registry name with conflicting classes.
        """
        package = importlib.import_module(package_name)
        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, prefix=f"{package_name}."
        ):
            module = importlib.import_module(module_name)
            for name, obj in module.__dict__.items():
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseCameraConfig)
                    and obj != BaseCameraConfig
                ):
                    if (
                        obj.registry_name in cls._registry
                        and obj != cls._registry[obj.registry_name]
                    ):
                        raise ValueError(
                            f"Conflicting registry types for key {obj.registry_name}: {cls._registry[obj.registry_name]} and {obj}"
                        )
                    cls._registry[obj.registry_name] = obj

    @classmethod
    def create_config(cls, config_type: str, **kwargs: Any) -> Any:
        """Create a camera config instance based on the given config type.

        :param config_type: The type of the camera config to create.
        :raises ValueError: If the given config type is not registered.
        :raises TypeError: If the keyword arguments are invalid for the camera config class.
        """
        config_class = cls._registry.get(config_type)
        if not config_class:
            raise ValueError(
                f"Unknown config type: {config_type}, known types: {cls._registry.keys()}"
            )
        return config_class(**kwargs)


CameraConfigRegistry.register_all()
