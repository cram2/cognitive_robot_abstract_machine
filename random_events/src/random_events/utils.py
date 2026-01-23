from abc import abstractmethod

from typing_extensions import Dict, Any, Self


def get_full_class_name(cls):
    """
    Returns the full name of a class, including the module name.

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__


def recursive_subclasses(cls):
    """
    :param cls: The class.
    :return: A list of the classes subclasses.
    """
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in recursive_subclasses(s)]


class SubclassJSONSerializer:
    """
    Class for automatic (de)serialization of subclasses.
    Classes that inherit from this class can be serialized and deserialized automatically by calling this classes
    'from_json' method.
    """

    def to_json(self) -> Dict[str, Any]:
        return {"type": get_full_class_name(self.__class__)}

    @classmethod
    @abstractmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create a variable from a json dict.
        This method is called from the from_json method after the correct subclass is determined and should be
        overwritten by the respective subclass.

        :param data: The json dict
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :return: The correct instance of the subclass
        """
        for subclass in recursive_subclasses(SubclassJSONSerializer):
            if get_full_class_name(subclass) == data["type"]:
                return subclass._from_json(data)

        raise ValueError("Unknown type {}".format(data["type"]))


class CPPWrapper:
    """
    Class for wrapping C++ objects.
    """

    _cpp_object: Any
    """
    The C++ object that this class wraps.
    """

    @abstractmethod
    def _from_cpp(self, cpp_object: Any) -> Self:
        """
        Create a new instance of this class from a C++ object.

        This method should also add fields that are python only to the instance that is created.
        This cannot be a class method since the values of the python-only fields are instance-specific.
        """
        raise NotImplementedError
