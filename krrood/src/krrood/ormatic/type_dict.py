from dataclasses import dataclass, field
from typing import Type, Any, Iterator, Tuple


from typing_extensions import Dict

from ..utils import inheritance_path_length


@dataclass
class TypeDict:
    """
    A dictionary that gets the closest key, using the inheritance path length instead of looking for direct matches.
    """

    _dict: Dict[Type, Any] = field(default_factory=dict)
    """
    The pure data dict
    """

    def _get_inheritance_path_length_of_keys(self, clazz: Type) -> Dict[Type, int]:
        """
        Get the inheritance path length of all keys in the dictionary relative to the given class.

        :param clazz: The class to get distances from.
        :return: The class distances.
        """
        return {
            k: v
            for k in self._dict.keys()
            if (v := inheritance_path_length(clazz, k)) is not None
        }

    def __getitem__(self, key: Type) -> Any:
        distances = self._get_inheritance_path_length_of_keys(key)
        if distances:
            return self._dict[min(distances, key=distances.get)]
        else:
            raise KeyError(
                f"No matching key found for {key} using inheritance path length"
            )

    def get(self, key: Type, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: Type, value: Any) -> None:
        return self._dict.__setitem__(key, value)

    def __delitem__(self, key: Type) -> None:
        return self._dict.__delitem__(key)

    def __iter__(self) -> Iterator[Type]:
        return self._dict.__iter__()

    def __len__(self) -> int:
        return self._dict.__len__()

    def __contains__(self, key: Type) -> bool:
        return len(self._get_inheritance_path_length_of_keys(key)) > 0

    def keys(self) -> Iterator[Type]:
        return self._dict.keys()

    def values(self) -> Iterator[Any]:
        return self._dict.values()

    def items(self) -> Iterator[Tuple[Type, Any]]:
        return self._dict.items()
