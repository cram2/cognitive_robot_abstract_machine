from typing_extensions import Union, Iterator, Type, Tuple
from krrood.ripple_down_rules.datastructures.tracked_object import TrackedObjectMixin

PredicateArgElementType = Union[Type[TrackedObjectMixin], TrackedObjectMixin]
PredicateArgType = Union[Iterator[PredicateArgElementType], PredicateArgElementType]
PredicateOutputType = Iterator[Tuple[PredicateArgElementType, PredicateArgElementType]]
