from dataclasses import fields, is_dataclass
from typing_extensions import (
    Type,
    get_origin,
    get_args,
    List,
    Union,
    Sequence,
    get_type_hints,
)
from enum import Enum
from random_events.variable import Continuous, Integer, Symbolic
from random_events.set import Set


class Parameterizer:
    """
    Tool to create random_events variables from a dataclass:
    - float -> Continuous
    - int -> Integer
    - Enum -> Symbolic
    Recursively flattens nested dataclasses to include all fields.
    """

    def __call__(self, wrapped_class: Type):
        if not is_dataclass(wrapped_class):
            raise TypeError(f"Expected a dataclass, got {wrapped_class}")
        return self._parameterize(wrapped_class, wrapped_class.__name__)

    def _parameterize(self, cls: Type, prefix: str):
        variables = []

        type_hints = get_type_hints(cls)
        for field in fields(cls):
            field_name = field.name
            qualified_name = f"{prefix}.{field_name}"
            field_type = type_hints[field_name]

            # Handle Optional[T]
            origin = get_origin(field_type)
            args = get_args(field_type)
            if origin is Union:
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    field_type = non_none_args[0]

            # Handle Sequence[T]
            origin = get_origin(field_type)
            args = get_args(field_type)
            if origin in (list, List, Sequence) and args:
                field_type = args[0]

            #  Nested dataclass to recurse
            if isinstance(field_type, type) and is_dataclass(field_type):
                variables.extend(self._parameterize(field_type, qualified_name))

            # Float to Continuous
            elif field_type == float:
                variables.append(Continuous(name=qualified_name))

            #  Int to Integer
            elif field_type == int:
                variables.append(Integer(name=qualified_name))

            # Enum to Symbolic
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                domain = Set.from_iterable(list(field_type))
                variables.append(Symbolic(name=qualified_name, domain=domain))

        return variables
