from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class ExampleConnection:
    type_code: int
    name: str

@dataclass(unsafe_hash=True)
class ExampleView:
    connection: ExampleConnection

@dataclass(eq=False)
class ExampleFixedView(ExampleView): pass

@dataclass(eq=False)
class ExampleRevoluteView(ExampleView): pass