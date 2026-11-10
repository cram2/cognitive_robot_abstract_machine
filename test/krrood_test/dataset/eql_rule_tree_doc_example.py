from dataclasses import dataclass


@dataclass(kw_only=True, unsafe_hash=True)
class ExampleConnection:
    type_code: int
    name: str


@dataclass(kw_only=True, unsafe_hash=True)
class ExampleView:
    connection: ExampleConnection


@dataclass(kw_only=True, eq=False)
class ExampleFixedView(ExampleView):
    pass


@dataclass(kw_only=True, eq=False)
class ExampleRevoluteView(ExampleView):
    pass
