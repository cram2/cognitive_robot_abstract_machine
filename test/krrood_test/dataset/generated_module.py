from __future__ import annotations
from dataclasses import dataclass

import krrood.symbol_graph.symbol_graph


@dataclass(kw_only=True)
class InMemoryClass(krrood.symbol_graph.symbol_graph.Symbol): ...


@dataclass(kw_only=True)
class InMemoryChildClass(InMemoryClass): ...
