from __future__ import annotations
from dataclasses import dataclass

import krrood.symbol_graph.symbol_graph


@dataclass
class InMemoryClass(krrood.symbol_graph.symbol_graph.Symbol): ...


@dataclass
class InMemoryChildClass(InMemoryClass): ...
