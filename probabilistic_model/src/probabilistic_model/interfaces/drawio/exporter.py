import networkx
import numpy as np
from N2G import drawio_diagram

from ...probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit


class DrawIoExporter:
    """
    Export a probabilistic circuit to a drawio diagram.
    """

    model: ProbabilisticCircuit
    """
    The probabilistic circuit to export.
    """

    def __init__(self, model: ProbabilisticCircuit):
        self.model = model

    def export(self) -> drawio_diagram:
        unique_id = 1000
        diagram = drawio_diagram()
        diagram.add_diagram("Structure", width=1360, height=1864)

        for unit, (x, y) in networkx.drawing.bfs_layout(self.model.graph, self.model.root).items():
            diagram.add_node(id=str(hash(unit)), x_pos=x * 100, y_pos=y * 100, **unit.drawio_style)
            if not unit.is_leaf:
                diagram.current_root[-1].attrib["label"] = ""
            else:
                node_text = "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Helvetica;fontSize=12;fontColor=default;"
                diagram.add_node(id=str(unique_id), x_pos=x * 100 + 35, y_pos=y * 100, style=node_text,
                                 label=unit.variables[0].name, width=100, height=30)
                unique_id += 1

        for source, target in self.model.unweighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), style='endArrow=classic;html=1;rounded=0;')

        for source, target, weight in self.model.log_weighted_edges:
            diagram.add_link(str(hash(source)), str(hash(target)), label=f"{round(weight, 2)}",
                             style=f'endArrow=classic;html=1;rounded=0;opacity={np.exp(weight) * 100};')

        return diagram
