import tempfile
import unittest

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.interfaces.drawio.exporter import DrawIoExporter
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *


@unittest.skip
class DrawIOExporterTestCase(unittest.TestCase):
    variables: List[Variable]

    x = Continuous("x")
    y = Continuous("y")

    def setUp(self) -> None:
        model = ProbabilisticCircuit()
        sum1, sum2, sum3 = (
            SumUnit(probabilistic_circuit=model),
            SumUnit(probabilistic_circuit=model),
            SumUnit(probabilistic_circuit=model),
        )
        sum4, sum5 = SumUnit(probabilistic_circuit=model), SumUnit(
            probabilistic_circuit=model
        )
        prod1, prod2 = ProductUnit(probabilistic_circuit=model), ProductUnit(
            probabilistic_circuit=model
        )

        sum1.add_subcircuit(prod1, np.log(0.5))
        sum1.add_subcircuit(prod2, np.log(0.5))
        prod1.add_subcircuit(sum2)
        prod1.add_subcircuit(sum4)
        prod2.add_subcircuit(sum3)
        prod2.add_subcircuit(sum5)

        d_x1 = leaf(
            UniformDistribution(self.x, SimpleInterval(0, 1)),
            probabilistic_circuit=model,
        )
        d_x2 = leaf(
            UniformDistribution(self.x, SimpleInterval(2, 3)),
            probabilistic_circuit=model,
        )
        d_y1 = leaf(
            UniformDistribution(self.y, SimpleInterval(0, 1)),
            probabilistic_circuit=model,
        )
        d_y2 = leaf(
            UniformDistribution(self.y, SimpleInterval(3, 4)),
            probabilistic_circuit=model,
        )

        sum2.add_subcircuit(d_x1, np.log(0.8))
        sum2.add_subcircuit(d_x2, np.log(0.2))
        sum3.add_subcircuit(d_x1, np.log(0.7))
        sum3.add_subcircuit(d_x2, np.log(0.3))

        sum4.add_subcircuit(d_y1, np.log(0.5))
        sum4.add_subcircuit(d_y2, np.log(0.5))
        sum5.add_subcircuit(d_y1, np.log(0.1))
        sum5.add_subcircuit(d_y2, np.log(0.9))

        self.model = sum1.probabilistic_circuit

    def test_export_to_drawio(self):
        diagram = DrawIoExporter(self.model.root.probabilistic_circuit).export()
        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as temp_file:
            temp_file_name = temp_file.name

        # temp_file_name = os.path.join(os.path.expanduser("~"), "Documents", "test.drawio")

        # Write the diagram data to the temporary file
        with open(temp_file_name, "w") as f:
            f.write(diagram.dump_xml())
        print(f"Diagram exported to temporary file: {temp_file_name}")


if __name__ == "__main__":
    unittest.main()
