from typing_extensions import Dict, Any

circled_product = "verticalLabelPosition=bottom;verticalAlign=top;html=1;shape=mxgraph.flowchart.or;"
circled_sum = "verticalLabelPosition=bottom;verticalAlign=top;html=1;shape=mxgraph.flowchart.summing_function;"

class DrawIOInterface:
    """
    Mixin class for objects that can be used with a drawio exporter.
    """

    @property
    def drawio_label(self) -> str:
        """
        The label of the object as a drawio compatible string.
        """
        raise NotImplementedError()

    @property
    def drawio_style(self) -> Dict[str, Any]:
        """
        The style of the object.
        """
        raise NotImplementedError()
