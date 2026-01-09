krrood.entity_query_language.rxnode
===================================

.. py:module:: krrood.entity_query_language.rxnode


Attributes
----------

.. autoapisummary::

   krrood.entity_query_language.rxnode.GraphVisualizer


Classes
-------

.. autoapisummary::

   krrood.entity_query_language.rxnode.ColorLegend
   krrood.entity_query_language.rxnode.RWXNode


Module Contents
---------------

.. py:data:: GraphVisualizer
   :value: None


.. py:class:: ColorLegend

   .. py:attribute:: name
      :type:  str
      :value: 'Other'



   .. py:attribute:: color
      :type:  str
      :value: 'white'



.. py:class:: RWXNode

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: weight
      :type:  str
      :value: ''



   .. py:attribute:: data
      :type:  typing_extensions.Optional[typing_extensions.Any]
      :value: None



   .. py:attribute:: color
      :type:  ColorLegend


   .. py:attribute:: wrap_subtree
      :type:  bool
      :value: False



   .. py:attribute:: wrap_facecolor
      :type:  typing_extensions.Optional[str]
      :value: None



   .. py:attribute:: wrap_edgecolor
      :type:  typing_extensions.Optional[str]
      :value: None



   .. py:attribute:: wrap_alpha
      :type:  float
      :value: 0.08



   .. py:attribute:: enclosed
      :type:  bool
      :value: False



   .. py:attribute:: id
      :type:  int


   .. py:attribute:: enclosed_name
      :type:  typing_extensions.ClassVar[str]
      :value: 'enclosed'



   .. py:method:: add_parent(parent: RWXNode, edge_weight=None)


   .. py:method:: remove()


   .. py:method:: remove_node(node: RWXNode)


   .. py:method:: remove_child(child: RWXNode)


   .. py:method:: remove_parent(parent: RWXNode)


   .. py:property:: ancestors
      :type: typing_extensions.List[RWXNode]



   .. py:property:: parents
      :type: typing_extensions.List[RWXNode]



   .. py:property:: parent
      :type: typing_extensions.Optional[RWXNode]



   .. py:property:: children
      :type: typing_extensions.List[RWXNode]



   .. py:property:: descendants
      :type: typing_extensions.List[RWXNode]



   .. py:property:: leaves
      :type: typing_extensions.List[RWXNode]



   .. py:property:: root
      :type: RWXNode



   .. py:method:: visualize(figsize=(35, 30), node_size=7000, font_size=25, spacing_x: float = 4, spacing_y: float = 4, curve_scale: float = 0.5, layout: str = 'tidy', edge_style: str = 'orthogonal', label_max_chars_per_line: typing_extensions.Optional[int] = 13)

      Visualizes the graph using the specified layout and style options.

      Provides a graphical visualization of the graph with customizable options for
      size, layout, spacing, and labeling. Requires the rustworkx_utils library for
      execution.

      :param figsize (tuple of float): Size of the figure in inches (width, height). Default is (35, 30).
      :param node_size (int): Size of the nodes in the visualization. Default is 7000.
      :param font_size (int): Size of the font used for node labels. Default is 25.
      :param spacing_x (float): Horizontal spacing between nodes. Default is 4.
      :param spacing_y (float): Vertical spacing between nodes. Default is 4.
      :param curve_scale (float): Scaling factor for edge curvature. Default is 0.5.
      :param layout (str): Graph layout style (e.g., "tidy"). Default is "tidy".
      :param edge_style (str): Style of the edges (e.g., "orthogonal"). Default is "orthogonal".
      :param label_max_chars_per_line (Optional[int]): Maximum characters per line for node labels. Default is 13.

      :returns: The rendered visualization object.

      :raises: `ModuleNotFoundError` If rustworkx_utils is not installed.



