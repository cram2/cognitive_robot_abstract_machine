"""
Domain-agnostic planning over the Entity Query Language.

A *fluent* is a :class:`~krrood.entity_query_language.ground_term.GroundTerm` whose value
a :class:`~krrood.planning.state.State` can change; the
:class:`~krrood.entity_query_language.ground_term.FunctionKey` it applies is its fluent
symbol. The Entity Query Language itself only knows ground terms.
"""
