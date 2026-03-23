
from __future__ import annotations

import logging
from typing_extensions import Any, Dict, Optional, Tuple, Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import END, START, StateGraph
from pydantic import BaseModel

from llmr.workflows.llm_configuration import default_llm
from llmr.workflows.prompts.pick_up import pick_up_resolver_prompt
from llmr.workflows.prompts.place import place_resolver_prompt
from llmr.workflows.schemas.pick_up import PickUpDiscreteResolutionSchema
from llmr.workflows.schemas.place import PlaceDiscreteResolutionSchema
from llmr.workflows.states.all_states import DiscreteResolutionState

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ── Generic graph factory (cached per (prompt_id, schema_cls)) ────────────────

# ChatPromptTemplate is not hashable so functools.lru_cache cannot be used.
# Prompt objects are module-level singletons; caching by object identity (id)
# is stable for the lifetime of the process.
_graph_cache: Dict[Tuple[int, Type[BaseModel]], Any] = {}


def _build_resolver_graph(
    prompt: ChatPromptTemplate,
    schema_cls: Type[BaseModel],
) -> Any:
    """Build and compile a resolver graph for a given prompt + schema pair.

    Results are cached by ``(id(prompt), schema_cls)`` so the graph is compiled
    only once per action type across the lifetime of the process.
    """
    cache_key = (id(prompt), schema_cls)
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]
    structured_llm = default_llm.with_structured_output(schema_cls, method="function_calling")

    def _resolver_node(state: DiscreteResolutionState) -> Dict[str, Any]:
        chain = prompt | structured_llm
        try:
            schema: BaseModel = chain.invoke(
                {
                    "world_context": state["world_context"],
                    "known_parameters": state["known_parameters"],
                    "parameters_to_resolve": state["parameters_to_resolve"],
                }
            )
            logger.debug(
                "resolver_node (%s) – %s",
                schema_cls.__name__,
                schema.model_dump(),
            )
            return {"resolved_schema": schema.model_dump(), "error": None}
        except Exception as exc:  # noqa: BLE001
            logger.error("resolver_node (%s) failed: %s", schema_cls.__name__, exc)
            return {"resolved_schema": None, "error": str(exc)}

    builder = StateGraph(DiscreteResolutionState)
    builder.add_node("resolver", _resolver_node)
    builder.add_edge(START, "resolver")
    builder.add_edge("resolver", END)
    compiled = builder.compile()
    _graph_cache[cache_key] = compiled
    return compiled


# ── Generic run function ──────────────────────────────────────────────────────


def run_resolver(
    world_context: str,
    known_parameters: str,
    parameters_to_resolve: str,
    prompt: ChatPromptTemplate,
    schema_cls: Type[T],
) -> Optional[T]:
    """Run the resolver node for any action type.

    :param world_context: Serialised world snapshot for scene reasoning.
    :param known_parameters: Human-readable summary of known slot values.
    :param parameters_to_resolve: Names of the parameters still to be decided.
    :param prompt: The prompt template for this action's resolver.
    :param schema_cls: The Pydantic model class for the expected output.
    :return: Validated schema instance or ``None`` on failure.
    """
    graph = _build_resolver_graph(prompt, schema_cls)
    final_state = graph.invoke(
        {
            "world_context": world_context,
            "known_parameters": known_parameters,
            "parameters_to_resolve": parameters_to_resolve,
        }
    )

    if final_state.get("error"):
        logger.warning("run_resolver (%s): %s", schema_cls.__name__, final_state["error"])
        return None

    raw = final_state.get("resolved_schema")
    if raw is None:
        return None

    return schema_cls.model_validate(raw)


# ── Typed convenience wrappers ────────────────────────────────────────────────


def run_pickup_resolver(
    world_context: str,
    known_parameters: str,
    parameters_to_resolve: str,
) -> Optional[PickUpDiscreteResolutionSchema]:
    """Resolve discrete parameters for a PickUpAction.

    :return: ``PickUpDiscreteResolutionSchema`` or ``None`` on failure.
    """
    return run_resolver(
        world_context=world_context,
        known_parameters=known_parameters,
        parameters_to_resolve=parameters_to_resolve,
        prompt=pick_up_resolver_prompt,
        schema_cls=PickUpDiscreteResolutionSchema,
    )


def run_place_resolver(
    world_context: str,
    known_parameters: str,
    parameters_to_resolve: str,
) -> Optional[PlaceDiscreteResolutionSchema]:
    """Resolve discrete parameters for a PlaceAction.

    :return: ``PlaceDiscreteResolutionSchema`` or ``None`` on failure.
    """
    return run_resolver(
        world_context=world_context,
        known_parameters=known_parameters,
        parameters_to_resolve=parameters_to_resolve,
        prompt=place_resolver_prompt,
        schema_cls=PlaceDiscreteResolutionSchema,
    )
