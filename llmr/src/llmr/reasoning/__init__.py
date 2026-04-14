"""LLM reasoning pipeline.

  slot_filler  — classify NL instruction → action class; build and invoke slot-filling prompt.
  decomposer   — split compound instructions into atomic steps with object-flow dependencies.
  llm_config   — provider-agnostic LangChain model factory (OpenAI / Ollama).
"""
