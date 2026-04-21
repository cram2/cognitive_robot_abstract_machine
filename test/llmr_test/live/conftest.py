"""Fixtures for opt-in live LLM tests.

These tests are skipped by default and require llmr/.env or environment
variables with LLMR_LIVE_TESTS=1 and OPENAI_API_KEY set.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from llmr.reasoning.llm_config import LLMProvider, make_llm

load_dotenv("llmr/.env", override=True)


@pytest.fixture
def live_llm():
    provider = LLMProvider(os.getenv("LLMR_TEST_PROVIDER", "openai"))
    model = os.getenv("LLMR_TEST_MODEL") or "gpt-4o-mini"
    return make_llm(provider, model=model, temperature=0.0)
