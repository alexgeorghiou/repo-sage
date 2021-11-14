from __future__ import annotations

from repo_sage.config import Settings
from repo_sage.llm.base import LLM
from repo_sage.llm.fake import FakeLLM


def build_llm(settings: Settings) -> LLM:
    if settings.llm == "anthropic":
        from repo_sage.llm.anthropic_llm import DEFAULT_MODEL, AnthropicLLM

        return AnthropicLLM(
            model=settings.llm_model or DEFAULT_MODEL, max_tokens=settings.llm_max_tokens
        )
    if settings.llm == "openai":
        from repo_sage.llm.openai_llm import DEFAULT_MODEL, OpenAILLM

        return OpenAILLM(
            model=settings.llm_model or DEFAULT_MODEL, max_tokens=settings.llm_max_tokens
        )
    return FakeLLM()
