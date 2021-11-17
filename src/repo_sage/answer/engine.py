"""Retrieval-augmented answering with validated citations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from repo_sage.answer.citations import parse_citations, validate_citations
from repo_sage.answer.prompt import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, build_user_prompt
from repo_sage.llm.base import LLM, Message
from repo_sage.models import Answer, ScoredChunk
from repo_sage.retrieval.retriever import HybridRetriever

StreamEvent = dict[str, Any]


@dataclass(slots=True)
class AnswerEngine:
    retriever: HybridRetriever
    llm: LLM
    top_k: int = 8
    history_turns: int = 6

    def _messages(self, prompt: str, history: Sequence[Message] | None) -> list[Message]:
        prior = list(history or [])[-self.history_turns :]
        return [*prior, Message("user", prompt)]

    def _finish(self, question: str, text: str, context: list[ScoredChunk]) -> Answer:
        valid, invalid = validate_citations(parse_citations(text), [s.chunk for s in context])
        return Answer(
            question=question,
            text=text,
            citations=valid,
            invalid_citations=invalid,
            context=context,
        )

    def ask(
        self,
        repo_id: str,
        repo_name: str,
        question: str,
        history: Sequence[Message] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        context = self.retriever.retrieve(repo_id, question, top_k or self.top_k)
        if not context:
            return self._finish(question, NO_CONTEXT_ANSWER, [])
        prompt = build_user_prompt(question, context, repo_name)
        text = self.llm.complete(SYSTEM_PROMPT, self._messages(prompt, history))
        return self._finish(question, text, context)

    def ask_stream(
        self,
        repo_id: str,
        repo_name: str,
        question: str,
        history: Sequence[Message] | None = None,
        top_k: int | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield ``context`` first, then ``token`` events, then a final ``answer``."""
        context = self.retriever.retrieve(repo_id, question, top_k or self.top_k)
        yield {"type": "context", "chunks": context}
        if not context:
            yield {"type": "token", "text": NO_CONTEXT_ANSWER}
            yield {"type": "answer", "answer": self._finish(question, NO_CONTEXT_ANSWER, [])}
            return
        prompt = build_user_prompt(question, context, repo_name)
        parts: list[str] = []
        for token in self.llm.stream(SYSTEM_PROMPT, self._messages(prompt, history)):
            parts.append(token)
            yield {"type": "token", "text": token}
        yield {"type": "answer", "answer": self._finish(question, "".join(parts), context)}
