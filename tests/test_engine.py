from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from repo_sage.answer.engine import AnswerEngine
from repo_sage.answer.prompt import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, format_context
from repo_sage.cli import app as cli_app
from repo_sage.embeddings import HashingEmbedder
from repo_sage.llm import FakeLLM, Message
from repo_sage.llm.anthropic_llm import AnthropicLLM
from repo_sage.llm.openai_llm import OpenAILLM
from repo_sage.models import ScoredChunk
from repo_sage.retrieval.retriever import HybridRetriever
from repo_sage.store import InMemoryStore

SAMPLE = Path(__file__).parent / "fixtures" / "sample_repo"


def _engine(llm: FakeLLM, chunks=()) -> tuple[AnswerEngine, InMemoryStore]:
    store = InMemoryStore()
    emb = HashingEmbedder(64)
    for c in chunks:
        c.embedding = emb.embed([c.search_text()])[0]
        store.replace_file(c.repo_id, c.path, [c])
    return AnswerEngine(HybridRetriever(store, emb), llm), store


def test_prompt_numbers_lines_from_chunk_start(make_chunk):
    ctx = format_context([ScoredChunk(make_chunk(start=41, end=42, text="a = 1\nb = 2"), 1.0)])
    assert "41 | a = 1" in ctx and "42 | b = 2" in ctx
    assert "[path/to/file.ext:START-END]" in SYSTEM_PROMPT


def test_no_context_skips_the_llm():
    llm = FakeLLM()
    engine, _ = _engine(llm)
    answer = engine.ask("r", "r", "anything?")
    assert answer.text == NO_CONTEXT_ANSWER
    assert llm.calls == []
    assert not answer.grounded


def test_history_is_trimmed_and_forwarded(make_chunk):
    llm = FakeLLM()
    engine, _ = _engine(llm, [make_chunk(text="def retry(): pass", symbol="retry")])
    engine.history_turns = 2
    history = [Message("user", f"q{i}") for i in range(5)]
    engine.ask("r", "r", "retry", history=history)
    _, messages = llm.calls[0]
    assert [m.content for m in messages[:2]] == ["q3", "q4"]
    assert messages[-1].content.startswith("Repository: r")


def test_stream_events_reassemble_the_answer(make_chunk):
    engine, _ = _engine(FakeLLM(), [make_chunk(text="def retry(): pass", symbol="retry")])
    events = list(engine.ask_stream("r", "r", "retry"))
    assert events[0]["type"] == "context"
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    final = events[-1]["answer"]
    assert tokens == final.text
    assert final.grounded


def test_anthropic_provider_uses_messages_api_and_filters_text_blocks():
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="hi")],
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    llm = AnthropicLLM(_client=client)
    assert llm.complete("sys", [Message("user", "q")]) == "hi"
    assert calls["system"] == "sys"
    assert calls["model"] == "claude-opus-5-5"
    assert calls["messages"] == [{"role": "user", "content": "q"}]


def test_openai_provider_prepends_system_message():
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        msg = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = OpenAILLM(model="m", _client=client)
    assert llm.complete("sys", [Message("user", "q")]) == "ok"
    assert calls["messages"][0] == {"role": "system", "content": "sys"}


def test_cli_ask_and_eval(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REPO_SAGE_DATA_DIR", str(tmp_path))
    from repo_sage.config import get_settings

    get_settings.cache_clear()
    runner = CliRunner()
    result = runner.invoke(cli_app, ["ask", str(SAMPLE), "visibility timeout requeue", "--json"])
    assert result.exit_code == 0, result.output
    assert '"grounded": true' in result.output

    eval_file = Path(__file__).resolve().parents[1] / "evals" / "tinyqueue.yaml"
    result = runner.invoke(cli_app, ["eval", str(eval_file), "--json", "--min-mrr", "0.5"])
    assert result.exit_code == 0, result.output
    assert '"mrr"' in result.output
    get_settings.cache_clear()
