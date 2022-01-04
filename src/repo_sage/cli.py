"""Command line interface: ``repo-sage index | ask | chat | repos | eval | serve``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from repo_sage.config import get_settings
from repo_sage.llm.base import Message
from repo_sage.models import Answer, Repo
from repo_sage.services import Services

app = typer.Typer(
    name="repo-sage",
    help="Chat with any git repository. Answers cite file paths and line ranges.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _services() -> Services:
    return Services.build(get_settings())


def _resolve(svc: Services, repo: str, *, full: bool = False, quiet: bool = False) -> Repo:
    """Accept a registered repo id or a source; register and (re)index sources on demand."""
    existing = svc.store.get_repo(repo)
    if existing is not None and existing.chunk_count > 0:
        return existing
    record = existing or svc.indexer.register(repo)
    progress = None if quiet else (lambda msg: console.print(f"[dim]{msg}[/dim]"))
    svc.indexer.index(record.id, full=full, progress=progress)
    refreshed = svc.store.get_repo(record.id)
    assert refreshed is not None
    return refreshed


def _print_answer(answer: Answer) -> None:
    console.print(Markdown(answer.text))
    if answer.citations:
        console.print("\n[bold]Citations[/bold]")
        for c in answer.citations:
            console.print(f"  [cyan]{c}[/cyan]")
    if answer.invalid_citations:
        console.print("[yellow]Unverified citations (not in retrieved context):[/yellow]")
        for c in answer.invalid_citations:
            console.print(f"  [yellow]{c}[/yellow]")


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def index(
    source: Annotated[str, typer.Argument(help="Git URL or local path.")],
    name: Annotated[str | None, typer.Option(help="Display name / id seed.")] = None,
    branch: Annotated[str | None, typer.Option(help="Branch to clone for remote sources.")] = None,
    full: Annotated[bool, typer.Option(help="Ignore the last indexed commit.")] = False,
) -> None:
    """Register a repository (if new) and index it, incrementally when possible."""
    svc = _services()
    repo = svc.indexer.register(source, name=name, branch=branch)
    report = svc.indexer.index(
        repo.id, full=full, progress=lambda m: console.print(f"[dim]{m}[/dim]")
    )
    console.print(f"[green]{report.summary()}[/green]")


@app.command()
def ask(
    repo: Annotated[str, typer.Argument(help="Repo id, git URL or local path.")],
    question: Annotated[str, typer.Argument()],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Ask one question and print a cited answer."""
    svc = _services()
    record = _resolve(svc, repo, quiet=as_json)
    answer = svc.engine.ask(record.id, record.name, question, top_k=top_k)
    if as_json:
        from repo_sage.api.schemas import AnswerOut

        typer.echo(AnswerOut.of(answer).model_dump_json(indent=2))
        return
    _print_answer(answer)


@app.command()
def chat(repo: Annotated[str, typer.Argument(help="Repo id, git URL or local path.")]) -> None:
    """Interactive multi-turn chat with streamed answers. Ctrl-D to quit."""
    svc = _services()
    record = _resolve(svc, repo)
    history: list[Message] = []
    console.print(f"[bold]Chatting with {record.name}[/bold] ({record.chunk_count} chunks)")
    while True:
        try:
            question = console.input("[bold magenta]you>[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not question:
            continue
        answer: Answer | None = None
        for event in svc.engine.ask_stream(record.id, record.name, question, history):
            if event["type"] == "token":
                console.print(event["text"], end="", markup=False, highlight=False)
            elif event["type"] == "answer":
                answer = event["answer"]
        console.print()
        if answer is not None:
            for c in answer.citations:
                console.print(f"  [cyan]{c}[/cyan]")
            history += [Message("user", question), Message("assistant", answer.text)]


@app.command()
def repos() -> None:
    """List registered repositories (useful with the Postgres store)."""
    svc = _services()
    table = Table("id", "name", "commit", "files", "chunks", "indexed at")
    for r in svc.store.list_repos():
        table.add_row(
            r.id,
            r.name,
            (r.last_commit or "-")[:10],
            str(r.file_count),
            str(r.chunk_count),
            r.indexed_at.isoformat(timespec="seconds") if r.indexed_at else "-",
        )
    console.print(table)


@app.command("eval")
def run_eval_cmd(
    eval_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = Path(
        "evals/tinyqueue.yaml"
    ),
    as_json: Annotated[bool, typer.Option("--json")] = False,
    min_mrr: Annotated[float, typer.Option(help="Exit non-zero if MRR is below this.")] = 0.0,
) -> None:
    """Score retrieval on a labelled question set (recall@k, hit-rate@k, MRR)."""
    from repo_sage.embeddings.factory import build_embedder
    from repo_sage.evaluation.runner import EvalSet, run_eval
    from repo_sage.retrieval.rerank import build_reranker

    settings = get_settings()
    report = run_eval(
        EvalSet.load(eval_file),
        embedder=build_embedder(settings),
        reranker=build_reranker(settings.reranker, settings.cross_encoder_model),
    )
    summary = report.as_dict()
    if as_json:
        typer.echo(json.dumps(summary, indent=2))
    else:
        table = Table("question", "expected", "rank", "top files", title=f"eval: {report.name}")
        for case in summary["cases"]:
            rank = case["first_hit_rank"]
            table.add_row(
                case["question"],
                ", ".join(case["expected"]),
                f"[green]{rank}[/green]" if rank == 1 else f"[yellow]{rank or 'miss'}[/yellow]",
                ", ".join(case["top_files"][:3]),
            )
        console.print(table)
        metrics = "  ".join(
            [f"MRR {summary['mrr']:.3f}"]
            + [f"recall{k} {v:.3f}" for k, v in summary["recall"].items()]
            + [f"hit{k} {v:.3f}" for k, v in summary["hit_rate"].items()]
        )
        console.print(f"[bold]{metrics}[/bold]")
    if report.mrr < min_mrr:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("repo_sage.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
