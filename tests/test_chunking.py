from textwrap import dedent

import pytest

from repo_sage.chunking import CodeChunker, LineWindowChunker
from repo_sage.chunking.lines import split_span
from repo_sage.models import ChunkKind, SourceFile


def _chunk(path: str, language: str | None, code: str, **kw: int):
    return CodeChunker(**kw).chunk("repo", SourceFile(path, language, dedent(code).lstrip("\n")))


def _summary(chunks):
    return [(c.kind, c.symbol, c.start_line, c.end_line) for c in chunks]


def test_python_functions_and_classes_get_symbols_and_lines():
    chunks = _chunk(
        "pkg/mod.py",
        "python",
        """
        import os

        LIMIT = 3


        # attached comment
        @decorator
        def top(a):
            return a


        class Box:
            def open(self):
                return True
        """,
    )
    assert _summary(chunks) == [
        (ChunkKind.MODULE, None, 1, 3),
        (ChunkKind.FUNCTION, "top", 6, 9),
        (ChunkKind.CLASS, "Box", 12, 14),
    ]
    assert chunks[1].text.startswith("# attached comment")


def test_large_class_is_split_into_methods_and_header():
    methods = "\n".join(
        f"    def m{i}(self):\n        x = {i}\n        return x\n" for i in range(6)
    )
    code = f"class Big:\n    '''Docstring.'''\n    size = 1\n\n{methods}"
    chunks = CodeChunker(max_lines=10).chunk("r", SourceFile("big.py", "python", code))
    symbols = [c.symbol for c in chunks]
    assert symbols[0] == "Big"
    assert chunks[0].kind is ChunkKind.CLASS
    assert chunks[0].start_line == 1
    assert [s for s in symbols if s and s.startswith("Big.m")] == [f"Big.m{i}" for i in range(6)]
    assert all(c.kind is ChunkKind.METHOD for c in chunks[1:])


def test_oversized_function_is_windowed_but_keeps_symbol():
    body = "\n".join(f"    v{i} = {i}" for i in range(50))
    code = f"def huge():\n{body}\n    return 0\n"
    chunks = CodeChunker(max_lines=20, window=15, overlap=3).chunk(
        "r", SourceFile("h.py", "python", code)
    )
    assert len(chunks) > 1
    assert {c.symbol for c in chunks} == {"huge"}
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 52


def test_typescript_exports_arrow_functions_and_interfaces():
    chunks = _chunk(
        "src/queue.ts",
        "typescript",
        """
        import { clock } from "./clock";

        export const backoff = (attempt: number): number => 2 ** attempt;

        export interface Job {
          id: string;
        }

        export class Queue {
          private jobs: Job[] = [];
          push(job: Job): void {
            this.jobs.push(job);
          }
        }
        """,
    )
    assert [(c.symbol, c.kind) for c in chunks] == [
        (None, ChunkKind.MODULE),
        ("backoff", ChunkKind.FUNCTION),
        ("Job", ChunkKind.CLASS),
        ("Queue", ChunkKind.CLASS),
    ]


def test_go_methods_are_qualified_by_receiver():
    chunks = _chunk(
        "stack.go",
        "go",
        """
        package stack

        type Stack struct{ items []int }

        func (s *Stack) Push(v int) { s.items = append(s.items, v) }

        func New() *Stack { return &Stack{} }
        """,
    )
    assert [c.symbol for c in chunks] == [None, "Stack", "Stack.Push", "New"]


def test_rust_impl_and_java_methods():
    rust = _chunk(
        "lib.rs",
        "rust",
        """
        pub struct Ring { cap: usize }

        impl Ring {
            pub fn new(cap: usize) -> Self { Ring { cap } }
        }
        """,
    )
    assert [c.symbol for c in rust] == ["Ring", "Ring"]

    java = _chunk(
        "Greeter.java",
        "java",
        """
        package demo;

        public class Greeter {
            private final String name;

            public Greeter(String name) { this.name = name; }

            public String greet() { return "hi " + name; }
        }
        """,
        max_lines=4,
    )
    symbols = [c.symbol for c in java]
    assert "Greeter.greet" in symbols
    assert "Greeter.Greeter" in symbols
    assert symbols[0] is None  # package declaration


def test_unsupported_language_falls_back_to_windows():
    code = "\n".join(f"line {i}" for i in range(1, 131))
    chunks = _chunk("notes.txt", "text", code, window=60, overlap=10)
    assert all(c.kind is ChunkKind.WINDOW for c in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 130
    # consecutive windows overlap so a fact on a boundary is never split in two
    assert chunks[1].start_line <= chunks[0].end_line


def test_markdown_sections_use_heading_as_symbol():
    md = "# Title\n\nIntro.\n\n## Install\n\npip install x\n\n```\n# not a heading\n```\n"
    chunks = LineWindowChunker().chunk("r", SourceFile("README.md", "markdown", md))
    assert [c.symbol for c in chunks] == ["Title", "Install"]
    assert chunks[1].start_line == 5


@pytest.mark.parametrize(("window", "overlap"), [(10, 0), (10, 3), (7, 6)])
def test_split_span_covers_every_line(window: int, overlap: int):
    lines = ["x"] * 45
    spans = split_span(lines, 1, 45, window, overlap)
    covered = {i for s, e in spans for i in range(s, e + 1)}
    assert covered == set(range(1, 46))
    assert all(e - s + 1 <= window for s, e in spans)


def test_empty_file_produces_no_chunks():
    assert _chunk("empty.py", "python", "\n\n") == []
