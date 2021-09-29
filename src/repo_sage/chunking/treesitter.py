"""Syntax-aware chunking with tree-sitter.

Definitions (functions, classes, methods, impls, interfaces...) become one chunk each,
labelled with a qualified symbol name such as ``Scheduler.enqueue``. Containers that
are too large to fit in one chunk are split into their members, and any leftover
lines (imports, module constants, class fields) are kept as their own chunks so that
nothing in the file becomes unsearchable.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cache

from tree_sitter import Language, Node, Parser

from repo_sage.chunking.lines import is_blank_span, split_span, trim_span
from repo_sage.models import Chunk, ChunkKind, SourceFile

_GRAMMARS: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
}

_COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment"})
_TRIVIAL_LINES = frozenset({"}", "};", ")", ");", "]", "end", "})", "});"})

_JS_DEFS: dict[str, ChunkKind] = {
    "function_declaration": ChunkKind.FUNCTION,
    "generator_function_declaration": ChunkKind.FUNCTION,
    "class_declaration": ChunkKind.CLASS,
    "abstract_class_declaration": ChunkKind.CLASS,
    "method_definition": ChunkKind.METHOD,
    "interface_declaration": ChunkKind.CLASS,
    "enum_declaration": ChunkKind.CLASS,
    "type_alias_declaration": ChunkKind.CLASS,
}


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    definitions: dict[str, ChunkKind]
    wrappers: dict[str, str] = field(default_factory=dict)
    """Node types that wrap a definition, mapped to the field holding it."""
    bodies: dict[str, str] = field(default_factory=dict)
    """Container node types mapped to the field holding their members."""
    namer: Callable[[Node], str | None] | None = None


def _text(node: Node | None) -> str | None:
    if node is None or node.text is None:
        return None
    return node.text.decode("utf-8", errors="replace")


def _default_name(node: Node) -> str | None:
    return _text(node.child_by_field_name("name"))


def _go_name(node: Node) -> str | None:
    if node.type == "type_declaration":
        for child in node.named_children:
            if child.type in {"type_spec", "type_alias"}:
                return _text(child.child_by_field_name("name"))
        return None
    name = _default_name(node)
    if node.type == "method_declaration":
        receiver = node.child_by_field_name("receiver")
        recv_type = _first_descendant(receiver, "type_identifier")
        if recv_type is not None and name:
            return f"{_text(recv_type)}.{name}"
    return name


def _rust_name(node: Node) -> str | None:
    if node.type == "impl_item":
        type_name = _text(node.child_by_field_name("type"))
        trait = _text(node.child_by_field_name("trait"))
        return f"{type_name} ({trait})" if trait and type_name else type_name
    return _default_name(node)


def _first_descendant(node: Node | None, node_type: str) -> Node | None:
    if node is None:
        return None
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type == node_type:
            return cur
        stack.extend(reversed(cur.named_children))
    return None


SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        definitions={
            "function_definition": ChunkKind.FUNCTION,
            "class_definition": ChunkKind.CLASS,
        },
        wrappers={"decorated_definition": "definition"},
        bodies={"class_definition": "body"},
    ),
    "javascript": LanguageSpec(
        definitions=_JS_DEFS,
        wrappers={"export_statement": "declaration"},
        bodies={"class_declaration": "body"},
    ),
    "typescript": LanguageSpec(
        definitions=_JS_DEFS,
        wrappers={"export_statement": "declaration"},
        bodies={
            "class_declaration": "body",
            "abstract_class_declaration": "body",
            "interface_declaration": "body",
        },
    ),
    "go": LanguageSpec(
        definitions={
            "function_declaration": ChunkKind.FUNCTION,
            "method_declaration": ChunkKind.METHOD,
            "type_declaration": ChunkKind.CLASS,
        },
        namer=_go_name,
    ),
    "rust": LanguageSpec(
        definitions={
            "function_item": ChunkKind.FUNCTION,
            "struct_item": ChunkKind.CLASS,
            "enum_item": ChunkKind.CLASS,
            "trait_item": ChunkKind.CLASS,
            "impl_item": ChunkKind.CLASS,
            "mod_item": ChunkKind.MODULE,
        },
        bodies={"impl_item": "body", "trait_item": "body", "mod_item": "body"},
        namer=_rust_name,
    ),
    "java": LanguageSpec(
        definitions={
            "class_declaration": ChunkKind.CLASS,
            "interface_declaration": ChunkKind.CLASS,
            "enum_declaration": ChunkKind.CLASS,
            "record_declaration": ChunkKind.CLASS,
            "method_declaration": ChunkKind.METHOD,
            "constructor_declaration": ChunkKind.METHOD,
        },
        bodies={
            "class_declaration": "body",
            "interface_declaration": "body",
            "enum_declaration": "body",
            "record_declaration": "body",
        },
    ),
}
SPECS["tsx"] = SPECS["typescript"]


@cache
def get_parser(language: str) -> Parser | None:
    """Return a cached parser, or ``None`` if the grammar package is unavailable."""
    target = _GRAMMARS.get(language)
    if target is None:
        return None
    module_name, attr = target
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return Parser(Language(getattr(module, attr)()))


Span = tuple[int, int]


@dataclass(slots=True)
class TreeSitterChunker:
    max_lines: int = 80
    window: int = 60
    overlap: int = 10
    max_depth: int = 3

    def supports(self, language: str | None) -> bool:
        return language in SPECS and get_parser(language) is not None

    def chunk(self, repo_id: str, source: SourceFile) -> list[Chunk]:
        language = source.language
        if language is None or not self.supports(language):
            return []
        parser = get_parser(language)
        assert parser is not None
        tree = parser.parse(source.content.encode("utf-8"))
        lines = source.content.split("\n")
        ctx = _Ctx(repo_id=repo_id, source=source, lines=lines, spec=SPECS[language])
        chunks, covered = self._walk(ctx, tree.root_node.named_children, parent=None, depth=0)
        chunks.extend(self._gaps(ctx, 1, len(lines), covered, ChunkKind.MODULE, None))
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    # -- traversal -----------------------------------------------------------------

    def _walk(
        self, ctx: _Ctx, nodes: Iterable[Node], parent: str | None, depth: int
    ) -> tuple[list[Chunk], list[Span]]:
        chunks: list[Chunk] = []
        covered: list[Span] = []
        for node in nodes:
            found = self._definition(ctx.spec, node)
            if found is None:
                # e.g. `const handler = () => {}` in JS/TS
                found = self._js_variable_function(node)
            if found is None:
                continue
            kind, name, inner = found
            symbol = f"{parent}.{name}" if parent and name else (name or parent)
            if parent and kind is ChunkKind.FUNCTION:
                kind = ChunkKind.METHOD
            start = self._leading_comment_start(node)
            end = node.end_point.row + 1
            covered.append((start, end))

            body_field = ctx.spec.bodies.get(inner.type)
            body = inner.child_by_field_name(body_field) if body_field else None
            if end - start + 1 <= self.max_lines:
                chunks.append(ctx.make(start, end, kind, symbol))
            elif body is not None and depth < self.max_depth:
                sub_chunks, sub_covered = self._walk(ctx, body.named_children, symbol, depth + 1)
                chunks.extend(sub_chunks)
                chunks.extend(self._gaps(ctx, start, end, sub_covered, kind, symbol))
            else:
                for s, e in split_span(ctx.lines, start, end, self.window, self.overlap):
                    chunks.append(ctx.make(s, e, kind, symbol))
        return chunks, covered

    @staticmethod
    def _definition(spec: LanguageSpec, node: Node) -> tuple[ChunkKind, str | None, Node] | None:
        inner = node
        if node.type in spec.wrappers:
            wrapped = node.child_by_field_name(spec.wrappers[node.type])
            if wrapped is None:
                return None
            inner = wrapped
        kind = spec.definitions.get(inner.type)
        if kind is None:
            return None
        namer = spec.namer or _default_name
        return kind, namer(inner), inner

    @staticmethod
    def _js_variable_function(node: Node) -> tuple[ChunkKind, str | None, Node] | None:
        target = node
        if target.type == "export_statement":
            decl = target.child_by_field_name("declaration")
            if decl is None:
                return None
            target = decl
        if target.type not in {"lexical_declaration", "variable_declaration"}:
            return None
        for declarator in target.named_children:
            if declarator.type != "variable_declarator":
                continue
            value = declarator.child_by_field_name("value")
            if value is None:
                continue
            if value.type in {"arrow_function", "function_expression", "function"}:
                return ChunkKind.FUNCTION, _default_name(declarator), value
            if value.type == "class":
                return ChunkKind.CLASS, _default_name(declarator), value
        return None

    @staticmethod
    def _leading_comment_start(node: Node) -> int:
        """Extend a definition upwards over directly attached comments / doc comments."""
        start_row = node.start_point.row
        prev = node.prev_sibling
        while prev is not None and prev.type in _COMMENT_TYPES:
            if prev.end_point.row < start_row - 1:
                break
            start_row = prev.start_point.row
            prev = prev.prev_sibling
        return start_row + 1

    def _gaps(
        self,
        ctx: _Ctx,
        start: int,
        end: int,
        covered: list[Span],
        kind: ChunkKind,
        symbol: str | None,
    ) -> list[Chunk]:
        """Emit chunks for lines in ``start..end`` not covered by any definition."""
        chunks: list[Chunk] = []
        cursor = start
        for c_start, c_end in [*sorted(covered), (end + 1, end + 1)]:
            if c_start > cursor:
                span = trim_span(ctx.lines, cursor, min(c_start - 1, end))
                if span is not None and not self._trivial(ctx.lines, *span):
                    for s, e in split_span(ctx.lines, *span, self.window, self.overlap):
                        if not is_blank_span(ctx.lines, s, e):
                            chunks.append(ctx.make(s, e, kind, symbol))
            cursor = max(cursor, c_end + 1)
        return chunks

    @staticmethod
    def _trivial(lines: list[str], start: int, end: int) -> bool:
        return all(lines[i - 1].strip() in _TRIVIAL_LINES for i in range(start, end + 1))


@dataclass(slots=True)
class _Ctx:
    repo_id: str
    source: SourceFile
    lines: list[str]
    spec: LanguageSpec

    def make(self, start: int, end: int, kind: ChunkKind, symbol: str | None) -> Chunk:
        return Chunk(
            repo_id=self.repo_id,
            path=self.source.path,
            start_line=start,
            end_line=end,
            text="\n".join(self.lines[start - 1 : end]),
            language=self.source.language,
            kind=kind,
            symbol=symbol,
        )
