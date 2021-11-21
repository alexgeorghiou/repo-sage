from repo_sage.answer.citations import parse_citations, validate_citations
from repo_sage.models import Citation

EN_DASH = chr(0x2013)


def test_parses_single_ranges_lists_and_line_prefixes():
    text = (
        "Jobs are retried [tinyqueue/retry.py:10-24]. The queue uses a heap "
        "[tinyqueue/queue.py:5, tinyqueue/queue.py:L30-L41; client/src/client.ts:7-9]."
    )
    assert parse_citations(text) == [
        Citation("tinyqueue/retry.py", 10, 24),
        Citation("tinyqueue/queue.py", 5, 5),
        Citation("tinyqueue/queue.py", 30, 41),
        Citation("client/src/client.ts", 7, 9),
    ]


def test_ignores_markdown_links_prose_and_duplicates():
    text = (
        "See [the docs](https://example.com/a.py:1-2) and [note: nothing here]. "
        "[a.py:3-4] repeated [a.py:3-4] and backticked [`b.go:9-7`]."
    )
    assert parse_citations(text) == [Citation("a.py", 3, 4), Citation("b.go", 7, 9)]


def test_keeps_dotfile_paths_and_en_dash_ranges():
    cites = parse_citations(f"CI runs ruff [.github/workflows/ci.yml:3{EN_DASH}8].")
    assert cites == [Citation(".github/workflows/ci.yml", 3, 8)]


def test_validation_requires_range_inside_retrieved_chunk(make_chunk):
    context = [make_chunk(path="a.py", start=10, end=30), make_chunk(path="b.py", start=1, end=5)]
    cites = [
        Citation("a.py", 12, 20),  # inside
        Citation("a.py", 9, 31),  # within slack
        Citation("a.py", 40, 50),  # outside the chunk
        Citation("c.py", 1, 2),  # file never shown
    ]
    valid, invalid = validate_citations(cites, context)
    assert valid == cites[:2]
    assert invalid == cites[2:]


def test_citation_formatting_and_overlap():
    assert str(Citation("a.py", 3, 3)) == "a.py:3"
    assert str(Citation("a.py", 3, 9)) == "a.py:3-9"
    assert Citation("a.py", 1, 5).overlaps(Citation("a.py", 5, 8))
    assert not Citation("a.py", 1, 5).overlaps(Citation("b.py", 1, 5))
