from pathlib import Path

from repo_sage.ingest.filters import FileFilter
from repo_sage.ingest.git_source import is_remote, repo_name_from_source, slugify
from repo_sage.ingest.languages import detect_language


def _write(root: Path, rel: str, data: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data)


def test_filter_honours_root_and_nested_gitignores(tmp_path: Path):
    _write(tmp_path, ".gitignore", "*.log\n/generated/\n")
    _write(tmp_path, "pkg/.gitignore", "fixtures/\n!keep.log\n")
    _write(tmp_path, "main.py", "print(1)\n")
    _write(tmp_path, "debug.log", "noise\n")
    _write(tmp_path, "generated/out.py", "x = 1\n")
    _write(tmp_path, "pkg/fixtures/data.py", "x = 2\n")
    _write(tmp_path, "pkg/keep.log", "kept\n")
    _write(tmp_path, "pkg/mod.ts", "export const a = 1;\n")
    _write(tmp_path, "web/node_modules/lib/index.js", "x\n")
    paths = [sf.path for sf in FileFilter(tmp_path).walk()]
    assert paths == [".gitignore", "main.py", "pkg/.gitignore", "pkg/keep.log", "pkg/mod.ts"]


def test_filter_skips_binary_large_and_empty(tmp_path: Path):
    _write(tmp_path, "blob.dat", b"abc\x00def")
    _write(tmp_path, "big.py", "x = 1\n" * 1000)
    _write(tmp_path, "empty.py", "")
    _write(tmp_path, "ok.py", "y = 2\r\n")
    filt = FileFilter(tmp_path, max_bytes=1000)
    loaded = {sf.path: sf for sf in filt.walk()}
    assert list(loaded) == ["ok.py"]
    assert loaded["ok.py"].content == "y = 2\n"  # CRLF normalised so line numbers are stable
    assert filt.skipped == {"blob.dat": "binary", "big.py": "too-large", "empty.py": "empty"}


def test_language_detection():
    assert detect_language("a/b/c.py") == "python"
    assert detect_language("Component.TSX") == "tsx"
    assert detect_language("deploy/Dockerfile") == "dockerfile"
    assert detect_language("weird.xyz") is None


def test_source_naming():
    assert is_remote("https://github.com/org/proj.git")
    assert is_remote("git@github.com:org/proj.git")
    assert not is_remote("C:/code/proj")
    assert repo_name_from_source("https://github.com/org/proj.git") == "org/proj"
    assert repo_name_from_source("git@github.com:org/proj.git") == "org/proj"
    assert slugify("Org/My_Proj") == "org-my-proj"
