#!/usr/bin/env python3
"""Markdown link and documentation-index check.

Two checks, both cheap and deterministic:

1. **Relative links.** Every relative Markdown link target in every tracked
   `*.md` file must resolve on disk. External `http(s)` links, `mailto:` links,
   and in-page `#anchor` targets are not fetched or validated.
2. **Index coverage.** Every long-lived page directly under `docs/` must be
   linked from `docs/documentation-index.md`, so the flat index cannot silently
   fall behind the tree. `docs/README.md`, `docs/documentation-index.md`, and
   the generated `docs/_generated/` tree are exempt.

Run through the gate: `./run.sh links`.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
INDEX = DOCS_DIR / "documentation-index.md"
INDEX_EXEMPT = {"README.md", "documentation-index.md"}
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
LINK = re.compile(r"\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+\"[^\"]*\")?\s*\)")
FENCE = re.compile(r"^\s*(```|~~~)")


def tracked_markdown() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / name for name in listed.split("\0") if name]


def links_in(path: Path) -> list[tuple[int, str]]:
    """`(line_number, target)` for each inline link outside fenced code."""
    found: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK.finditer(line):
            found.append((number, match.group(1)))
    return found


def resolves(source: Path, target: str) -> bool:
    if not target or target.startswith("#") or SCHEME.match(target):
        return True
    path = urllib.parse.unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path:
        return True
    candidate = Path(path)
    if candidate.is_absolute():
        return (REPO_ROOT / candidate.relative_to("/")).exists()
    return (source.parent / candidate).exists()


def check_links(files: list[Path]) -> list[str]:
    problems = []
    for source in files:
        for number, target in links_in(source):
            if not resolves(source, target):
                problems.append(
                    f"{source.relative_to(REPO_ROOT)}:{number}: "
                    f"broken relative link -> {target}"
                )
    return problems


def check_index(files: list[Path]) -> list[str]:
    if not INDEX.exists():
        return ["docs/documentation-index.md: missing"]
    indexed = {
        urllib.parse.unquote(target.split("#", 1)[0])
        for _, target in links_in(INDEX)
        if not SCHEME.match(target)
    }
    problems = []
    for page in sorted(DOCS_DIR.glob("*.md")):
        if page.name in INDEX_EXEMPT:
            continue
        candidates = {page.name, f"docs/{page.name}"}
        if not candidates & indexed:
            problems.append(
                f"docs/{page.name}: not listed in docs/documentation-index.md"
            )
    return problems


def main() -> int:
    files = tracked_markdown()
    problems = check_links(files) + check_index(files)
    if problems:
        print(f"docs check failed ({len(problems)} problem(s)):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"docs check passed: {len(files)} Markdown files, links and index OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
