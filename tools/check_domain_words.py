#!/usr/bin/env python3
"""Keep domain vocabulary out of the transaction core.

Run without arguments to scan the entire core. ``--core-dir`` selects a source
tree for negative controls. Fixtures under tests/core/lint_fixtures are always
excluded. Waivers belong only in domain_words_allowlist.toml, using [[entries]]
with file, token, reason and exactly one of symbol or line. File paths are
relative to the repository or the selected core directory; line text is exact
apart from surrounding whitespace, and symbols use qualified Python names.
"""

import argparse
import ast
import io
import re
import sys
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "src/spaceodyssey/core"
ALLOWLIST = REPO / "tools/domain_words_allowlist.toml"
FIXTURES = REPO / "tests/core/lint_fixtures"
DENIED = frozenset(
    "planet star exoplanet coronagraph telescope exposure slew keepout band "
    "wavelength flux magnitude orbit ephemeris exozodi tdb visit mission target "
    "facility nondetection completeness photometry detection "
    # Agenda levels are numbers in the core; their role names stay outside it.
    "planner allocator integration".split()
)
PLURALS = {
    word[:-1] + "ies"
    if word.endswith("y")
    else word + "es"
    if word.endswith(("s", "x", "z", "ch", "sh"))
    else word + "s"
    for word in DENIED
} | {"ephemerides"}
WORDS = re.compile(
    r"[A-Z]{2,}s(?=[A-Z]|[^a-z]|$)|[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)"
    r"|[A-Z]?[a-z]+|[A-Z]+"
)
STRING_START = re.compile("(?i)^([rubf]*)(\"\"\"|'''|\"|')")
ESCAPES = re.compile(
    r"\\(?:\r?\n|[0-7]{1,3}|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}"
    r"|U[0-9a-fA-F]{8}|N\{[^}]+\}|.)",
    re.DOTALL,
)


@dataclass(frozen=True, order=True)
class Hit:
    """A denied whole token at a source line."""

    line: int
    token: str


def core_files(root):
    """Find Python sources, excluding the negative-control fixture path."""
    return sorted(
        path
        for path in root.rglob("*.py")
        if not path.resolve().is_relative_to(FIXTURES)
    )


def denied_hits(text, first_line):
    """Split prose and identifiers at case, underscore and digit boundaries."""
    for offset, line in enumerate(text.splitlines()):
        for match in WORDS.finditer(line):
            word = match[0].lower()
            if word in DENIED or word in PLURALS:
                yield Hit(first_line + offset, word)


def literal_characters(token):
    """Map decoded characters to physical lines, preserving escape provenance."""
    start = STRING_START.match(token.string)
    prefix, quote = start.groups()
    if "f" in prefix.lower():
        return []
    body = token.string[start.end() : -len(quote)]
    line = token.start[0]
    characters = []
    cursor = 0
    escapes = [] if "r" in prefix.lower() else list(ESCAPES.finditer(body))
    for escape in escapes:
        for char in body[cursor : escape.start()]:
            characters.append((char, line))
            line += char == "\n"
        fragment = escape[0]
        literal = ('b"' if "b" in prefix.lower() else '"') + fragment + '"'
        value = ast.literal_eval(literal)
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace")
        characters.extend((char, line) for char in value)
        line += fragment.count("\n")
        cursor = escape.end()
    for char in body[cursor:]:
        characters.append((char, line))
        line += char == "\n"
    return characters


def literal_hits(node, tokens):
    """Locate decoded tokens, including ones spanning adjacent literals."""
    value = node.value
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    characters = [
        char
        for token in tokens
        if token.type == tokenize.STRING
        and token.start >= (node.lineno, node.col_offset)
        and token.end <= (node.end_lineno, node.end_col_offset)
        for char in literal_characters(token)
    ]
    # F-string constant spans vary across Python versions. Their AST start is
    # always a real source line, even when the tokenizer exposes no fragments.
    lines = (
        [line for _, line in characters]
        if "".join(char for char, _ in characters) == value
        else [node.lineno] * len(value)
    )
    for match in WORDS.finditer(value):
        word = match[0].lower()
        if word in DENIED or word in PLURALS:
            yield Hit(lines[match.start()], word)


def scan_source(source, filename):
    """Read identifiers and comments lexically and decoded literals via AST."""
    tree = ast.parse(source, filename=filename)
    hits = set()
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for token in tokens:
        if token.type in (tokenize.NAME, tokenize.COMMENT):
            hits.update(denied_hits(token.string, token.start[0]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            hits.update(literal_hits(node, tokens))
        elif isinstance(node, ast.Name):
            hits.update(denied_hits(node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            hits.update(denied_hits(node.attr, node.end_lineno))
        elif isinstance(node, ast.arg):
            hits.update(denied_hits(node.arg, node.lineno))
        elif isinstance(node, ast.keyword) and node.arg:
            hits.update(denied_hits(node.arg, node.lineno))
    return sorted(hits), tree


def load_allowlist(path):
    """Reject broad, malformed or poorly explained exceptions."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if set(data) - {"entries"} or not isinstance(entries, list) or len(entries) > 3:
        raise ValueError("allowlist must contain at most 3 entries")
    for entry in entries:
        required = ("file", "token", "reason")
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip()
            for key in required
        ):
            raise ValueError("allowlist entries need file, token and reason")
        if len(entry["reason"].strip()) < 20:
            raise ValueError("allowlist reason must be at least 20 characters")
        if set(entry) - {*required, "symbol", "line"} or (
            ("symbol" in entry) == ("line" in entry)
        ):
            raise ValueError("allowlist entries need exactly one of symbol or line")
        scope = entry.get("symbol", entry.get("line"))
        file = Path(entry["file"])
        if (
            not isinstance(scope, str)
            or not scope.strip()
            or file.is_absolute()
            or ".." in file.parts
            or entry["token"] not in DENIED | PLURALS
        ):
            raise ValueError("allowlist file, token or scope is invalid")
    return entries


def symbol_ranges(tree, prefix=""):
    """Associate declarations with their qualified names and source extent."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{node.name}" if prefix else node.name
            yield name, node.lineno, node.end_lineno
            yield from symbol_ranges(node, name)
        else:
            yield from symbol_ranges(node, prefix)


def waived(entry, hit, relative, lines, symbols):
    """Match one exception to its file, exact token and explicit scope."""
    if entry["file"] not in (relative, f"src/spaceodyssey/core/{relative}"):
        return False
    if entry["token"] != hit.token:
        return False
    if "line" in entry:
        return entry["line"].strip() == lines[hit.line - 1].strip()
    return any(
        name == entry["symbol"] and start <= hit.line <= end
        for name, start, end in symbols
    )


def main(argv):
    """Scan the whole core and fail on hits, stale exceptions or scan errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=CORE)
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    args = parser.parse_args(argv)
    try:
        entries = load_allowlist(args.allowlist)
    except (OSError, ValueError) as error:
        print(f"{args.allowlist}:0:invalid allowlist: {error}")
        return 1
    files = core_files(args.core_dir)
    used = set()
    failed = not files
    if not files:
        print(f"{args.core_dir}:0:no Python sources found")
    for path in files:
        try:
            with tokenize.open(path) as stream:
                source = stream.read()
            hits, tree = scan_source(source, str(path))
        except (OSError, SyntaxError, UnicodeError, tokenize.TokenError) as error:
            print(f"{path}:{getattr(error, 'lineno', 0)}:cannot scan: {error}")
            failed = True
            continue
        relative = path.relative_to(args.core_dir).as_posix()
        lines = source.splitlines()
        symbols = list(symbol_ranges(tree))
        for hit in hits:
            matches = {
                i
                for i, entry in enumerate(entries)
                if waived(entry, hit, relative, lines, symbols)
            }
            used.update(matches)
            if not matches:
                print(f"{path}:{hit.line}:{hit.token}")
                failed = True
    for i, entry in enumerate(entries):
        if i not in used:
            print(
                f"{args.allowlist}:0:stale allowlist entry: "
                f"{entry['file']}:{entry['token']}"
            )
            failed = True
    print(f"Scanned {len(files)} Python files for domain words.")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
