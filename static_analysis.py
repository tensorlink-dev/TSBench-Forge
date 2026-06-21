"""Static analysis of miner submissions: the hardcoding gate.

Anti-gaming role
----------------
The live/augment/splice/seed layers make it impossible to *know* the challenges
ahead of time. This layer closes the complementary vector: a submission that
ignores its input and returns canned answers, or that phones home for them, or
that rewrites itself at runtime. None of these is a forecaster; all are trivially
flagged from the source without executing it.

``scan_submission`` returns a list of human-readable findings (empty == clean):

* **Embedded numeric arrays** (>= ``ARRAY_FLAG_THRESHOLD`` values) -- memorised
  data or lookup tables baked into the code.
* **Network imports** (``socket``/``http``/``urllib``/``requests``/...) -- a
  submission that forecasts honestly never needs the network.
* **Dynamic execution** (``eval``/``exec``/``__import__``/``compile``) -- used to
  smuggle behaviour past static analysis.
* **File writes** (``open(..., 'w')`` and friends) -- persisting state across
  challenges to defeat per-challenge isolation.

It is intentionally an AST analysis (robust to formatting) backed by a couple of
targeted regexes (robust to code that does not parse).
"""

from __future__ import annotations

import ast
import re

# A literal sequence with at least this many numbers looks like baked-in data.
ARRAY_FLAG_THRESHOLD = 20

# Modules whose presence indicates a submission reaching outside its sandbox.
_NETWORK_MODULES = {
    "socket",
    "http",
    "urllib",
    "urllib2",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "telnetlib",
    "smtplib",
    "asyncio",
    "websocket",
    "websockets",
    "paramiko",
    "boto3",
}

_DYNAMIC_CALLS = {"eval", "exec", "compile", "__import__"}

# Write-ish file modes (any of these characters in the mode string).
_WRITE_MODE_RE = re.compile(r"[wax+]")


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _count_numbers(node: ast.AST) -> int:
    """Count numeric constants reachable through nested list/tuple literals."""
    total = 0
    for elt in getattr(node, "elts", []):
        if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float, complex)):
            total += 1
        elif isinstance(elt, (ast.List, ast.Tuple)):
            total += _count_numbers(elt)
        elif (
            isinstance(elt, ast.UnaryOp)
            and isinstance(elt.operand, ast.Constant)
            and isinstance(elt.operand.value, (int, float, complex))
        ):
            total += 1  # negative literals like -1.5
    return total


class _Scanner(ast.NodeVisitor):
    """Collect findings while walking the AST once."""

    def __init__(self) -> None:
        self.findings: list[str] = []
        self._inside_literal = 0

    def _flag(self, lineno: int, msg: str) -> None:
        self.findings.append(f"line {lineno}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _root_module(alias.name) in _NETWORK_MODULES:
                self._flag(node.lineno, f"network import '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and _root_module(node.module) in _NETWORK_MODULES:
            self._flag(node.lineno, f"network import from '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _DYNAMIC_CALLS:
            self._flag(node.lineno, f"dynamic execution '{func.id}(...)'")
        if self._is_write_open(node):
            self._flag(node.lineno, "file write via open(..., mode) with a write mode")
        self.generic_visit(node)

    @staticmethod
    def _is_write_open(node: ast.Call) -> bool:
        func = node.func
        is_open = (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "open"
        )
        if not is_open:
            return False
        mode = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = node.args[1].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        return isinstance(mode, str) and bool(_WRITE_MODE_RE.search(mode))

    def _visit_seq(self, node: ast.AST) -> None:
        # Only count a top-level literal (not one nested inside another), so a big
        # table is reported once rather than once per row.
        if self._inside_literal == 0:
            count = _count_numbers(node)
            if count >= ARRAY_FLAG_THRESHOLD:
                self._flag(
                    getattr(node, "lineno", 0),
                    f"embedded numeric array with {count} values "
                    f"(>= {ARRAY_FLAG_THRESHOLD}); looks like memorised data",
                )
        self._inside_literal += 1
        self.generic_visit(node)
        self._inside_literal -= 1

    visit_List = _visit_seq
    visit_Tuple = _visit_seq


def _regex_fallback(code: str) -> list[str]:
    """Catch the same vectors when the source does not parse as Python."""
    findings: list[str] = []
    if re.search(r"\b(eval|exec|compile|__import__)\s*\(", code):
        findings.append("regex: dynamic execution call present")
    if re.search(r"\b(import|from)\s+(" + "|".join(sorted(_NETWORK_MODULES)) + r")\b", code):
        findings.append("regex: network import present")
    if re.search(r"open\s*\([^)]*,\s*['\"][wax+][^'\"]*['\"]", code):
        findings.append("regex: file write via open(..., 'w')")
    if re.search(r"[\[(]\s*(?:-?\d+(?:\.\d+)?\s*,\s*){19,}", code):
        findings.append("regex: long embedded numeric array")
    return findings


def scan_submission(code: str) -> list[str]:
    """Scan submission source and return a list of anti-gaming findings.

    Empty list == clean. A clean numpy forecaster (imports numpy, reads its
    input, returns an array) produces no findings; a submission with a baked-in
    table, a ``requests`` import, or an ``eval`` produces one finding each.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _regex_fallback(code) or ["unparseable submission (could not AST-parse)"]
    scanner = _Scanner()
    scanner.visit(tree)
    return scanner.findings
