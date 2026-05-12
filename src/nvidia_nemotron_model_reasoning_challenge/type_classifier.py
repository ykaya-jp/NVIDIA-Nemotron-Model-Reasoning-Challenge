"""Six-type puzzle classifier for Nemotron Reasoning Challenge.

Six categories, perfectly balanced across the 9500 train puzzles:

* roman     — Wonderland numeral system, "11 -> XI", etc.
* physics   — Gravitational, distance = 0.5 * g * t**2
* unit      — "10.08 m becomes 6.69", linear ratio per puzzle
* cipher    — Substitution cipher on natural-language text
* bit       — 8-bit binary -> 8-bit binary, latent boolean rule
* equation  — Cryptarithm / symbolic transformation rules on
              equation-like strings ("`!*[{ = '\"[", etc.)

Keyword regex is deterministic; verified to fully cover the 9500 train
rows (no `unknown`).
"""

from __future__ import annotations

import re

PuzzleType = str  # "roman" | "physics" | "unit" | "cipher" | "bit" | "equation"

# Precompiled, case-insensitive patterns. Order matters — `bit` is checked
# before `equation` because the bit prompt also says "transformation".
_PATTERNS: list[tuple[re.Pattern[str], PuzzleType]] = [
    (re.compile(r"bit manipulation|8-bit binary", re.I), "bit"),
    (re.compile(r"numeral system|secretly converted .* numeral", re.I), "roman"),
    (re.compile(r"unit conversion| m becomes ", re.I), "unit"),
    (re.compile(r"gravitational|distance\s*=\s*0\.5|t\s*=\s*[\d.]+s", re.I), "physics"),
    (re.compile(r"encrypt|decrypt|encryption", re.I), "cipher"),
    (
        re.compile(
            r"transformation rules .* applied to equations|" r"symbol(?:ic)?|expressions transform",
            re.I,
        ),
        "equation",
    ),
]


def classify(prompt: str) -> PuzzleType:
    """Return one of the six puzzle types, or "unknown" on failure.

    The keyword set is exhaustive on the 9500 train rows; "unknown" is a
    defensive fallback that should never trigger in practice. If it does,
    the hybrid router must fall back to the LLM, not to a wrong solver.
    """
    if not isinstance(prompt, str) or not prompt:
        return "unknown"
    for pat, label in _PATTERNS:
        if pat.search(prompt):
            return label
    return "unknown"


def classify_all(prompts):
    """Vectorised helper for pandas Series / list[str]."""
    return [classify(p) for p in prompts]
