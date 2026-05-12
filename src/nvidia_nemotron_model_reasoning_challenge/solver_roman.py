"""Deterministic solver for the `roman` puzzle type.

Prompt shape:

    In Alice's Wonderland, numbers are secretly converted into a different
    numeral system. Some examples are given below:
    11 -> XI
    15 -> XV
    94 -> XCIV
    19 -> XIX
    Now, write the number 38 in the Wonderland numeral system.

The "secret" turns out to be standard Roman numerals — verified against
the 1576 train puzzles in this category. Output: e.g. "XXXVIII".

Approach:

1. Parse the query integer with regex.
2. Apply the canonical 13-symbol Roman conversion.

The solver returns `None` for any prompt it cannot parse (defensive,
abstention is preferred over a confident wrong answer per Codex review
2026-05-13 §1.1).
"""

from __future__ import annotations

import re

_QUERY = re.compile(r"write the number\s+(\d+)\s+in the", re.I)

_ROMAN_VAL = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
_ROMAN_SYM = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]


def _int_to_roman(n: int) -> str:
    out: list[str] = []
    for v, s in zip(_ROMAN_VAL, _ROMAN_SYM, strict=True):
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def solve(prompt: str) -> str | None:
    m = _QUERY.search(prompt or "")
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    if n <= 0 or n > 3999:
        # Roman numerals are typically defined for 1..3999; abstain otherwise.
        return None
    return _int_to_roman(n)
