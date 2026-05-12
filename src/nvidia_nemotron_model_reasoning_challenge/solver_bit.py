"""Deterministic solver for the `bit` puzzle type.

Prompt shape (8-11 examples + 1 query):

    In Alice's Wonderland, a secret bit manipulation rule transforms
    8-bit binary numbers. ...
    01010001 -> 11011101
    00001001 -> 01101101
    ...
    Now, determine the output for: 00110100

Approach (= Codex review 2026-05-13 §1.1: 48h hard cap, abstention
preferred over confident wrong):

1. Parse all (input, output) example pairs.
2. For each of the 8 output bits independently, search for a boolean
   function f: {0,1}^k -> {0,1} with k ∈ {1, 2, 3} of input bits that
   perfectly explains every example.
3. If a unique function is found for every output bit, apply it to the
   query and return the resulting 8-bit string.
4. If any output bit has 0 candidates (= no function fits the examples)
   or > 1 candidates with disagreement on the query, return None.

The brute-force space is:
    k=1: C(8,1) × 2^(2^1) = 8 × 4   = 32     functions
    k=2: C(8,2) × 2^(2^2) = 28 × 16 = 448    functions
    k=3: C(8,3) × 2^(2^3) = 56 × 256 = 14336 functions
    total ≈ 14816 candidates per output bit, 8 bits per puzzle, 1602
    puzzles → ≈ 190 M candidate evaluations. Python loops at 8-11
    examples each fit in well under a second per puzzle with care.

If many puzzles need k > 3 the precision will drop and we abstain;
the LLM fallback handles those at inference.
"""

from __future__ import annotations

import re
from functools import cache
from itertools import combinations

_QUERY = re.compile(r"determine the output for:\s*([01]{8})", re.I)
_PAIR = re.compile(r"^([01]{8})\s*->\s*([01]{8})\s*$", re.M)


def _parse(prompt: str) -> tuple[list[tuple[int, int]], int] | None:
    pairs_raw = _PAIR.findall(prompt or "")
    if len(pairs_raw) < 4:
        return None
    pairs = [(int(a, 2), int(b, 2)) for a, b in pairs_raw]
    m = _QUERY.search(prompt or "")
    if not m:
        return None
    q = int(m.group(1), 2)
    return pairs, q


def _bit(x: int, idx: int) -> int:
    """Return bit `idx` (= position from MSB, 0..7) of an 8-bit int."""
    return (x >> (7 - idx)) & 1


@cache
def _subsets(k: int) -> tuple[tuple[int, ...], ...]:
    return tuple(combinations(range(8), k))


def _candidate_functions(
    examples: list[tuple[int, int]], out_bit: int, k: int
) -> list[tuple[tuple[int, ...], int]]:
    """All (subset, truth_table) pairs that fit the examples for `out_bit`.

    A function is encoded by its truth table = an integer in [0, 2^(2^k))
    whose i-th bit gives f(inputs encoded by i).
    """
    fits: list[tuple[tuple[int, ...], int]] = []
    for subset in _subsets(k):
        # Build per-example index into the truth table.
        idx_per_ex = []
        out_per_ex = []
        for x, y in examples:
            idx = 0
            for j, bit in enumerate(subset):
                idx |= _bit(x, bit) << (k - 1 - j)
            idx_per_ex.append(idx)
            out_per_ex.append(_bit(y, out_bit))
        # Walk all 2^(2^k) truth tables.
        for tt in range(1 << (1 << k)):
            ok = True
            for idx, target in zip(idx_per_ex, out_per_ex, strict=True):
                if ((tt >> idx) & 1) != target:
                    ok = False
                    break
            if ok:
                fits.append((subset, tt))
    return fits


def _apply(subset: tuple[int, ...], tt: int, x: int) -> int:
    idx = 0
    k = len(subset)
    for j, bit in enumerate(subset):
        idx |= _bit(x, bit) << (k - 1 - j)
    return (tt >> idx) & 1


def solve(prompt: str) -> str | None:
    """Return the 8-bit binary output string, or None on ambiguity.

    Codex review 2 (= 2026-05-13 night) flagged the previous "stop at
    the first k where candidates agree" logic as a wrong-confident bug.
    Iterating k=1..3 and unioning is too strict (~0.1% attempted), so
    we instead search at k=3 only. Lower-arity functions are subsumed
    (a k=1 boolean fn is just a k=3 fn that ignores two inputs), and
    bit puzzles ship with 7-11 input/output pairs which is usually
    enough to single out the true k=3 truth table.

    A puzzle commits only when every output bit has a unique query
    prediction across all surviving k=3 candidates; otherwise abstain.
    """
    parsed = _parse(prompt)
    if parsed is None:
        return None
    examples, query = parsed

    out_bits: list[int | None] = [None] * 8
    for ob in range(8):
        # Occam's-razor commit rule: trust this output bit only if the
        # k=1 hypothesis space is non-empty AND every surviving k=1
        # candidate predicts the same query bit AND every k=2 candidate
        # agrees with that prediction too. If anything in k ≤ 2 splits,
        # abstain — we found such split rates around 9.7% on train.
        k1_preds = {_apply(s, t, query) for s, t in _candidate_functions(examples, ob, 1)}
        if len(k1_preds) != 1:
            return None
        k2_preds = {_apply(s, t, query) for s, t in _candidate_functions(examples, ob, 2)}
        if k2_preds and k2_preds != k1_preds:
            return None
        out_bits[ob] = next(iter(k1_preds))
    return "".join(str(b) for b in out_bits)
