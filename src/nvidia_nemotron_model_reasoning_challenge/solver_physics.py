"""Deterministic solver for the `physics` puzzle type.

Prompt shape:

    In Alice's Wonderland, gravity has a secret value. From observations:
      t=1s → distance=4.9m
      t=2s → distance=19.6m
      t=3s → distance=44.1m

    Using the formula: distance = 0.5 * g * t**2
    Find the distance when t=4s.

Approach:

1. Regex-extract all (t, d) example pairs.
2. Compute g_i = 2*d_i / t_i**2 for each example.
3. Take the mean (g_i should agree exactly if the puzzle is well-formed).
4. Apply d = 0.5 * g_mean * t_query**2 and round to 2 decimal places —
   the train answer column uses 2-decimal rounding consistently.

The solver returns `None` if it cannot parse pairs or the query.
"""

from __future__ import annotations

import re
import statistics

_PAIR = re.compile(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m", re.I | re.S)
_QUERY = re.compile(r"(?:find|determine|compute|when)\s*[^t]*t\s*=\s*([\d.]+)\s*s", re.I)


def solve(prompt: str) -> str | None:
    if not prompt:
        return None
    pairs = _PAIR.findall(prompt)
    if len(pairs) < 2:
        return None
    gs: list[float] = []
    for t_str, d_str in pairs:
        try:
            t = float(t_str)
            d = float(d_str)
        except ValueError:
            continue
        if t == 0.0:
            continue
        gs.append(2.0 * d / (t * t))
    if not gs:
        return None
    g_mean = statistics.fmean(gs)

    # The query line is "Find the distance when t=Ns" — match the last
    # t=... that is NOT in the example block.
    m = _QUERY.search(prompt.split("\n")[-1]) or _QUERY.search(prompt)
    if not m:
        # Fall back: last t= occurrence
        all_t = re.findall(r"t\s*=\s*([\d.]+)\s*s", prompt, re.I)
        if not all_t:
            return None
        try:
            t_q = float(all_t[-1])
        except ValueError:
            return None
    else:
        try:
            t_q = float(m.group(1))
        except ValueError:
            return None
    d_q = 0.5 * g_mean * t_q * t_q
    # Return as a string with the same format as `str(round(x, 2))`,
    # matching mohankrishnathalla's public solver and the format used in
    # train.answer (e.g. "140.44", "45.0", "19.0"). The official metric
    # uses relative-tolerance float comparison so trailing-zero drift is
    # harmless, but `{:g}` strips zeros aggressively which can hurt the
    # LLM fallback's pattern recognition.
    return str(round(d_q, 2))
