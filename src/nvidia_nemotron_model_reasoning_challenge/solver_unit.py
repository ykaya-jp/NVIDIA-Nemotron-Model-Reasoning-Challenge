"""Deterministic solver for the `unit` puzzle type.

Prompt shape:

    In Alice's Wonderland, a secret unit conversion is applied to
    measurements. For example:
      10.08 m becomes 6.69
      17.83 m becomes 11.83
      ...
    Now, convert the following measurement: 25.09 m

Rule: a fixed linear ratio per puzzle (output = ratio * input).

Approach:

1. Regex-extract every "X m becomes Y" example pair.
2. Compute ratio = mean(Y / X) over examples.
3. Apply to the query input, round to 2 decimals.

Returns `None` on parse failure.
"""

from __future__ import annotations

import re
import statistics

_PAIR = re.compile(r"([\d.]+)\s*m\s*becomes\s*([\d.]+)", re.I)
_QUERY = re.compile(
    r"(?:convert (?:the following measurement|the measurement|to wonderland units)?:?\s*)([\d.]+)\s*m",
    re.I,
)


def solve(prompt: str) -> str | None:
    if not prompt:
        return None
    pairs = _PAIR.findall(prompt)
    if len(pairs) < 2:
        return None
    ratios: list[float] = []
    for x_str, y_str in pairs:
        try:
            x = float(x_str)
            y = float(y_str)
        except ValueError:
            continue
        if x == 0.0:
            continue
        ratios.append(y / x)
    if not ratios:
        return None
    ratio = statistics.fmean(ratios)

    # Query is the LAST "<num> m" occurrence that is NOT followed by "becomes"
    # (i.e. it is not a demonstration pair).
    all_m = re.findall(r"([\d.]+)\s*m(?!\s*becomes)", prompt, re.I)
    if not all_m:
        m = _QUERY.search(prompt)
        if not m:
            return None
        try:
            x_q = float(m.group(1))
        except ValueError:
            return None
    else:
        try:
            x_q = float(all_m[-1])
        except ValueError:
            return None

    y_q = ratio * x_q
    return str(round(y_q, 2))
