"""Deterministic solver for the `cipher` puzzle type.

Prompt shape:

    In Alice's Wonderland, secret encryption rules are used on text.
    Here are some examples:
      ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
      pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
      ...
    Now, decrypt the following text: trb wzrswvog hffk

Rule: monoalphabetic substitution cipher (case-insensitive). The same
letter always maps to the same letter; word boundaries (spaces) are
preserved.

Approach:

1. Parse the example lines, splitting on " -> ".
2. Align each (cipher_word, plain_word) pair character by character.
3. Build a single global cipher_letter -> plain_letter map; abstain on
   any conflict.
4. Translate the query. If any character in the query is missing from
   the map, return `None` (LLM fallback handles it).

This is intentionally conservative — wrong abstention is preferred over
a confident wrong answer (Codex review 2026-05-13 §1.1).
"""

from __future__ import annotations

import re

_QUERY = re.compile(r"decrypt the following text:\s*(.+)", re.I)


def _parse_examples(prompt: str) -> list[tuple[str, str]]:
    """Return [(cipher_text, plain_text), ...] for each example line."""
    pairs: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        if "->" not in line:
            continue
        left, _, right = line.partition("->")
        left = left.strip().lower()
        right = right.strip().lower()
        # Skip the query line and any junk.
        if not left or not right:
            continue
        if "decrypt" in left or "encrypt" in left:
            continue
        pairs.append((left, right))
    return pairs


def _build_map(pairs: list[tuple[str, str]]) -> dict[str, str] | None:
    """Build cipher_char -> plain_char; return None on irreconcilable conflict."""
    mapping: dict[str, str] = {}
    for src, tgt in pairs:
        if len(src) != len(tgt):
            # Word-boundary mismatch — try word-level alignment.
            src_words = src.split()
            tgt_words = tgt.split()
            if len(src_words) != len(tgt_words):
                continue
            for sw, tw in zip(src_words, tgt_words, strict=True):
                if len(sw) != len(tw):
                    continue
                for sc, tc in zip(sw, tw, strict=True):
                    if not sc.isalpha() or not tc.isalpha():
                        continue
                    if sc in mapping and mapping[sc] != tc:
                        return None
                    mapping[sc] = tc
            continue
        for sc, tc in zip(src, tgt, strict=True):
            if sc == " " and tc == " ":
                continue
            if not sc.isalpha() or not tc.isalpha():
                continue
            if sc in mapping and mapping[sc] != tc:
                return None
            mapping[sc] = tc
    return mapping


def solve(prompt: str) -> str | None:
    if not prompt:
        return None
    pairs = _parse_examples(prompt)
    if len(pairs) < 2:
        return None
    mapping = _build_map(pairs)
    if mapping is None:
        return None

    m = _QUERY.search(prompt)
    if not m:
        return None
    query = m.group(1).strip().lower()
    if not query:
        return None

    out: list[str] = []
    for ch in query:
        if ch == " ":
            out.append(" ")
        elif ch.isalpha():
            if ch not in mapping:
                # Unknown letter — abstain rather than guess.
                return None
            out.append(mapping[ch])
        else:
            # Punctuation in the query is rare; preserve as-is.
            out.append(ch)
    result = "".join(out).strip()
    return result if result else None
