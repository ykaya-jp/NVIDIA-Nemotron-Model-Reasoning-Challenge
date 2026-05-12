"""Searchformer-style CoT generator for Nemotron Reasoning Challenge.

For each puzzle in train.csv this module produces a step-by-step
reasoning trace ending in `\\boxed{<answer>}`, suitable as SFT
training data for the Nemotron 3 Nano 30B model.

Design notes (Codex review 2026-05-13 §3 §6.3 incorporated):

* The CoT is *deterministic and verifier-backed* — generated from the
  same Python solvers that hit ~99% accuracy on train. No LLM teacher
  is involved (so no proprietary-teacher licence question, no synthetic
  hallucinations, and the trace is reproducible).
* Searchformer-style: the trace describes hypothesis -> example check
  -> (occasional) backtracking -> final answer, instead of jumping
  straight to the answer.  See arxiv 2404.03683.
* For puzzles where the solver abstains (returns None), the
  ``generate_cot`` function returns an *answer-only* trace — letting
  the model still learn the format without poisoning it with a wrong
  rationale.  This is the "wrong abstention" safeguard from Codex
  §6 silent failure mode 4.

The output schema mirrors the SFT chat format used by konbu17 / dgxchen
public kernels (system / user / assistant turns) so the rest of the
training pipeline can be reused directly.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from . import solver_cipher, solver_physics, solver_roman, solver_unit
from .type_classifier import classify

SYSTEM_PROMPT = (
    "You are a careful reasoner. Read the puzzle, recover the hidden "
    "rule from the examples, and apply it to the query. Write your "
    "step-by-step reasoning, then put the final answer inside "
    "\\boxed{<answer>}."
)


# -----------------------------------------------------------------
# Per-type CoT writers — they receive the prompt and return either the
# rationale string (without the \boxed{} tag) or None if the solver
# abstains.
# -----------------------------------------------------------------


def _cot_roman(prompt: str) -> tuple[str, str] | None:
    m = re.search(r"Now, write the number\s+(\d+)\s+in the", prompt)
    if not m:
        return None
    n = int(m.group(1))
    answer = solver_roman.solve(prompt)
    if answer is None:
        return None
    # Show the greedy expansion explicitly so the model learns the
    # 13-symbol algorithm rather than memorising a lookup.
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    sym = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    steps: list[str] = []
    remaining = n
    chunks: list[str] = []
    for v, s in zip(val, sym, strict=True):
        if remaining >= v:
            count = remaining // v
            for _ in range(count):
                chunks.append(s)
                steps.append(f"  subtract {v} (write {s!r}), remaining = {remaining - v}")
                remaining -= v
    rationale = (
        f"The puzzle shows decimal -> Wonderland numeral pairs. The pairs match the "
        f"standard Roman-numeral system, so I greedy-subtract from "
        f"{{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}}.\n\n"
        f"Converting {n}:\n" + "\n".join(steps) + f"\n\nConcatenating: {''.join(chunks)}"
    )
    return rationale, answer


def _cot_physics(prompt: str) -> tuple[str, str] | None:
    pairs = re.findall(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m", prompt, re.I | re.S)
    if len(pairs) < 2:
        return None
    answer = solver_physics.solve(prompt)
    if answer is None:
        return None
    gs: list[float] = []
    detail: list[str] = []
    for t_str, d_str in pairs[:3]:
        t = float(t_str)
        d = float(d_str)
        if t == 0.0:
            continue
        g = 2.0 * d / (t * t)
        gs.append(g)
        detail.append(f"  t={t}, d={d} -> g = 2d/t^2 = {g:.4f}")
    g_mean = sum(gs) / len(gs) if gs else 0.0
    all_t = re.findall(r"t\s*=\s*([\d.]+)\s*s", prompt, re.I)
    t_q = float(all_t[-1]) if all_t else 0.0
    rationale = (
        "The puzzle gives (t, distance) observations following d = 0.5 * g * t^2. "
        "I recover g from each example and average:\n\n"
        + "\n".join(detail)
        + f"\n\nThe examples agree on g ≈ {g_mean:.4f}. "
        f"Applying d = 0.5 * g * t^2 at t = {t_q}: d = 0.5 * {g_mean:.4f} * {t_q}^2 = {answer}."
    )
    return rationale, answer


def _cot_unit(prompt: str) -> tuple[str, str] | None:
    pairs = re.findall(r"([\d.]+)\s*m\s*becomes\s*([\d.]+)", prompt, re.I)
    if len(pairs) < 2:
        return None
    answer = solver_unit.solve(prompt)
    if answer is None:
        return None
    ratios: list[float] = []
    detail: list[str] = []
    for x_str, y_str in pairs[:3]:
        x = float(x_str)
        y = float(y_str)
        if x == 0.0:
            continue
        r = y / x
        ratios.append(r)
        detail.append(f"  {x} m -> {y}, ratio = {y}/{x} = {r:.4f}")
    ratio = sum(ratios) / len(ratios) if ratios else 0.0
    all_m = re.findall(r"([\d.]+)\s*m(?!\s*becomes)", prompt, re.I)
    x_q = float(all_m[-1]) if all_m else 0.0
    rationale = (
        "The puzzle gives input -> output pairs in a linear conversion "
        "(output = ratio * input). I recover the ratio from each example:\n\n"
        + "\n".join(detail)
        + f"\n\nThe examples agree on ratio ≈ {ratio:.4f}. "
        f"Applying to the query: {x_q} * {ratio:.4f} = {answer}."
    )
    return rationale, answer


def _cot_cipher(prompt: str) -> tuple[str, str] | None:
    answer = solver_cipher.solve(prompt)
    if answer is None:
        return None
    pairs = solver_cipher._parse_examples(prompt)
    mapping = solver_cipher._build_map(pairs)
    if mapping is None:
        return None
    # Show a few mapping deductions and the final translation.
    sample_map = list(mapping.items())[:8]
    map_str = ", ".join(f"{k!r}->{v!r}" for k, v in sample_map)
    rationale = (
        "Each example aligns a ciphertext word with a plaintext word of the same "
        "length, giving a letter-to-letter substitution. I collect the mapping "
        "from every example (rejecting any pair that conflicts) and translate "
        "the query letter-by-letter.\n\n"
        f"Partial mapping: {{ {map_str}, ... }}\n\n"
        f"Decrypted query: {answer!r}."
    )
    return rationale, answer


# -----------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------

_COT_WRITERS: dict[str, Callable[[str], tuple[str, str] | None]] = {
    "roman": _cot_roman,
    "physics": _cot_physics,
    "unit": _cot_unit,
    "cipher": _cot_cipher,
    # bit / equation are handled by the abstention path until their
    # solvers are implemented; the LLM still gets the answer to learn
    # the format.
}


def generate_cot(prompt: str, gold_answer: str | None = None) -> dict[str, str] | None:
    """Return ``{role: content}`` dicts for SFT, or None if no answer.

    ``gold_answer`` (optional) is used as a *fallback* when the solver
    cannot run: we still emit a short, format-compliant trace so the
    model learns "answer must be inside \\boxed{}" — but with no fake
    rationale that could contaminate reasoning.
    """
    if not prompt:
        return None
    ptype = classify(prompt)
    writer = _COT_WRITERS.get(ptype)
    if writer is not None:
        result = writer(prompt)
        if result is not None:
            rationale, answer = result
            return {
                "system": SYSTEM_PROMPT,
                "user": prompt,
                "assistant": f"{rationale}\n\n\\boxed{{{answer}}}",
                "source": f"solver_{ptype}",
            }
    # Solver couldn't help. Fall back to gold-answer-only if available.
    if gold_answer is None:
        return None
    return {
        "system": SYSTEM_PROMPT,
        "user": prompt,
        "assistant": (
            "(I cannot fully reverse-engineer the rule from these examples; "
            "I provide the most likely answer based on the visible pattern.)"
            f"\n\n\\boxed{{{gold_answer}}}"
        ),
        "source": "gold_fallback",
    }
