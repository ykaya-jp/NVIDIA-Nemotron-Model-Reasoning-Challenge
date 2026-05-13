"""Deterministic prototype solver for the `equation` (Symbol Transformation) puzzle type.

Prompt shape:

    In Alice's Wonderland, a secret set of transformation rules is applied
    to equations. Below are a few examples:
      `!*[{ = '"[`
      \\'*'> = ![@
      \\'-!` = \\\\
      `!*\\& = '@'{
    Now, determine the result for: [[-!'

Rule (= host module observation):

* Each LHS is exactly 5 characters ``A B op C D`` where the middle char
  (``inp[2]``) is an operator. RHS length / content depends on the
  operator. The hidden rule is **operator-keyed**: same puzzle may use
  multiple operators in its examples, each implementing a different
  transformation.

* The query always uses one operator that may or may not appear in the
  example list.

Approach (= docs/research/2026-05-14-equation-dsl.md):

1. Parse all (LHS, RHS) pairs and the query LHS.
2. Group examples by ``op = LHS[2]``; pick the subset matching
   ``query[2]``. Let ``n_so = |same_op|``.
3. Run a depth-≤3 program search over an enumerated template set built
   from 9 primitives (= ``pick``, ``lit``, ``idxs``, ``reverse``,
   ``concat``, ``shift_chars``, ``replace_map``, ``dup``, ``arith``).
4. Apply the precision-first 3-tier policy:

   * ``n_so == 0``: abstain.
   * ``n_so == 1``: only templates that fit **every** example (= cross-
     op universal) are eligible.
   * ``n_so == 2``: string-only templates; ``arith`` is disabled to
     avoid 2-point carry/borrow false positives.
   * ``n_so >= 3``: string + arith templates both eligible.

5. Among surviving templates, commit only when all of them agree on the
   query prediction (= Occam's-razor unanimity rule). Otherwise abstain.

The prototype is intentionally **NOT polished** — it ships with the
explicit goal of measuring train precision / coverage before deciding
whether to expand the template set (= docs/research/2026-05-14-
equation-dsl.md §5.5).

Returns ``(answer, cot_trace)`` on commit, ``None`` on abstain.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------


def _parse(prompt: str) -> tuple[list[tuple[str, str]], str] | None:
    """Return ``(examples, query_lhs)`` or ``None`` if the prompt is
    malformed. ``examples`` is a list of ``(lhs, rhs)`` strings; each
    ``lhs`` is exactly 5 characters.
    """
    if not prompt:
        return None

    examples: list[tuple[str, str]] = []
    query: str | None = None
    for raw_line in prompt.splitlines():
        line = raw_line.rstrip()
        low = line.lower()
        if "determine the result" in low:
            idx = low.find("result for:")
            if idx >= 0:
                q = line[idx + len("result for:") :].strip()
                if q.startswith("`") and q.endswith("`") and len(q) >= 2:
                    q = q[1:-1]
                query = q
            continue
        if low.startswith("in alice") or "examples" in low:
            continue
        if " = " not in line:
            continue
        lhs, _, rhs = line.partition(" = ")
        lhs = lhs.strip()
        rhs = rhs.strip()
        # Strip surrounding backticks (used by the prompt when the string
        # contains ``'`` or ``"``).
        if lhs.startswith("`") and lhs.endswith("`") and len(lhs) >= 2:
            lhs = lhs[1:-1]
        if rhs.startswith("`") and rhs.endswith("`") and len(rhs) >= 2:
            rhs = rhs[1:-1]
        if len(lhs) == 5:
            examples.append((lhs, rhs))
    if not examples or query is None or len(query) != 5:
        return None
    return examples, query


# ---------------------------------------------------------------------
# DSL evaluation
# ---------------------------------------------------------------------


def _eval(prog: tuple, inp: str) -> str:
    """Evaluate a depth-≤3 program against a 5-char input. Returns the
    string output or the sentinel ``"\\x00ERR\\x00"`` for arithmetic
    programs that fail to parse digits (= treat as no-match)."""
    tag = prog[0]
    if tag == "pick":
        return inp[prog[1]]
    if tag == "lit":
        return prog[1]
    if tag == "idxs":
        return "".join(inp[i] for i in prog[1])
    if tag == "rev":
        return _eval(prog[1], inp)[::-1]
    if tag == "cat":
        return _eval(prog[1], inp) + _eval(prog[2], inp)
    if tag == "dup":
        return _eval(prog[1], inp) * 2
    if tag == "shift":
        s = _eval(prog[1], inp)
        k = prog[2]
        out: list[str] = []
        for c in s:
            if c.isdigit():
                out.append(chr((ord(c) - ord("0") + k) % 10 + ord("0")))
            elif c.isalpha():
                base = ord("A") if c.isupper() else ord("a")
                out.append(chr((ord(c) - base + k) % 26 + base))
            else:
                out.append(c)
        return "".join(out)
    if tag == "repl":
        s = _eval(prog[1], inp)
        mapping: dict[str, str] = prog[2]
        return "".join(mapping.get(c, c) for c in s)
    if tag == "arith":
        op = prog[1]
        try:
            ab = int(inp[0] + inp[1])
            cd = int(inp[3] + inp[4])
        except ValueError:
            return "\x00ERR\x00"
        if op == "+":
            v = ab + cd
        elif op == "-":
            v = ab - cd
        elif op == "rsub":
            v = cd - ab
        elif op == "abs_sub":
            v = abs(ab - cd)
        elif op == "*":
            v = ab * cd
        else:
            return "\x00ERR\x00"
        return str(v)
    raise ValueError(f"unknown DSL tag: {tag}")


# ---------------------------------------------------------------------
# Template enumeration (= depth ≤ 3)
# ---------------------------------------------------------------------

# Concrete index permutations seen in the training data. Includes:
#   * 4-position subsets that drop the operator (`(0,1,3,4)` etc.)
#   * 4-position swaps (= move the right half to the front, `(3,4,0,1)`)
#   * Half-reverses (`(1,0,3,4)`, `(0,1,4,3)`, ...)
#   * Single-position picks for "answer = one character" puzzles
#   * Operator-only `(2,)` for puzzles that return just the op
_INDEX_SETS_CORE: tuple[tuple[int, ...], ...] = (
    (0, 1, 3, 4),  # drop op
    (3, 4, 0, 1),  # swap halves
    (0, 1),  # left half
    (3, 4),  # right half
    (0, 1, 2, 3, 4),  # identity
    (4, 3, 2, 1, 0),  # full reverse
    (1, 0, 3, 4),  # left reverse, right intact
    (0, 1, 4, 3),  # left intact, right reverse
    (1, 0, 4, 3),  # both halves reversed
    (3, 4, 1, 0),  # swap + reverse left
    (4, 3, 0, 1),  # swap + reverse right
    (4, 3, 1, 0),  # ABCD reversed
    (0, 3),
    (1, 4),
    (1, 3),
    (0, 4),  # cross picks
    (0,),
    (1,),
    (3,),
    (4,),
    (2,),  # singletons
)


def _build_templates() -> tuple[list[tuple], list[tuple]]:
    """Return ``(string_templates, arith_templates)``.

    String templates use only string-manipulation primitives (= safe at
    n_so >= 2). Arithmetic templates interpret the two 2-digit halves as
    integers (= only enabled at n_so >= 3 to limit false positives).
    """
    string_templates: list[tuple] = []
    for s in _INDEX_SETS_CORE:
        string_templates.append(("idxs", s))
    # '-' prefix variants for puzzles that return a signed result.
    for s in _INDEX_SETS_CORE:
        string_templates.append(("cat", ("lit", "-"), ("idxs", s)))
    # dup of common halves (= L+L or R+R outputs are surprisingly common).
    for s in [(0, 1), (3, 4), (0, 3), (1, 4), (2,)]:
        string_templates.append(("dup", ("idxs", s)))

    arith_templates: list[tuple] = [
        ("arith", "+"),
        ("arith", "-"),
        ("arith", "rsub"),
        ("arith", "abs_sub"),
        ("arith", "*"),
    ]
    return string_templates, arith_templates


_STR_TEMPLATES, _ARITH_TEMPLATES = _build_templates()


# ---------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------


def _fits(prog: tuple, examples: list[tuple[str, str]]) -> bool:
    """Return True iff ``prog`` reproduces every (lhs, rhs) pair."""
    for lhs, rhs in examples:
        try:
            if _eval(prog, lhs) != rhs:
                return False
        except (ValueError, IndexError, ZeroDivisionError):
            return False
    return True


def _describe(prog: tuple) -> str:
    """Human-readable rendering of a program (for the CoT trace)."""
    tag = prog[0]
    if tag == "pick":
        return f"pick(inp[{prog[1]}])"
    if tag == "lit":
        return f"lit({prog[1]!r})"
    if tag == "idxs":
        return f"idxs{prog[1]}"
    if tag == "rev":
        return f"reverse({_describe(prog[1])})"
    if tag == "cat":
        return f"concat({_describe(prog[1])}, {_describe(prog[2])})"
    if tag == "dup":
        return f"dup({_describe(prog[1])})"
    if tag == "shift":
        return f"shift_chars({_describe(prog[1])}, k={prog[2]})"
    if tag == "repl":
        return f"replace_map({_describe(prog[1])}, {prog[2]!r})"
    if tag == "arith":
        return f"arith(int(inp[0:2]) {prog[1]} int(inp[3:5]))"
    return repr(prog)


def solve(prompt: str) -> tuple[str, str] | None:
    """Return ``(answer, cot_trace)`` on commit, else ``None``.

    The CoT trace is a Searchformer-style hypothesis -> verification ->
    decision narrative (= docs/research/2026-05-13-mathematical-
    foundations.md §3.4 inherits from solver_bit.py / solver_cipher.py).
    """
    parsed = _parse(prompt)
    if parsed is None:
        return None
    examples, query = parsed
    if len(query) != 5:
        return None
    q_op = query[2]
    same_op = [(lhs, rhs) for lhs, rhs in examples if len(lhs) == 5 and lhs[2] == q_op]
    n_so = len(same_op)
    # ---- precision-first 3-tier policy ----
    if n_so == 0:
        return None
    if n_so == 1:
        # universal fit required (= every example, not just same-op)
        templates = _STR_TEMPLATES
        relevant = examples
    elif n_so == 2:
        # string-only on same-op; arith disabled
        templates = _STR_TEMPLATES
        relevant = same_op
    else:  # n_so >= 3
        templates = _STR_TEMPLATES + _ARITH_TEMPLATES
        relevant = same_op

    fits = [prog for prog in templates if _fits(prog, relevant)]
    if not fits:
        return None

    # Unanimity commit rule
    preds: set[str] = set()
    keepers: list[tuple] = []
    for prog in fits:
        try:
            v = _eval(prog, query)
        except (ValueError, IndexError):
            continue
        if v == "\x00ERR\x00":
            continue
        preds.add(v)
        keepers.append(prog)
    if len(preds) != 1 or not keepers:
        return None
    answer = next(iter(preds))

    # Build Searchformer-style trace
    head = (
        "I observe that the equation puzzle uses a 5-character LHS "
        f"`A B op C D` where the middle char is the operator. The query is "
        f"{query!r} with operator {q_op!r}."
    )
    if n_so == 1:
        tier = (
            "Only 1 same-op example is available, so I require any candidate "
            "rule to also fit every other example (cross-op universal)."
        )
    elif n_so == 2:
        tier = (
            "I have 2 same-op examples; I restrict the search to "
            "string-manipulation templates (= arithmetic is disabled to "
            "avoid carry / borrow false positives at this count)."
        )
    else:
        tier = f"I have {n_so} same-op examples; I search over string and " "arithmetic templates."
    chosen = _describe(keepers[0])
    if len(keepers) > 1:
        chosen_note = (
            f"{len(keepers)} templates fit (e.g. `{chosen}`); they all "
            f"predict the same query output {answer!r}, so the answer is "
            f"unambiguous under Occam's-razor unanimity."
        )
    else:
        chosen_note = (
            f"Exactly one template fits: `{chosen}`. Applying it to the "
            f"query yields {answer!r}."
        )
    examples_block = "\n".join(f"  {lhs!r} -> {rhs!r}" for lhs, rhs in relevant[:4])
    cot = (
        f"{head}\n\n{tier}\n\nSame-op examples I must explain:\n"
        f"{examples_block}\n\n{chosen_note}"
    )
    return answer, cot


# ---------------------------------------------------------------------
# Backwards-compatible plain-string entry point (= matches
# solver_bit.solve / solver_cipher.solve signature so the existing
# hybrid router can call it interchangeably).
# ---------------------------------------------------------------------


def solve_str(prompt: str) -> str | None:
    """Return just the answer string, or None on abstain."""
    result = solve(prompt)
    if result is None:
        return None
    return result[0]
