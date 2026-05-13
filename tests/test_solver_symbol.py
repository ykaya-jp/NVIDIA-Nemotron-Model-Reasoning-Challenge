"""Unit tests for the prototype Symbol / Equation Transformation solver.

The 10 representative puzzles below come straight from
``data/processed/train_sft_v1.jsonl`` (filter: ``source == "gold_fallback"``
and prompt classified as Equation Transformation). Each puzzle's gold
answer is the ground truth that ``data/raw/train.csv`` carries.

We cover three signal classes:

* 7 commit puzzles where the prototype produces the correct answer.
* 3 abstain puzzles where the solver must return ``None`` rather than
  guess (= precision > coverage, parent CLAUDE.md §11).
"""

from __future__ import annotations

import pytest

from nvidia_nemotron_model_reasoning_challenge import solver_symbol

# ---------------------------------------------------------------------
# Commit cases — must produce ``(answer, cot_trace)`` matching the gold.
# ---------------------------------------------------------------------

# Each (id, prompt, gold) triple is taken verbatim from
# data/processed/train_sft_v1.jsonl. The id is purely for debugging.
COMMIT_CASES = [
    (
        "0133bcec",
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        '%|*"| = %|"|\n'
        "\\(*[^ = \\([^\n"
        "(%+[@ = (%[@\n"
        "|[*([ = |[([\n"
        "[^-[( = -^\n"
        "Now, determine the result for: \\(*[#",
        "\\([#",
    ),
    (
        "04322d27",
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "96$54 = 5184\n"
        "50$41 = 2050\n"
        "51$95 = 4845\n"
        "89$47 = 4183\n"
        "Now, determine the result for: 59$49",
        "2891",
    ),
    (
        "0cd170a0",
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "22:12 = 10\n"
        "23:33 = :10\n"
        "67^30 = 6730\n"
        "71^79 = 7179\n"
        "Now, determine the result for: 98^14",
        "9814",
    ),
    (
        "118889da",
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "47{42 = 5\n"
        "30[47 = 3047\n"
        "46{33 = 13\n"
        "79{56 = 23\n"
        "Now, determine the result for: 66{56",
        "10",
    ),
    (
        "1c48f9aa",
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "22{52 = 2252\n"
        "74*60 = 14\n"
        "31*65 = 34\n"
        "39{12 = 3912\n"
        "Now, determine the result for: 86{10",
        "8610",
    ),
    (
        "1f445c5e",
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "58?58 = 3364\n"
        "95}49 = }46\n"
        "81?13 = 1053\n"
        "53?11 = 583\n"
        "Now, determine the result for: 34?46",
        "1564",
    ),
    (
        "21b90d9f",
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "]{-]{ = :\n"
        "{'*/{ = {'/{\n"
        "{?-?' = ))\n"
        "/&*)) = /&))\n"
        "{/*&/ = {/&/\n"
        "Now, determine the result for: ){*?{",
        "){?{",
    ),
]


# ---------------------------------------------------------------------
# Abstain cases — must return None rather than guess.
# ---------------------------------------------------------------------

# These puzzles either (a) have the query operator absent from
# examples or (b) require arithmetic-with-2-only-examples (= n_so == 2
# but rule needs more constraints than 2 points provide), which the
# precision-first policy refuses to commit to.
ABSTAIN_CASES = [
    (
        "00457d26",
        # query op '-' appears 1x; cross-op universal fit not found.
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "`!*[{ = '\"[`\n"
        "\\'*'> = ![@\n"
        "\\'-!` = \\\\\n"
        "`!*\\& = '@'{\n"
        "Now, determine the result for: [[-!'",
    ),
    (
        "00c032a8",
        # query op '!' appears 1x; cross-op universal fit not found.
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "}`]?( = ())\n"
        "}#<)\\ = #?\n"
        "?(!&& = #@@#\n"
        "(?!@` = )#))\n"
        "Now, determine the result for: ))!\\)",
    ),
    (
        "00d8b3db",
        # query op '/' appears 2x but no string template fits; arith
        # is disabled at n_so == 2.
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "34/44 = 1\n"
        "41/32 = 9\n"
        "34|25 = 69\n"
        "87\\64 = 8853\n"
        "Now, determine the result for: 69/52",
    ),
]


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.parametrize("puzzle_id,prompt,gold", COMMIT_CASES)
def test_commit_cases(puzzle_id: str, prompt: str, gold: str) -> None:
    result = solver_symbol.solve(prompt)
    assert result is not None, f"{puzzle_id}: solver abstained but expected gold={gold!r}"
    answer, cot = result
    assert answer == gold, f"{puzzle_id}: got {answer!r}, expected {gold!r}"
    assert isinstance(cot, str) and cot, f"{puzzle_id}: empty CoT trace"
    # The trace should mention the query and the operator we keyed on.
    assert "operator" in cot.lower()


@pytest.mark.parametrize("puzzle_id,prompt", ABSTAIN_CASES)
def test_abstain_cases(puzzle_id: str, prompt: str) -> None:
    result = solver_symbol.solve(prompt)
    assert result is None, f"{puzzle_id}: solver returned {result!r} but should abstain"


def test_solve_str_matches_solve() -> None:
    """``solve_str`` must return just the answer of ``solve()``."""
    prompt = COMMIT_CASES[0][1]
    gold = COMMIT_CASES[0][2]
    assert solver_symbol.solve_str(prompt) == gold


def test_solve_str_returns_none_on_abstain() -> None:
    prompt = ABSTAIN_CASES[0][1]
    assert solver_symbol.solve_str(prompt) is None


def test_parse_handles_empty_prompt() -> None:
    assert solver_symbol.solve("") is None
    assert solver_symbol.solve("no examples here") is None


def test_parse_rejects_wrong_length_query() -> None:
    # Query with 4 chars instead of 5: must abstain
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is "
        "applied to equations. Below are a few examples:\n"
        "11+22 = 33\n"
        "Now, determine the result for: 1+2"
    )
    assert solver_symbol.solve(prompt) is None
