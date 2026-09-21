# Contributing

Drift-Sense is a measurement-driven repository. Most of the rules below exist
because a specific regression got through once; the parenthetical is the
reason, not decoration.

## Setup

Python **3.11** — it is what the reference machine runs and what CI pins.

```bash
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` is a frozen `pip freeze` from a CPU-only PyTorch
environment. Do not hand-edit a version in it without regenerating.

## Before you open a PR

All four must pass. CI runs all four.

```bash
python -m pytest -q
```

```bash
python scripts/check_register_contract.py
```

```bash
python scripts/release_gate.py
```

`pytest` covers both suites — `tests/` and the vendored generator's own
contract tests under `generator/tests/`. Do not narrow `testpaths`; excluding
`generator/tests` once let wrong-label generator regressions escape routine
runs (audit H-13).

`check_register_contract.py` proves `register.py` still satisfies the output
contract end to end. `release_gate.py` proves the submission archive still
builds, builds reproducibly, and contains nothing denied. pytest alone is not
enough: PR #20 added `driftsense/vst.py` and PR #26 authored the ship manifest
that has to name it. Each was green alone and red together.

## The rules that actually matter here

**One definition of the shipped configuration.** Every threshold, band flag,
verification mode and label convention lives in `driftsense/config.py`. Never
write a local literal for one. `register.py`, `scripts/eval_ext.py` and the
parity tests are pinned against that module precisely so the submission path
and the evaluator cannot drift apart.

**Scorer semantics are correctness-critical.** A change to `register.py` or
`scripts/eval_ext.py` that moves threshold, `band`, or the zero-credit rule for
declined-but-present pairs needs its own justification in the PR body. Every
`pair_id` must appear exactly once in the output.

**Declining beats disappearing.** The graded path fails soft: a pair that
errors still emits a row, because a missing row scores zero. Do not "fix" an
`except Exception` on that path by letting it propagate.

**`weights_only=True` on every `torch.load`.** No exceptions on any inference
path.

**No network access** from `register.py` or `phase3.py`. The reference machine
has none.

## Performance claims

A number in the README or in a PR body needs the measurement behind it:

* say which dataset and how many pairs,
* prefer a **paired** A/B on the same pairs over two independent runs,
* give the spread, not just the mean — at the sample sizes here a difference
  under about a point is not a result,
* name the artifact the numbers came from.

Do not fold an unmeasured change into a PR that claims a gain.

## Tests

Prefer explicit hand-derived literals over values generated from the
implementation under test — a test that recomputes the implementation's own
answer proves nothing.

Numerics: fp32 loss over bf16 autocast is deliberate (focal-term underflow).
Leave it alone unless you are fixing a genuine numerical hazard.

## Scripts vs library

`scripts/` are working research tools and are held to a lower bar — no
production hardening demanded there. `driftsense/`, `register.py` and
`phase3.py` are the product and are held to the rules above.

## Commits and PRs

Write the *why* in the commit body. The git history here is used as evidence;
`git log` is how a reviewer reconstructs why a calibration is what it is.

PRs are reviewed by CodeRabbit automatically (`.coderabbit.yaml`) and by a
human before merge.
