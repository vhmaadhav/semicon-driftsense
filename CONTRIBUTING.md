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

All five must pass. CI runs all five.

```bash
ruff check .
```

```bash
mypy
```

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

`ruff check` enforces `ruff.toml`, which documents why each rule is on or
off. The short version: rules that find defects are enabled, rules with
opinions about dense numeric code are not, and `scripts/`, `tests/` and
`experiments/` get pyflakes only -- the same policy `.coderabbit.yaml`
already states for `scripts/`. Install the local hook with `pip install
pre-commit && pre-commit install` to get the answer before the commit exists
rather than ten minutes after pushing. A suppression needs a reason next to
it, not just a code.

Run `mypy` in an environment built from `requirements.txt`. Its result
depends on the installed stub versions, not only on the code: under cv2
4.x the in-place add in `driftsense/cad_anchor.py` needs a
`# type: ignore`, and under the pinned cv2 5.x it does not, so the same
source disagrees with itself across environments. The pinned set is the
contract and CI is the authority -- a stray `unused-ignore` locally
usually means your cv2 or numpy is not the pinned one.

`mypy` takes no arguments on purpose: scope, Python version and settings all
come from `[tool.mypy]` in `pyproject.toml`, so your run and CI's are the same
check. It covers `driftsense/` only, by the same policy as ruff. Eight
`# type: ignore` comments exist there, every one for a numpy or cv2 stub that
is stricter than the library actually is, and every one confirmed by running
the code rather than by reading the stub. `warn_unused_ignores` is on, so when
an upstream release fixes a stub the stale ignore becomes an error instead of
sitting there pretending to be load-bearing. If you add one, say why beside
it.

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
