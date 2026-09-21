## What this changes

<!-- One paragraph. What moved, and why. -->

## Evidence

<!--
Delete this section only if the change cannot affect a measured number
(docs, tooling, CI). Otherwise fill it in.

- Dataset and pair count:
- Paired A/B on the same pairs?  yes / no
- Before -> after, with spread:
- Artifact the numbers came from:
-->

## Checklist

- [ ] `python -m pytest -q` passes (both `tests/` and `generator/tests/`)
- [ ] `python scripts/check_register_contract.py` passes
- [ ] `python scripts/release_gate.py` passes
- [ ] No new literal for a value that belongs in `driftsense/config.py`
- [ ] Every new `torch.load` passes `weights_only=True`
- [ ] No network access added to `register.py` / `phase3.py`

## Scorer semantics

- [ ] This PR does **not** change threshold, `band`, or the zero-credit rule
      for declined-but-present pairs

<!-- If it does, justify it here and say what was re-measured. -->
