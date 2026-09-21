# Security Policy

## Supported versions

This project ships from `main`. Only the current `main` receives fixes; there
are no maintained release branches.

## Reporting a vulnerability

Report privately through GitHub's
[Report a vulnerability](https://github.com/vhmaadhav/semicon-driftsense/security/advisories/new)
form. Please do **not** open a public issue for a security problem.

Include what you have: affected file or entry point, how to reproduce, and what
an attacker gets. A first response should arrive within 7 days.

## Threat model

Drift-Sense is a CPU-only batch inference tool. It reads a `pairs.csv`, reads
the image or GDSII files that CSV points at, and writes a `predictions.csv`.
It opens no sockets and makes no network calls — see
`scripts/check_register_contract.py`, which enforces that.

Untrusted input therefore means **the files named in `pairs.csv`**:

* **Image files** decoded via OpenCV and Pillow. A malformed or hostile PNG or
  TIFF reaches those decoders. Keep `opencv-python-headless` and `pillow`
  patched; that is the main reason this repo pins them.
* **GDSII layout files** parsed via `gdstk` on the Phase 3 path
  (`phase3.py`, `driftsense/gds.py`). GDSII is a binary format and `gdstk` is a
  C++ extension; treat unreviewed `.gds` input as untrusted.
* **CSV paths.** `register.py` and `phase3.py` open the paths given to them.
  Point them only at manifests you control.

### Model checkpoints

Every `torch.load` in this repository passes `weights_only=True`, which blocks
the pickle-execution class of attack. `tests/test_checkpoint_safety.py` pins
that. If you add a new load path, it must keep `weights_only=True` — a
checkpoint is otherwise arbitrary code execution.

Do not load `.pt` files from sources you do not trust, even with
`weights_only=True`.

### Not in scope

* Denial of service from deliberately huge inputs. The graded path has a 20 s
  per-pair timeout and fails soft by design (a declined row beats a missing
  row); it is not hardened against resource exhaustion beyond that.
* The vendored `generator/` tree, which is third-party code (see NOTICE).
