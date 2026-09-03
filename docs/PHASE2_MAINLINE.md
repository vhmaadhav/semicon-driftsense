# Phase 2 mainline policy

`phase2` is the repository's default branch and represents the current Phase 2
system. `main` was renamed to `phase1` and frozen as a historical snapshot of
the pre-Phase-2 baseline; it receives no further changes.

The canonical runtime contract is `register.py` plus `driftsense/`, the shipped
checkpoint `weights/driftsense.pt`, pinned `requirements.txt`, and the required
submission documentation.

Phase 1 compatibility may remain where it is cheap and tested, but new Phase 2
runtime code must not depend on the legacy `infer.py` CLI.

Experimental checkpoints, per-run logs, large scratch CSVs and local training
artifacts do not belong on the mainline. Preserve durable findings in Markdown
or curated result summaries instead.
