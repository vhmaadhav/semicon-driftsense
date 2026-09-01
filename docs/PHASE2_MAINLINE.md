# Phase 2 mainline policy

`main` represents the current Phase 2 system.

The canonical runtime contract is `register.py` plus `driftsense/`, the shipped
checkpoint `weights/driftsense.pt`, pinned `requirements.txt`, and the required
submission documentation.

Phase 1 compatibility may remain where it is cheap and tested, but new Phase 2
runtime code must not depend on the legacy `infer.py` CLI.

Experimental checkpoints, per-run logs, large scratch CSVs and local training
artifacts do not belong on the mainline. Preserve durable findings in Markdown
or curated result summaries instead.
