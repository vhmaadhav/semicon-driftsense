# Phase 1 legacy interface

Phase 1 is preserved for history and regression compatibility, but it is no
longer the canonical Drift-Sense product surface.

The historical command is:

```bash
python infer.py --reference reference.png --search search.png
```

It assumes the original fixed-pose single-pair contract and prints `x,y`.

For current work use the Phase 2 batch interface instead:

```bash
python register.py --input pairs.csv --output predictions.csv
```

The reusable checkpoint loading, image loading and fallback implementation now
live in `driftsense/runtime.py`; Phase 2 does not depend on the legacy CLI.

Git history before the Phase 2 promotion remains the authoritative record for
old Phase 1 experiments and documentation that were intentionally removed from
the main README.
