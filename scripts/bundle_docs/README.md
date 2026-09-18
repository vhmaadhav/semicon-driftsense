# Drift-Sense — inference

Navigation-Error Recovery for semiconductor metrology: find a known reference
inside a wide SEM search frame, report its pose, and say when it is not there.

This is the **inference delivery**. It contains the two graded entry points,
the weights, and nothing else — no training code, no dataset generator, no
evaluation harness, no development tooling.

```
.
├── phase_2/          reference is an SEM image
│   ├── register.py       <- entry point
│   ├── requirements.txt
│   ├── weights/
│   ├── sample/           3 real pairs + ground truth
│   └── README.md         full instructions
└── phase_3/          reference is a GDSII CAD design
    ├── phase3.py         <- entry point
    ├── requirements.txt
    ├── weights/
    ├── sample/           2 real pairs + ground truth
    └── README.md         full instructions
```

Each folder is **independent and portable**. It reads nothing outside itself,
resolves its weights relative to its own entry point rather than to your shell,
and can be copied anywhere and run from any working directory.

---

## Run it in three commands

**Phase 2** — SEM reference:

```bash
cd phase_2
python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python register.py --input sample/pairs.csv --output sample/predictions.csv
```

**Phase 3** — CAD reference:

```bash
cd phase_3
python3.11 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python phase3.py --input sample/pairs.csv --output sample/predictions.csv
```

Both bundled samples include a site with **no true match**, so you can confirm
the rejection path works as well as the localisation path.

---

## The contract, both phases

Same command shape, same output, different reference:

```bash
python <entry point> --input pairs.csv --output predictions.csv
```

`predictions.csv` — one row per input pair, in input order:

```
pair_id, x, y, theta, scale, found, score
```

* `x, y` — match centre in search-image pixels
* `theta` — degrees, CCW positive, about the match centre
* `scale` — recovered down-scaling factor (not its reciprocal)
* `found` — `1` present, `0` absent; when `0`, all pose columns are `0`
* `score` — confidence in `[0, 1]`, used for calibration ranking

**Every pair always gets a row**, including one that fails. A missing row
scores zero, so declining beats disappearing.

For the input schema, the flags, the path-resolution rules and troubleshooting,
see each folder's own README — they differ, and the Phase 3 one matters most.

---

## Environment

Python **3.11**, CPU only, no GPU, **no network access at run time**. Each
folder pins its own `requirements.txt` from a working environment, scoped to
what that entry point actually imports.

Measured per-pair time, 4 threads:

| | median | p90 | max |
| --- | --- | --- | --- |
| Phase 2 | 0.89 s | 1.11 s | 1.19 s |
| Phase 3 | 0.58 s | 0.70 s | 0.84 s |

Phase 3 is the faster of the two because its primary path is geometry, not a
network.
