# Judging-box campaign: 4-core x86, 8 GB — and why 81.6 was 2 points too high

**Date:** 2026-09-03. Run on the merge commit of PR #51 (`27b0be7`), with
`git status` clean over `driftsense/`, `register.py`, `infer.py` and
`scripts/` — the decode is byte-identical to what that PR shipped. Every
number here is from this repo's `judging/` tree.

## 0. Why this campaign exists

PR #51 measured on an **Apple M4, arm64** and said so. The task material names
a **4-core x86 CPU, 8 GB RAM, no GPU, no network, Python 3.11**. This is that
measurement, plus five sets instead of three so the between-set spread is
visible.

It also answers a question that was not the goal: the previous best figure of
**81.45 mean / 81.93 best** does not reproduce, and the reason is a generator
defect, not a regression.

## 1. The headline

Five independent 200-pair sets (`S1`–`S5`, seeds 1–5), A70/B70/C40/D20 per
slide 4, built by `judging/organizer_generator/gen_200.py`. **Ours**, not
organizer-issued; not the blind benchmark.

| set | loc /40 | scale /10 | rot /10 | reject /15 | calib /10 | **/85** | F1(rej) | Set D | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 36.17 | 9.57 | 9.25 | 14.63 | 9.93 | **79.54** | 0.9750 | 1.000 | 1.164 |
| S2 | 35.38 | 9.38 | 8.74 | 14.29 | 9.97 | **77.76** | 0.9524 | 1.000 | 1.099 |
| S3 | 37.96 | 9.40 | 8.51 | 14.81 | 9.98 | **80.67** | 0.9877 | 0.970 | 1.099 |
| S4 | 36.58 | 9.39 | 9.08 | 14.81 | 9.98 | **79.84** | 0.9877 | 0.980 | 1.202 |
| S5 | 37.17 | 9.23 | 8.95 | 14.63 | 9.94 | **79.92** | 0.9756 | 0.970 | 1.117 |
| **mean** | **36.65** | **9.39** | **8.91** | **14.63** | **9.96** | **79.54** | **0.9757** | **0.984** | **1.136** |
| **sd** | 0.98 | 0.12 | 0.29 | 0.22 | 0.02 | **1.08** | 0.0144 | 0.015 | 0.045 |

Latency pooled over all 1,000 pairs, idle machine, 4-thread cap: **median
1.147 s, mean 1.221 s, p90 1.883 s, p99 2.083 s, max 2.191 s**. Zero pairs over
the 5 s median budget, zero over the 20 s hard timeout; the slowest single pair
in 1,000 was 2.191 s. Both bonus gates met on all five sets under both readings
of the +6 condition: **+10 / +10**.

sd is 1.08 points across five sets. A difference under about a point is not a
result at this sample size — which is exactly why the three-set PR #51 campaign
could not have caught what section 3 describes.

## 2. The constraints were verified, not assumed

Applied twice and then read back from inside the running process. A cap that is
set but never takes effect has cost this project points before, so the evidence
is the realised value, never the flag. Per-run files: `judging/out/<tag>/env.txt`.

| constraint | mechanism | realised, all 6 runs |
|---|---|---|
| 4 cores | `taskset` onto 4 **distinct physical** cores (no HT siblings) | `sched_getaffinity` → exactly 4 CPUs |
| 8 GB RAM | transient systemd scope, `MemoryMax=8G` | cgroup `memory.max` = 8589934592 |
| no swap | `MemorySwapMax=0` | cgroup `memory.swap.max` = 0 |
| no GPU | CPU-only venv | `torch.cuda.is_available()` False, torch 2.13.0+cpu |
| Python 3.11 | — | 3.11.16 |
| x86 | — | `uname -m` x86_64 |
| no network | `unshare -n` spot-check, 5 pairs | 0 interfaces; all rows written, no download attempted |
| peak memory | `/usr/bin/time` | 0.93–1.07 GB — **13% of the cap** |
| thread cap | `min(4, os.cpu_count())` | 4 threads; measured CPU use 336–339% of 400% |

`os.cpu_count()` reports 24 here rather than 4, because affinity does not change
it. The shipped cap is `min(4, ...)`, so it lands on 4 either way and the judge's
box and this emulation agree — but a future change to that expression would
diverge silently between the two, and only the realised thread count would show it.

## 3. The 81.45/81.93 figures came from a severity ladder that never fired

`driftsense.generate.build_one` only calls `sample_severity_params` — the
function that draws the coherent per-knob degradation — when `_shi > _slo`.
A severity range pinned as a single point (`lo == hi`) fails that
strictly-greater test, so the pair renders as **generic independent per-knob
draws at severity 0.0** while still being labelled "Set-B severity N". That is
issue #31; the audit fixture fixes it with `SEVERITY_PIN_EPSILON = 1e-6`.

`gen_200.py` inherits the fixed `pose_for`, so its Set B is genuinely degraded.
PR #51's `gen_200.py` was never committed and is gone, so what it did cannot be
read — but the defect's fingerprint can be reproduced and priced.

Direct A/B. Identical code, identical seed (1), identical composition, identical
box, identical 4 pinned cores. The **only** difference is `severity=(t, t)`
versus `severity=(t, t + 1e-6)`:

| | realised Set B severity | Set B credit | loc /40 | rot /10 | **/85** | median s |
|---|---|---:|---:|---:|---:|---:|
| `S1` — ladder fires | 0.25 / 0.50 / 0.75 / 1.00 (17–18 pairs each) | 0.8257 | 36.17 | 9.25 | **79.54** | 1.164 |
| `S1_legacy` — degenerate pin | **0.0 × all 70 pairs** | 0.9343 | 38.55 | 8.98 | **81.57** | 0.897 |
| delta | — | −0.109 | −2.38 | +0.27 | **−2.03** | +0.267 |

**81.57.** That is the "highest was 81.6" this campaign was asked to explain,
reproduced on demand by switching the degradation off. PR #51's Set B credit of
0.9086 / 0.9143 sits inside the legacy arm's 0.9343 and nowhere near the
degraded arm's 0.8257.

Two independent corroborations that the legacy arm is easier data, not a scoring
difference:

* **Label verification needed fewer resamples.** `max_verify_attempts_used` was
  14 for `S1_legacy` against 23 for `S1`. Undegraded crops are easier to verify
  as globally unambiguous.
* **It ran faster.** 0.897 s median against 1.164 s. Undegraded pairs trip the
  early-exit gates more often, so the easier data also produced the better
  latency headline. Both of PR #51's leading numbers moved the same way for the
  same reason.

**Conclusion: 79.54 is the honest figure and 81.45/81.93 should be retired**,
the same way this repo already retired 72.55 / 75.51 / 76.23. Nothing
regressed; the earlier data was not degraded in the way it claimed to be. This
is the fourth time "set but not in effect" has cost this project — the lesson
is unchanged: assert on the realised distribution, never on the flag.
`gen_200.py` does assert it, per pair, and raises rather than shipping a pair
whose realised severity misses its target.

## 4. Where the remaining points are

Set B is the whole gap, and inside Set B it is monotone in severity. 350 B pairs
across the five sets:

| Set B severity | n | loc credit | med err | ≤1px | >5px | declined | rot credit | scale credit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 90 | 0.9644 | 0.44 px | 85.6% | 0.0% | 0 | 0.8989 | 0.9144 |
| 2 | 90 | 0.9111 | 0.77 px | 65.6% | 0.0% | 0 | 0.8900 | 0.9000 |
| 3 | 85 | 0.8329 | 0.80 px | 60.0% | 7.1% | 2 | 0.7494 | 0.8886 |
| 4 | 85 | **0.6988** | 1.51 px | 36.5% | **15.3%** | 6 | **0.6694** | 0.8014 |
| all B | 350 | 0.8543 | — | — | — | 8 | — | — |
| Set A | 350 | 0.9920 | 0.31 px | — | — | 1 | 0.9665 | 0.9957 |

Severity 4 alone costs about 1.5 of the 5.5 missing points. Of the 9 present
pairs wrongly declined across 1,000, 8 are severity 3–4 and every one had a true
error of 291–1004 px — genuine lock-on failures that the rejector correctly
declined, not calibration slop. Set C gave up exactly **1** false accept in 200.

### The composition caveat that bounds all of this

`SEVERITY_LEVELS = (1, 2, 3, 4)` mapped to `level / 4.0` puts a quarter of Set B
at severity_continuous **1.0 — the generator's ceiling**. The four levels are
disclosed by slide 4; their magnitudes are **not**. The even split matches the
reviewed audit fixture's mean level (2.5), so it is a defensible reading, but if
the organizers' level 4 is milder than our ceiling the true score sits somewhere
in **[79.54, 81.57]** — and that interval is exactly the legacy A/B above, since
the legacy arm is the degenerate case where level 4 means nothing at all.

Planning on 79.54 is the conservative choice and is what this campaign
recommends. `data/ext_p2` (2,250 pairs, **75.92**) remains harder still and
remains the planning number of record; these 1,000 pairs do not supersede it.

## 5. Slower-core sensitivity

This CPU is heterogeneous, so "4 cores" has more than one answer and the choice
is recorded rather than left implicit. The table in section 1 pins 4 P-cores
(4.8 GHz). The same 200 pairs on 4 E-cores (3.7 GHz):

| | subtotal /85 | every component | median | p90 | max |
|---|---:|---|---:|---:|---:|
| S1, 4 P-cores | 79.5422 | — | 1.164 s | 1.873 s | 2.092 s |
| S1, 4 E-cores | 79.5422 | **identical** | 2.859 s | 4.538 s | 4.972 s |

The decode is core-independent — a useful determinism check, and the same
property PR #51 established across processes. But the **sub-2 s median is a
property of the faster core, not of the code**: on E-cores the median is 2.86 s.
It still clears the 5 s contract with margin, and still never approaches the
20 s timeout. Any claim of "1.15 s/pair" must name the core it was measured on.

## 6. Methodology: only idle-machine readings

A 5-pair spot check taken while the generator held cores measured **5.90 s
median** on the same binary and the same 4 pinned cores that gave 1.15 s idle —
a 5× inflation from background load alone. Every run in this document was serial
by construction, on an otherwise idle machine, for that reason. Readings taken
under load are not quoted.

## 7. A reporting bug found and fixed

The driver aggregated `judging/out/S*/rubric.json`, which matched the E-core
rerun of S1 as a sixth "set". That double-counted S1's accuracy and pooled two
different machine configurations into one latency column, reporting **median
1.423 s** for what is actually 1.136 s on the P-core configuration — while the
table's own title still said "5 x 200". `judging/aggregate.py` now derives its
title from the data and **refuses** to pool runs whose labels collide or that
look like variants; sensitivity runs go through `--also` and print separately.

## 8. Files

* `judging/organizer_generator/gen_200.py` — the 200-pair set builder, with
  `--legacy-severity-pin` to reproduce section 3's degenerate arm
* `judging/run_judge.sh` — one constrained run, with the preflight that proves
  the cap is in effect before spending 200 pairs
* `judging/run_all.sh` — the whole campaign, serial
* `judging/score_rubric.py` — imports `scripts/eval_ext.py::score`, adds Set D
  and the two bonus gates
* `judging/aggregate.py` — cross-set table and gate verdicts
* `judging/out/<tag>/{env.txt,rubric.txt,rubric.json,register.stderr,time.txt}`
* `judging/AGGREGATE_CORRECTED.txt` — the corrected 5-set table
