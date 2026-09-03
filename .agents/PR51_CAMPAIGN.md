# PR #51 measurement campaign

Every number the PR body claims comes from here. Written in response to the
2026-09-03 review, which asked for reproducibility evidence, an A/B for the
candidate deduplication, documentation that matches the code, and an honest
end-to-end latency instead of a "sub-second" headline.

## 0. What was measured on, and what it is not

**Machine.** Apple M4, arm64, 10 cores (4 performance + 6 efficiency), macOS
26.5.1, Python 3.13.9, torch 2.13.0, OpenCV 5.0.0, NumPy 2.5.2. Thread cap 4
(the restored `min(4, os.cpu_count())` default).

This is **not** the judge's box. The judging environment is described as a
4-core x86 CPU; nothing here ran on x86. Capping to 4 threads makes the *shape*
of the measurement closer to the reference machine — it stops a 10-core number
being reported as if the judge could reproduce it — but an arm64 latency is not
an x86 latency and is not presented as one.

**Data.** `S1`, `S2`, `S3`: 200 pairs each (A70 / B70 / C40 present-absent mix
plus D20 optical), built by `judging/organizer_generator/gen_200.py`. That
script is **ours**. It implements the Phase 2 spec recipe on top of the
vendored upstream generator; it is not organizer-issued data, and these 600
pairs are **not** the official blind benchmark. They are an internal,
spec-compliant synthetic evaluation. Any figure below is evidence about this
pipeline on this data, and nothing more.

Scored with `judging/score_rubric.py`, which ports `scripts/eval_ext.py::score`
and adds the Set D credit and the two bonus gates.

## 1. Cross-process reproducibility (review item 2)

Three clean `register.py` processes over S1, compared row by row:

| | found flips | max abs dscore | Spearman(score) | max abs dx / dy / dtheta / dscale |
|---|---:|---:|---:|---|
| r1 vs r2 | 0 / 200 | 0.000e+00 | 1.000000000 | 0 / 0 / 0 / 0 |
| r1 vs r3 | 0 / 200 | 0.000e+00 | 1.000000000 | 0 / 0 / 0 / 0 |

Rubric, all three runs: SUBTOTAL 81.13, F1(reject) 1.0000, AUC 1.0000.

Output is **bit-identical** across processes, not merely stable, so the
calibration component cannot move between runs and `found` cannot flip. The
large cross-process variation seen in earlier measurements does not reproduce
on this head. The most likely reason is the restored thread cap: thread count
changes oneDNN's reduction order, and before the cap it was a free variable
that tracked whatever else the machine was doing.

## 2. Early exit is decision-neutral (review items 4)

Shipped gates versus evaluating every hypothesis, identical decode otherwise
(`EARLY_EXIT_GATES = ()`), S1:

| | shipped | no early exit |
|---|---:|---:|
| found flips | — | **0 / 200** |
| max abs dscore | — | **0.000e+00** |
| max abs dx / dy / dtheta / dscale | — | **0 / 0 / 0 / 0** |
| median latency (see caveat) | 1.648 s | 1.940 s |
| SUBTOTAL / 85 | 81.13 | 81.13 |

Bit-identical output: the gates skip work that provably could not have changed
the answer on these pairs. The two latency figures come from the same
sequential campaign but not from a guaranteed-idle machine, so treat the ~1.18x
as indicative of direction, not as a quotable speedup; the accuracy conclusion
does not depend on it, because the decode is deterministic (section 1).

The PR text originally described a `0.88 / 0.55 / 0.30` rule while the code
implemented `0.72 / 0.72 / 0.35 / 0.04`. The **code** is the version that was
measured, so the documentation was corrected to the code rather than the other
way round, and the constants now live in one place —
`driftsense/config.py:EARLY_EXIT_GATES` — with `tests/test_early_exit_gates.py`
pinning them as literals so prose and implementation cannot drift apart again.

## 3. Candidate deduplication (review item 6)

The review's objection was structural: dedup merged at `dscale < 0.35` and
`dtheta < 1.0 deg` while the downstream refinement only covers `+/-3%` scale
(~0.30 at m = 10) and `+/-0.8 deg`, so it could discard a hypothesis that the
polish could never have reached from the survivor.

Fixed by **deriving** the radius from the polish bands instead of writing it
out as literals: `POLISH_SCALE_BAND` / `POLISH_ROT_BAND` are now the defaults
of `polish_pose` *and* the dedup radius, so widening the polish window widens
the radius and never the reverse. `tests/test_early_exit_gates.py` asserts the
coupling.

A/B at the corrected radius versus no dedup at all, S1:

| | shipped | no dedup |
|---|---:|---:|
| found flips | — | **0 / 200** |
| max abs dscore | — | 8.83e-02 |
| max abs dx / dy | — | 0.370 px / 0.085 px |
| SUBTOTAL / 85 | 81.13 | **81.13** |

(Both from the same campaign; no latency figure is quoted for this A/B because
the no-dedup run overlapped another job.)

Not bit-identical — dedup does occasionally drop a candidate that would have
been selected with a slightly different sub-pixel result — but zero decisions
and zero rubric points move.

## 4. Polish budget: one set said revert, three sets said wash

PR #51 reduced `_refine_pose_local` from 2 rounds x 8 iterations to 1 x 4, and
`polish_pose` from 2 x 7 to 1 x 6, without an A/B. It got one.

**On S1 alone the deeper budget looks clearly better:**

| S1 | reduced (1/4 + 1/6) | deep (2/8 + 2/7) |
|---|---:|---:|
| set A loc credit | 0.9429 | 0.9457 |
| set B loc credit | 0.9086 | 0.9143 |
| localisation pts | 36.96 | 37.14 |
| scale | 9.86 | 9.89 |
| Set D credit | 0.880 | 0.900 |
| **SUBTOTAL / 85** | 81.13 | **81.34** |

+0.21 points, 0 found flips, median abs dx 0.0088 px. On that evidence the
reduction was reverted.

**Two more sets reversed the conclusion:**

| set | reduced | deep | delta |
|---|---:|---:|---:|
| S1 | 81.13 | 81.34 | +0.21 |
| S2 | **81.93** | 81.52 | **-0.41** |
| S3 | 81.30 | 81.32 | +0.02 |
| **mean** | **81.45** | 81.39 | **-0.06** |

S2's rotation credit falls 9.23 -> 8.94 under the deeper polish. The per-set
spread is about 0.4 points; a 0.06 mean difference across three sets does not
resolve against that, so the honest reading is **no measurable accuracy
difference**. Concluding otherwise from S1 was fitting the decision to one
sample -- the same error that produced the `fused6` result in PR #48, where a
statistic that gained on its fitting pool lost on untouched data.

**With accuracy neutral it becomes a latency decision.** Both budgets, S1,
same 200 pairs, machine idle for both:

| | median | mean | p90 | max |
|---|---:|---:|---:|---:|
| reduced (1/4 + 1/6) | **0.915 s** | **1.015 s** | **1.457 s** | 2.337 s |
| deep (2/8 + 2/7) | 1.421 s | 1.421 s | 1.741 s | 2.071 s |

**1.55x** for nothing measurable. The reduced budget ships.

### A note on every latency figure in this document

Earlier readings of this same comparison gave 2.588 s and 1.586 s for the deep
budget and 1.648 s for the reduced one -- they were taken while other jobs held
cores, including the test suite. They are discarded. Only the back-to-back
idle-machine numbers above are quoted, and accuracy is unaffected either way
because the decode is deterministic (section 1).

## 5. What `ref_feat` precomputation actually does

The PR body claims it "eliminates redundant forward passes". It does not.
`DriftSenseNet.forward` has kept a single-slot template-embedding cache since
E1 (`model.py`, `use_template_cache = True`), keyed on the input bytes, and the
template handed to the encoder is `make_template(reference, SCALE, 0)` for
every pose hypothesis — byte-identical. The encoder therefore already ran
**once per pair**, not once per hypothesis, before this PR.

What hoisting `ref_feat` into `locate_phase2` saves is the cache *lookup*: a
`.tobytes()` of a 100x100 float array plus a dict comparison, per hypothesis.
That is real but small, and the claim in the PR body is corrected accordingly.
The change is still worth keeping — it makes the sharing explicit at the call
site instead of implicit in a cache — but it is not a forward-pass saving.

## 6. Final configuration

Measured with the shipped configuration: 4-thread cap, `EARLY_EXIT_GATES` as
committed, dedup at the polish radius, polish budget 1x4 and 1x6. Machine idle
throughout; nothing else was run against these three passes.

| set | loc /40 | scale /10 | rot /10 | reject /15 | calib /10 | **subtotal /85** | F1(reject) | Set D | median s/pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 36.96 | 9.86 | 9.31 | 15.00 | 10.00 | **81.13** | 1.0000 | 0.880 | 0.915 |
| S2 | 37.73 | 9.97 | 9.23 | 15.00 | 10.00 | **81.93** | 1.0000 | 0.870 | 0.911 |
| S3 | 37.22 | 9.77 | 9.31 | 15.00 | 10.00 | **81.30** | 1.0000 | 0.950 | 1.037 |
| **mean / all 600** | **37.30** | **9.87** | **9.28** | **15.00** | **10.00** | **81.45** | **1.0000** | **0.900** | **0.966** |

Latency over all 600 pairs, idle machine, 4-thread cap: **median 0.966 s, mean
1.011 s, p90 1.377 s, max 2.337 s**. End-to-end wall clock including process
start and model load was 190 s and 216 s for the two 200-pair sets timed that
way — about **1.0 s/pair**. The +6 and +4 bonus gates are met on all three sets
(Set D credit >= 0.40 with A/B/C >= 0.50; rejection F1 >= 0.90).
