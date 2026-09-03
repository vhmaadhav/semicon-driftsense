# The efficiency lever is the network, not the coarse sweep

**Date:** 2026-09-02 · Measured on the CPU-only venv (`./venv`, torch 2.13.0+cpu),
4 threads, single process — the configuration the graders actually run.

## Issue #7's split is a GPU measurement and points at the wrong stage

Issue #7 states the breakdown as **pose_candidates 66.8% / locate (network)
21.3%** and concludes "the efficiency lever is the coarse sweep". That profile
was taken on `venv313`, which has CUDA-capable torch. On the reference
configuration the ordering is reversed:

| stage | CPU (`./venv`, graded config) | issue #7 (GPU venv) |
|---|---:|---:|
| **locate (network)** | **90.6%** | 21.3% |
| pose_candidates | 8.2% | 66.8% |
| polish_pose | 0.9% | 10.8% |

`scripts/profile_pair.py data/ext_p2/test_B_0000 --n 8 --threads 4`

So SEA/FGSE-style coarse-sweep elimination — the whole of issue #7's spec — is
aimed at **8% of the graded cost**. Anything that does not reduce network time
cannot move the efficiency component materially.

This also explains why the E1 template-embedding cache measured as "a clock
dud" (PR #18) and why the early-exit sweep (`.agents/EARLY_EXIT.txt`) traded
accuracy for a speedup that looked small: both were clocked where the network
was not the bottleneck.

## The fix: channels_last, 2.73x, no decision changes

The network was executing in NCHW. oneDNN's convolution kernels want a blocked
layout, so every convolution paid a reorder of its activations.
`model.to(memory_format=torch.channels_last)` lets the blocked layout persist
across the stack.

`scripts/profile_pair.py`, same command, before and after:

| | NCHW | channels_last |
|---|---:|---:|
| median | 4.97 s | **1.82 s** |
| mean | 5.08 s | 1.76 s |
| p90 | 6.29 s | 1.99 s |
| max | 6.60 s | 2.08 s |

**2.73x on the median**, against a 5 s median target the previous number was
already failing on a slower box (an external CPU benchmark reported 5.89 s/pair
mean at `75c4572`).

Post-fix the split is network 74.0% / pose_candidates 22.6%, so the coarse
sweep becomes worth looking at *after* this, not before.

### It is not bit-identical, and that is stated rather than glossed

Re-association inside the blocked kernels moves values slightly. Measured over
252 pairs (sets A/B/C, CPU, same shards, stride 6):

| field | max abs diff |
|---|---|
| x | 3.56e-05 px |
| y | 2.77e-05 px |
| score | 3.16e-06 |
| scale, theta | 0.00e+00 |

That is ~28,000x below the 1 px credit tier. What matters for the rubric is
whether any *decision* moves, and none do:

* `found` decisions flipped at the shipped 0.18 threshold: **0**
* localisation credit tiers changed: **0**
* full rubric on the 252-pair subset: **78.65 both arms**, localisation 35.81
  both arms, rejection F1 0.9701 both arms

An earlier 6-pair spot check reported "bit-identical"; that was too small a
sample to see the 1e-5 differences and the claim is withdrawn.

### Escape hatch

`DRIFTSENSE_CHANNELS_LAST=0` restores NCHW, for a platform where the oneDNN
path misbehaves. The conversion is also wrapped: a failure warns and keeps the
default layout rather than costing the run.

## Related hardening

`locate_phase2`'s sub-pixel row refinement is now wrapped in a try/except.
`register.py` zero-fills the entire row on any exception, so a throw inside a
late-added refinement stage would turn a correctly located pair into
`found=0, score=0.0` — which, in the fixed output contract, is indistinguishable
from a confident rejection. `np.polyfit` can raise `LinAlgError` on a degenerate
fit. The failure now degrades to the rigid answer, which is what the pipeline
produced before that stage existed.

This matters for diagnosis as well: an external benchmark at `75c4572` showed
two pairs with `score` exactly `0.0`, which is that zero-fill path, not a
rejection. Of five missed present pairs there, two were crashes and three were
genuine low scores — a distinction invisible in the CSV and only recoverable
from `register.py`'s stderr warning.
