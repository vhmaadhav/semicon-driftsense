# Workstream A (CPU efficiency) — efficiency report

**Agent:** workstream A · **Machine:** dev Mac (ARM), venv313 · All timings below
are **dev machine, indicative** — the integrator re-times on an idle 4-core x86
reference box per the Compute protocol.

## Outline (updated as work lands)

1. **D1 — register.py thread sanity + timing emission** (status: in progress)
   - [x] Read INFERENCE_TWEAKS.md + RUBRIC_ROADMAP.md Compute protocol
   - [ ] Pre-edit baseline of official-20 saved (`/tmp/reg_before.csv` + `.err`)
   - [ ] tests/test_register_runtime_meta.py (TDD: watched fail first)
   - [ ] Thread caps: torch.set_num_threads(min(4, cpu_count)) default,
         cv2.setNumThreads same, before heavy cv2 work; --threads flag still
         overrides; torch.set_flush_denormal(True) in try/except
   - [ ] stderr timing emission: `# per-pair seconds` header,
         `# t,<pair_id>,<seconds>` per row, final `# runtime: median X p90 Y max Z n=N`;
         stdout/CSV byte-identical
   - [ ] Verification: pytest fresh output; official-20 diff before/after

2. **D2 — redundant-FFT ceiling microbenchmark** (status: done)
   - [x] .agents/fft_ceiling_tmp.py on 3 official pairs
   - [x] Go/no-go: GO by the letter (15.6% projected coarse-stage saving;
         19.8% with the full 207-call count) → D3 attempted
   - [x] BUT two premise corrections recorded (50 probe calls not ~100–207;
         template construction ~47% of coarse stage) — see findings

3. **D3 — driftsense/coarse_fft.py spectral coarse scorer** (status: done,
   flag off by default, not wired into the decode — integrator wires after
   audit)
   - [x] tests/test_coarse_fft.py (TDD, 8 tests) -- REMOVED before merge, see D3
   - [x] .agents/coarse_fft_ab.py A/B: value parity 4.8e-08 (25x inside the
         1e-6 bar), argmax 0/150 disagreements, x1.15 vs matchTemplate
   - [x] Net win is honest but small: ~48 ms/pair ≈ 1.5% of pair median

## Findings

### D1 — register.py thread caps + stderr timing (2026-09-03, dev machine)

**Changed (`register.py` only):**
- New `cap_threads(requested=0)`: `torch.set_num_threads(min(4, os.cpu_count() or 4))`
  by default, `--threads` overrides; `torch.set_flush_denormal(True)` best-effort
  in try/except (x86 only; ARM raises, swallowed); `cv2.setNumThreads(same)`
  called before any heavy work (before `I.load_model`).
- stderr emission: `# per-pair seconds` header, `# t,<pair_id>,<seconds>` after
  each row, final `# runtime: median X p90 Y max Z n=N`. stdout untouched.

**Verification (fresh output):**
- TDD: watched `test_stderr_contains_per_pair_timing_lines` FAIL first
  (`'# per-pair seconds' in ''`), then pass.
- `pytest tests/test_register_runtime_meta.py tests/test_submission_parity.py`:
  8 passed. Full suite (excluding 3 PRE-EXISTING broken untracked files
  `test_subpixel.py`/`test_calibration.py`/`test_zip_audit_checkpoint.py`,
  unrelated): **261 passed**.
- Official-20 before/after: the FIRST post-edit run was byte-identical to the
  pre-edit baseline (`cmp` clean, 20 rows + header); baseline stderr empty,
  after stderr carries the 20 `# t,` lines + summary.
- **Cross-process nondeterminism finding (pre-existing, NOT caused by this
  change):** a second post-edit run differed from the baseline in the
  `score` column only (bimodal, up to 0.52 on some pairs). Root-caused as
  pre-existing by stashing the register.py edit and re-running the ORIGINAL
  code: it reproduced the second run's output BIT-EXACTLY. Across all four
  runs (baseline / after-run-1 / after-run-2 / original-code-rerun), `found`,
  `x`, `y`, `theta`, `scale` are identical on all 20 pairs — only the
  network-head score fluctuates (likely nondeterministic reduction order in
  the conv reductions under a many-core thread pool; two output clusters:
  {baseline, after-1} and {after-2, original-rerun}). Conclusion: the edit is
  output-neutral (first post-edit run byte-matched the baseline), and the
  score bimodality is an environment property the integrator should know
  about — it does not affect found/coords, hence no scoring impact under the
  grader's found/pose semantics.

**Official-20 stderr timing summary (dev machine, indicative, no other load):**
```
# runtime: median 3.09 p90 3.49 max 6.14 n=20
```
(baseline run before the edit had stdout `median 3.15s p90 3.55s max 6.19s` —
same noise band, confirming no regression.)

**cv2.setNumThreads caveat (dev Mac only):** the macOS opencv wheel uses the
**GCD parallel framework** (`cv2.getBuildInformation()` → "Parallel framework:
GCD"), which IGNORES `cv2.setNumThreads` — `getNumThreads()` keeps reporting 10
on this 10-core machine no matter what is set. On Linux x86 (the grader box)
the pthreads/TBB backend honors the cap, so the call is correct and required
there; on dev Mac it is a harmless no-op. torch.set_num_threads DOES work here
(verified 4 after cap). Measured microcheck: `cv2.gemm` 4k² 10-thread vs 2-thread
≈126 vs 118 ms — mild oversubscription penalty on this box, but cv2 work here is
DFT-based matchTemplate, not gemm; the grader-box effect (7.08 vs 1.58 s/pair)
is where the real oversubscription damage lives.

### D2 — redundant-FFT ceiling microbenchmark (2026-09-03, dev machine)

**Script:** `.agents/fft_ceiling_tmp.py` (3 official pairs p001–p003,
100 calls each, plain FFT cross-correlation, ZNCC normalization excluded as
instructed).

**Correctness of the harness itself** (worth recording — two FFT-correlation
conventions were refuted before the working one):
- `conjB=True` correlation with the template at the TOP-LEFT of the canvas
  gives `cc[y, x] = Σ s[y+u, x+v]·t[u, v]` at offset (y, x) directly.
- The classic "flip + no conjB" route needs a (th−1, tw−1) extraction offset
  and caused the large mismatches in the first two attempts.
- Working route verified: 3e-6 absolute (2e-7 relative) vs a naive sliding
  dot on a random toy; then exact argmax agreement with
  `cv2.matchTemplate(TM_CCORR)` on all three real probes.

**Timing (mean of 3 pairs):**
```
matchTemplate   : 7.638 ms/call
precomputed-FFT : 4.673 ms/call  (search dft ~2.1 ms amortized; loop 4.65 ms)
per-call saving : 2.966 ms (38.8% of the call)
```
**Projection per the brief's model** (~100-call coarse sweep, coarse = 61.4%
of the 3.09 s median → 1897 ms): **297 ms/pair = 15.6% of the coarse stage →
GO by the letter** (a second projection with the full ~207-call count gives
19.8%). **Verdict: GO at the ≥15% threshold → D3 built.**

**Two premise corrections found while instrumenting the real decode
(matchTemplate patched with a counting/timing wrapper, full
locate_phase2 on p001–p003):**
1. The real decode makes **214 matchTemplate calls/pair, of which only 50 use
   the 500x500 probe** (the brief's "~207-call coarse sweep" actually spends
   most of its calls on the refine crops: 331², 386², 409x499…). The probe
   correlation time is 337–393 ms/pair (6–13% of pair time), and probe-search
   DFT redundancy is at most ~40% of that (~2 ms × 50) ≈ **100 ms/pair ≈
   3.4% of pair median** — an order of magnitude below the brief's model.
2. Correlation is only ~53% of a coarse call: **make_template+_probe costs
   ~6.95 ms/call × 50 ≈ 347 ms/pair**, which no search-side FFT change can
   touch.

### D3 — spectral coarse scorer (2026-09-03, dev machine) — **CODE REMOVED**

> **Removed before merge (PR #48 review).** `driftsense/coarse_fft.py`,
> `tests/test_coarse_fft.py` and the A/B harness were deleted. The module was
> never wired into `locate_phase2`, it measured out at ~1.5% net, and its parity
> test failed on the CI image (OpenCV 5: 16x16 template, spectral
> 0.9999999999999997 vs cv2 0.999978244304657, diff 2.18e-05 against a 1e-06
> assertion). Loosening a parity assertion to green up CI for an unshipped
> module is the wrong trade before submission. The measurement below is kept as
> the record of *why* the approach does not pay.

**Built (flag-gated, default OFF, NOT wired into locate_phase2 — integrator
wires after audit):**
- `driftsense/coarse_fft.py`: `prepare_search(search) -> SpectralSearchIndex`;
  `index.peak_score(template)` returns the same peak ZNCC
  cv2.matchTemplate(TM_CCOEFF_NORMED) would report. Search DFT + float64
  integral moment tables computed once; per call: template DFT +
  mulSpectrums(conjB) + idft + O(1)-per-window moments from the integrals.
- `tests/test_coarse_fft.py`: 8 tests (TDD: watched the module-import failure,
  then the 1e-5 value test fail, then pass). Full contract in the test
  docstring.
- `.agents/coarse_fft_ab.py`: 50 random (scale, rot) templates × 3 official
  pairs, values + wall-clock.

**Three measured engineering findings folded into the implementation:**
1. **Mean-subtract the template before the DFT.** The raw-correlation float32
   DFT carries ~±1.0 absolute error on DC-dominated values ~1.6e7 → 1e-5 ZNCC
   error (fails 1e-6). With centering, the spectral map IS the ZNCC numerator
   exactly (identity: corr(s, t−t̄) = Σs·t − SX·Σt/n since Σ(t−t̄)=0), and
   magnitudes drop to ~1e5 → ~1e-8 ZNCC error.
2. **float64 DFT required.** Even centered, the float32 DFT roundoff is ~2e-6
   relative on the numerator. f64 costs ~18% more (4.44 vs 3.75 ms measured).
3. **CCS-packed spectra** (`DFT_REAL_OUTPUT`, what templmatch uses internally)
   are bit-equivalent (9e-10 vs the complex route) and another ~1.2 ms/call
   cheaper.

**A/B result (official p001–p003, 150 templates total, dev machine indicative):**
```
worst |value diff| : 4.8e-08   (tolerance 1e-6)   PASS
argmax disagreement: 0/150
index setup (once) : 3.02 ms/pair
spectral           : 6.786 ms/call
cv2.matchTemplate  : 7.809 ms/call   (x1.15)
projected coarse-stage saving (50 calls): 48 ms/pair (net win)
```

**Honest bottom line for the integrator:** value parity is solid (25x tighter
than the 1e-6 bar, zero peak disagreements), but the net win is **~48 ms/pair
(≈1.5% of pair median)** — the D2 "15.6–19.8% of coarse stage" projection did
NOT survive contact with (a) the precision requirement (f64 + centering eat
the 38.8% raw per-call saving down to x1.15) and (b) the corrected call
accounting (50 probe calls, not ~100–207). The bigger coarse-stage lever left
on the table is the ~347 ms/pair of make_template+_probe construction
(~47% of probe-stage cost), which is a separate work item.

## What was skipped and why

- **Wiring coarse_fft into locate_phase2 / register.py decode:** explicitly
  out of scope — matching.py and config.py are owned by the integrator, and
  the deliverable says "flag-gated, default off, no wiring into the shipped
  decode". The integrator audits the A/B above and wires it (worth ~48 ms
  per pair ≈ 1.5% of pair median; consider bundling with a make_template
  fast-path or caching, the bigger lever at ~347 ms/pair).
- **FFT rewrite of the refine-stage correlations** (the 331²–409² crops,
  ~160 calls/pair): each crop is a DIFFERENT image per candidate, so there is
  no shared DFT to hoist — cv2 is already optimal there.
- **float32 spectral path:** would be ~1.3 ms/call faster but caps at ~2e-6
  ZNCC error — fails the 1e-6 parity bar; measured and rejected.
- **D3 "peak score" cache reuse of window-moment maps across template
  sizes:** moments depend on (th, tw); the sweep touches ~12–13 sizes × 50
  calls so a per-size (sx, sxx) memo could save ~0.8 of the 1.3 ms/call the
  moment maps cost (≈6% of the call) at ~44 MB for 13 sizes. Measured the
  breakdown (template dft 2.43 + mul 0.33 + idft 2.26 + moments 0.92 +
  assembly 0.38 ms) and left the memo out: sub-millisecond upside, real
  memory cost, and the integrator may prefer restructuring the sweep
  loop-size-first instead.
- **Full 2,250-pair runs / training:** per instructions, nothing beyond the
  official 20 pairs and small tests was run.

## Files created/changed (summary)

| file | change |
|---|---|
| `register.py` | cap_threads() + stderr timing emission (only edits) |
| `tests/test_register_runtime_meta.py` | NEW — 3 tests, TDD |
| `driftsense/coarse_fft.py` | NEW — spectral precomputed-search scorer |
| `tests/test_coarse_fft.py` | NEW — 8 tests, TDD |
| `.agents/fft_ceiling_tmp.py` | NEW — D2 microbenchmark |
| `.agents/coarse_fft_ab.py` | NEW — D3 A/B validation |
| `.agents/A_EFFICIENCY_REPORT.md` | NEW — this report |

---

## CPU campaign, round 2 (2026-09-03, PR #48 review)

All measured on `./venv` (CPU-only torch 2.13, 4 threads) — the graders'
configuration. **Every number below was taken on an idle machine**; a runtime
figure measured while another job holds cores is fiction (one intermediate
reading of 3.49 s median in this session was exactly that, and is discarded).

### Where the time is

Re-measured on CPU rather than the CUDA-capable venv, because the two disagree
about which stage dominates:

| stage | CPU (graded) | issue #7 (GPU venv) |
|---|---:|---:|
| locate (network) | **90.6%** | 21.3% |
| pose_candidates | 8.2% | 66.8% |

Issue #7's SEA-elimination spec therefore targets ~8% of the graded cost. That
also explains the E1 embedding cache measuring as "a clock dud" and the
early-exit sweep looking cheap: both were clocked where the network was not the
bottleneck.

### What paid

| change | end-to-end median | exactness |
|---|---:|---|
| `channels_last` (PR #42) | 4.97 s -> 1.82 s, **2.73x** | 3.6e-05 px, 0 decision changes over 252 pairs |
| **conv+BN folding** | 2.04 s -> 1.91 s, **1.07x** | 1.2e-05 px, 0 decision changes over 252 pairs |

**Correction worth recording.** An isolated forward-pass benchmark put BN folding
at **1.39x** (979.8 -> 706.3 ms on a synthetic 924x924 input), which would imply
~1.26x end-to-end at a 74% network share. The paired pipeline measurement on an
idle machine gives **1.07x** (median 2.04 -> 1.91 s, mean 1.92 -> 1.79 s, p90
2.18 -> 2.05 s, n=10 each). The isolated number is the one to distrust: a
synthetic tensor in a tight loop has different cache behaviour from the real
decode, which interleaves OpenCV work between forwards. Only the pipeline figure
is quoted anywhere else in this PR.

The two profiles above were taken back to back on an idle machine in one run, so
they are directly comparable; the 1.82 s figure in the `channels_last` row comes
from an earlier session and should not be differenced against the 2.04 s
baseline here.

BN folding is an algebraic identity in eval mode: BatchNorm applies a fixed
per-channel scale and shift from its running statistics, which composes into the
preceding convolution's weight and bias. 17 BatchNorm2d layers fold away, so the
network makes 17 fewer kernel launches and 17 fewer passes over the 924x924
activations. Guarded by `DRIFTSENSE_FUSE_BN=0`, and the pass only fuses a
BatchNorm whose immediately preceding sibling is the Conv2d feeding it — an
unrecognised layout degrades to no fusion rather than to a wrong graph.

### What did not pay — measured, not assumed

| idea | result | why |
|---|---:|---|
| Batch the 3 pose hypotheses into one forward | **0.97x** | The 924x924 search input already saturates 4 threads; there is no underutilisation to recover. Outputs bit-identical (0.00e+00), so the idea was sound and simply has no headroom here. |
| Dynamic INT8 quantisation | **0.88x** | The model is 19 Conv2d and **0 Linear**; `quantize_dynamic` covers Linear/LSTM only, so it adds observer overhead and quantises nothing. |
| `torch.jit.trace` + `freeze` | **fails** | `TracingCheckError: Graphs differed across invocation` — `model.py:245` branches on a tensor shape, so the trace does not generalise. |
| Hand-written C / SIMD kernels | **not attempted, and should not be** | Every hot path is already compiled SIMD: 19 convolutions execute in oneDNN's hand-tuned AVX kernels, and `make_template`/`matchTemplate` are OpenCV's C++ SIMD. Beating oneDNN in hand-written C is not a realistic use of the remaining time, and a C extension adds a compile step to a submission that must run unmodified on the graders' machine. The remaining levers here are algorithmic, not low-level. |

### Algorithmic levers deliberately NOT taken

* **Crop the search around the coarse estimate.** 924^2 -> ~400^2 would be
  ~3-5x, and `pose_candidates` already yields `coarse_x_native`. But
  `ContextBranch` exists precisely to see the whole lattice and decide *which*
  repeat is correct; cropping would remove that evidence on exactly the
  wrong-basin pairs that are the dominant failure mode. Needs a full A/B before
  anyone ships it.
* **Static INT8 PTQ.** Plausibly ~2x, but it changes numerics, needs a
  calibration set, and `torch.ao.quantization` is deprecated in torch 2.13.
