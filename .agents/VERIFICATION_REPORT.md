# Phase-2 robust hypothesis verification report

Development used a 200-pair, generator-based proxy: 40 default-acquisition
Set-A pairs and 40 pairs at each low/medium/high/severe Set-B proxy severity.
The frozen confirmation used 100 newly generated pairs with disjoint seeds.
These are local synthetic proxies, not organizer-held-out or external data.
There were 149 present development pairs and 83 present confirmation pairs.

## 1. Baseline

The baseline is the unchanged maximum native-ZNCC hypothesis. Candidate
coordinates are native-ZNCC refined and are measured before `polish_pose()`.
The A+B value uses the competition weighting `0.45*A + 0.55*B`.

| Split | Set A credit | Set B credit | Weighted A+B | <=1 px | <=2 px | <=3 px | <=5 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.9655 | 0.8883 | 0.9231 | 76.51% | 87.92% | 93.96% | 96.64% |
| Held-out proxy | 0.9846 | 0.9057 | 0.9412 | 80.72% | 91.57% | 93.98% | 96.39% |

The checkpoint SHA-256 remained
`90db89f9861c2c9ea386eaa03e45ff03fc4962dc7e349aa00423621a5fce1488`.
No training or weight modification occurred.

## 2. K=3 oracle ceiling

| Group | Present | Baseline <=5 px | K=3 oracle <=5 px | K=3 oracle <=1 px | Recoverable gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development Set A | 29 | 100.00% | 100.00% | 89.66% | 0.00 pp |
| Development Set B severity 1 | 33 | 100.00% | 100.00% | 100.00% | 0.00 pp |
| Development Set B severity 2 | 30 | 93.33% | 100.00% | 93.33% | 6.67 pp |
| Development Set B severity 3 | 26 | 96.15% | 100.00% | 80.77% | 3.85 pp |
| Development Set B severity 4 | 31 | 93.55% | 96.77% | 38.71% | 3.23 pp |
| Development overall | 149 | 96.64% | 99.33% | 80.54% | 2.68 pp |
| Held-out proxy overall | 83 | 96.39% | 98.80% | 81.93% | 2.41 pp |

Four development baseline failures and two held-out-proxy failures contained a
correct alternative hypothesis. One development failure and one held-out-proxy
failure had no <=5 px candidate and therefore cannot be fixed by verification.

## 3. Rank-only results

| Split | Set A | Set B | Weighted A+B | <=1 px | <=2 px | <=3 px | <=5 px | Rescued | Broken | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.8621 | 0.8800 | 0.8719 | 75.84% | 85.91% | 91.28% | 92.62% | 3 | 9 | -6 |
| Held-out proxy | 0.9846 | 0.8514 | 0.9114 | 77.11% | 86.75% | 89.16% | 91.57% | 1 | 5 | -4 |

Rank has rescue capacity but is rejected as an independent selector because it
breaks substantially more baseline successes than it rescues.

## 4. Band-only results

| Split | Set A | Set B | Weighted A+B | <=1 px | <=2 px | <=3 px | <=5 px | Rescued | Broken | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.9310 | 0.9050 | 0.9167 | 77.18% | 88.59% | 94.63% | 97.32% | 2 | 1 | +1 |
| Held-out proxy | 0.9846 | 0.9143 | 0.9459 | 80.72% | 91.57% | 95.18% | 97.59% | 2 | 1 | +1 |

Band is the strongest independent robust selector on confirmation, but its
development Set-A regression is too large for direct promotion.

## 5. DoG-only results

| Split | Set A | Set B | Weighted A+B | <=1 px | <=2 px | <=3 px | <=5 px | Rescued | Broken | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.9655 | 0.9000 | 0.9295 | 77.85% | 89.26% | 94.63% | 97.32% | 3 | 2 | +1 |
| Held-out proxy | 0.9846 | 0.8914 | 0.9334 | 79.52% | 90.36% | 92.77% | 95.18% | 0 | 1 | -1 |

DoG did not reproduce its development gain and remains diagnostic only. It is
not computed by the optional production consensus path.

## 6. Majority results

Voters are native ZNCC, rank, and common band. Two agreeing voters select a
hypothesis; otherwise native ZNCC wins.

| Split | Set A | Set B | Weighted A+B | <=1 px | <=2 px | <=3 px | <=5 px | Rescued | Broken | Net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.9655 | 0.9050 | 0.9322 | 77.85% | 89.26% | 95.30% | 97.99% | 2 | 0 | +2 |
| Held-out proxy | 0.9846 | 0.9200 | 0.9491 | 81.93% | 92.77% | 95.18% | 97.59% | 1 | 0 | +1 |

Weighted localisation credit improves by 0.0092 in development and 0.0079 on
confirmation. At 40 localisation points these correspond to approximately
`+0.37` and `+0.31` competition points.

## 7. Consensus-override results

Consensus changes the ZNCC winner only when rank and band agree on the same
different hypothesis. With exactly these three voters, this is algebraically
equivalent to the majority rule, so the measured tables are identical.

| Split | Set A delta | Set B delta | Weighted delta | <=5 px delta | Scale credit delta | Rotation credit delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | +0.0000 | +0.0167 | +0.0092 | +1.34 pp | -0.0014 | -0.0021 before polish |
| Held-out proxy | +0.0000 | +0.0143 | +0.0079 | +1.20 pp | +0.0010 | +0.0007 before polish |

After applying the existing `polish_pose()` once to only the consensus winner,
development localisation/scale/rotation credits were `0.9168 / 0.8986 /
0.9514`; held-out-proxy values were `0.9301 / 0.9160 / 0.9519`. Polishing did
not move x/y or scale and was never run on all candidates.

## 8. Rescue-vs-break analysis

| Split | Recoverable failures | Rank can rescue | Band can rescue | DoG can rescue | Any robust verifier | Consensus rescued | Consensus broken |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 4 | 3 | 2 | 3 | 4 | 2 | 0 |
| Held-out proxy | 2 | 1 | 2 | 0 | 2 | 1 | 0 |

The robust-score union covers every recoverable failure in both runs. The
conservative consensus captures half of development headroom and half of
confirmation headroom without breaking a baseline <=5 px success.

## 9. Results by Set-B severity

Development localisation credit:

| Severity | ZNCC | Rank | Band | DoG | Majority | Consensus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0000 | 0.9697 | 1.0000 | 0.9697 | 1.0000 | 1.0000 |
| 2 | 0.9200 | 0.9533 | 0.9533 | 0.9867 | 0.9533 | 0.9533 |
| 3 | 0.9077 | 0.9154 | 0.9077 | 0.9077 | 0.9077 | 0.9077 |
| 4 | 0.7226 | 0.6839 | 0.7548 | 0.7355 | 0.7548 | 0.7548 |

Frozen held-out-proxy comparison:

| Severity | ZNCC | Consensus | Delta |
| --- | ---: | ---: | ---: |
| 1 | 1.0000 | 1.0000 | +0.0000 |
| 2 | 1.0000 | 1.0000 | +0.0000 |
| 3 | 0.8947 | 0.9474 | +0.0526 |
| 4 | 0.7125 | 0.7125 | +0.0000 |

Severity 4 improves in development and is unchanged in confirmation; it does
not worsen materially.

## 10. Runtime overhead

Single-process microbenchmark, `cv2.setNumThreads(4)` and
`torch.set_num_threads(4)`, seven repeats:

| Operation | Median time |
| --- | ---: |
| Rank full 1000x1000 search | 33.87 ms |
| Band full search | 5.60 ms |
| DoG full search | 12.87 ms |
| Three rank templates + local matches | 3.70 ms |
| Three band templates + local matches | 2.21 ms |
| Three DoG templates + local matches | 2.38 ms |
| All diagnostic verifier work | 60.64 ms |

Observed candidate instrumentation medians were 62.99 ms in development and
60.11 ms on confirmation. The optional consensus path skips DoG, so its
component-sum estimate is 45.39 ms per pair. The research harness total was
7.03 s/pair development and 6.02 s/pair confirmation; that includes up to
three existing neural forwards and is not verifier overhead.

## 11. Held-out confirmation

The configuration was frozen in `.agents/VERIFICATION_FREEZE.md` before
confirmation. The confirmation data used new seeds and was evaluated once.
No sigma, radius, score rule, threshold, or weight was changed afterward.

| Acceptance item | Development | Held-out proxy | Result |
| --- | ---: | ---: | --- |
| Weighted A+B credit > baseline | +0.0092 | +0.0079 | Pass |
| Set B credit improves | +0.0167 | +0.0143 | Pass |
| Set A regression <=0.005 | +0.0000 | +0.0000 | Pass |
| Rescued substantially exceed broken | 2 vs 0 | 1 vs 0 | Pass |
| Severity 4 not materially worse | +0.0323 | +0.0000 | Pass |
| Pose scale meaningful regression | -0.0014 | +0.0010 | Pass |
| Pose rotation meaningful regression | -0.0021 pre-polish | +0.0007 pre-polish | Pass |
| Added median runtime <0.10 s | 0.0630 s diagnostic | 0.0601 s diagnostic | Pass |
| Rough +0.5 point promotion target | about +0.37 | about +0.31 | Not met |

An organizer/external shard was not available locally, so the held-out result
is confirmation against an independent synthetic generation run only.

## 12. Recommendation: SHIP / REJECT / INVESTIGATE

**SHIP the frozen consensus selector as an optional, explicitly enabled
Phase-2 mode; keep submission/default behavior at `verification="zncc"` until
an organizer-like external shard is available.**

The recommendation is SHIP because every minimum acceptance criterion passed,
the effect reproduced with zero observed broken baseline successes, and the
runtime/dependency cost is small. Confidence, rejection, model, K=3 pose
search, native-ZNCC subpixel placement, and `register.py` remain unchanged.
The gain is below the stronger +0.5-point target, so enabling it by default is
not recommended from synthetic evidence alone.
