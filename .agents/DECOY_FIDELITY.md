
## Same-seed A/B decode (200 absent pairs, shipped decode, 2026-09-02)

Generated twice with identical seed 424242: once with the old same-band decoys
(bb91eb0 worktree), once with the pitch-factor decoys (this head). Same decoder,
same shipped config (0.18, band=False). This isolates the decoy-pitch effect:

| | before | after |
|---|---:|---:|
| absent median | 0.0381 | 0.0368 |
| absent max | 0.4807 | **0.4084** (−15%) |
| false-accepts @0.18 | 32/200 (16.0%) | 29/200 (14.5%) |
| above 0.30 | 11 (5.5%) | **6 (3.0%)** |

Reading: the pitch fix shrinks the worst tail (the 0.45-0.60 bucket empties,
max −15%) but the false-accept rate at 0.18 only moves 16.0% -> 14.5%. The
residual tail SUGGESTS non-pitch factors dominate the remaining risk (no
paired bad->good / good->bad transition analysis was run to attribute it
fully): pitch is one cue, and the learned score
still fires on same-family same-preset texture. Implication for the blind
set: the organizers' ~20% absent pairs will produce false accepts at a rate
our ext_p2 F1 (0.908) understates, because our C decoys are still easier than
organizer-real ones on non-pitch axes. The verification-veto (unused
zncc/pose_peak/margin signals) and the threshold re-sweep are the remaining
inference-side counters; the rest belongs to the #30 retrain with the widened
severity ladder AND organizer-style decoys.
