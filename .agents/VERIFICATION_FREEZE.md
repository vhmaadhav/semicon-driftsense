# Verification configuration freeze

Frozen before held-out inspection on 2026-08-29.

- Existing pose hypotheses: up to K=3, unchanged.
- Native ZNCC refinement radius: existing value, unchanged.
- Rank transform: radius 2 (5x5, 24 neighbours).
- Common band: Gaussian sigma 2.0.
- DoG diagnostic: sigma 0.8 minus sigma 3.0.
- Local verification radius: 4 px.
- Promoted confirmation policy: `verify_consensus_override` / `verification="consensus"`.
- Override rule: change the native-ZNCC winner only when rank and band choose
  the same different hypothesis.
- DoG is diagnostic only and is not a voter.
- Confidence, rejection threshold, final native-ZNCC placement, pose polishing,
  model weights, K, and pose search remain unchanged.

No configuration value may be changed after inspecting held-out results.
