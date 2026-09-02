# Issue 45 — Phase 2 generator integration

The four public entry points are thin adapters over `generator/src/phase2_audit.py`; structural rendering remains in the existing `generator/src` pipeline. Run from the repository root:

```text
python generator/generate_phase2.py --output-dir generator/output --seed 45045 --pairs 20 --force
python generator/baseline.py --output-dir generator/output --threshold 0.55
python generator/score.py --output-dir generator/output --threshold 0.55
python generator/contact_sheet.py --output-dir generator/output
python generator/check_submission.py --output-dir generator/output
```

The measured audit contains exactly A8/B6/C4/D2: 16 present and 4 absent pairs, both DRAM and FinFET, all 12 repository presets, z=8 and z=12, theta=-5, 0, and +5 degrees. Seed 45045 generated the package in 23.38 seconds with 647.65 MiB Python peak traced allocation using NumPy 2.3.5 and OpenCV 4.13.0.

All 16 present pairs passed raw-intensity primary verification and an independent gradient verifier. R1 maximum affine round-trip error was 2.572e-12 px; R4 found every present footprint visible; R5 maximum realised-raster label correction was 4.903 px. Both z=12/theta=5 and non-integer-z resampling cases beat the no-antialiasing control in MAE and PSNR.

The coarse NCC baseline produced 0.550 mean present credit and classification F1 0.9412 at threshold 0.55. Set-C same-family absent NCC scores ranged 0.3254–0.8658, demonstrating plausible negatives while the manifest retains the semantic absence contract. Severity-level baseline error was not monotone at level 4 because periodic structure produced a harder wrong basin at level 3; this is reported as a limitation, not relabelled away.

No organizer reference or sample data was used for training, fine-tuning, threshold fitting, or generator tuning. Detailed per-pair metrics are generated in `output/score.json`, and the runtime report is generated in `output/REPORT.md`.
