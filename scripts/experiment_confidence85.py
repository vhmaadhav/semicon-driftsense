#!/usr/bin/env python3
"""One predeclared confidence candidate; separate fit, tune and assessment.

All partitions are DEVELOPMENT because the source pool was used historically.
Never describe these assessment splits as a newly untouched external test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driftsense.confidence_experiment import fit_head, partition, predict
from driftsense.rubric import score as shared_score
from scripts.grade_emulation import rubric


def assess(frame, statistic, threshold):
    d = frame.copy()
    d['score'] = statistic
    r = rubric(d, threshold)
    shared, _ = shared_score(d, threshold, quiet=True)
    r['auc_submitted'] = shared['calibration_submitted'][0]
    found = np.asarray(statistic) >= threshold
    gt = d.gt_found.to_numpy() == 1
    r.update(threshold=float(threshold), correct_rejections=int((~found & ~gt).sum()),
             declined_real=int((~found & gt).sum()), accepted_absent=int((found & ~gt).sum()))
    return r


def tune_threshold(frame, statistic):
    """Select using tuning data only, maximising the complete /85 formula."""
    d = frame.copy()
    d['score'] = statistic
    thresholds = np.unique(np.r_[0., statistic])
    candidates = [(rubric(d, float(t))['total'], float(t)) for t in thresholds]
    finite = [(v, t) for v, t in candidates if np.isfinite(v)]
    if not finite:
        raise ValueError('no finite /85 score in tuning partition')
    # Conservative tie break: lower threshold retains more real localisations.
    return max(finite, key=lambda x: (x[0], -x[1]))[1]


def run(frame, output, seed):
    parts = partition(frame.source_group, seed)
    split = {s: frame.loc[parts == s].copy() for s in ('fit', 'tune', 'assess')}
    for s, d in split.items():
        if set(d['set']) != {'A', 'B', 'C'} or set(d.gt_found.unique()) != {0, 1}:
            raise ValueError(f'{s} needs all grayscale strata and both classes')
    head = fit_head(split['fit'], split['fit'].gt_found)
    tuned = predict(split['tune'], head)
    threshold = tune_threshold(split['tune'], tuned)
    control_threshold = tune_threshold(split['tune'], split['tune'].score.to_numpy())
    head.update(seed=seed, threshold=threshold,
                split_counts={k: len(v) for k,v in split.items()},
                split_groups={k: sorted(v.source_group.unique().tolist()) for k,v in split.items()})
    # Persist frozen fit/tune choices before evaluating the assessment labels.
    model_path = output / f'candidate-{seed}.json'
    model_path.write_text(json.dumps(head, indent=2) + '\n')
    d = split['assess']; candidate = predict(d, head)
    baseline = assess(d, d.score.to_numpy(), .18)
    control = assess(d, d.score.to_numpy(), control_threshold)
    result = assess(d, candidate, threshold)
    delta = result['total'] - baseline['total']
    table = d.copy(); table['candidate_score'] = candidate
    table['candidate_threshold'] = threshold; table['candidate_found'] = (candidate >= threshold).astype(int)
    table.to_csv(output/f'assessment-{seed}.csv', index=False)
    return dict(seed=seed, split_counts=head['split_counts'], fixed_baseline=baseline,
                threshold_only_control=control, candidate=result, delta85=delta,
                delta_vs_threshold_control=result['total']-control['total'],
                passes_development_gate=bool(delta >= .35), model=model_path.name)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',required=True,type=Path)
    ap.add_argument('--groups',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args(argv)
    d=pd.read_csv(a.csv); meta=pd.read_csv(a.groups)
    d=d[d['set'].isin(('A','B','C'))].copy()
    if not d.pair_id.is_unique:
        raise ValueError('prediction pair IDs must be unique')
    d=d.merge(meta[['pair_id','source_group']],on='pair_id',how='left',validate='one_to_one')
    if d.source_group.isna().any():
        raise ValueError('every grayscale prediction needs source-group metadata')
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(protocol='development assessment only; previously reused pool',
                input_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),
                groups_sha256=hashlib.sha256(a.groups.read_bytes()).hexdigest(),
                model_imported_by_submission=False, runs=[])
    for seed in (20260908,20260909):
        r=run(d,a.output,seed); result['runs'].append(r)
        print(json.dumps(r),flush=True)
    result['continue_to_fresh_confirmation']=all(r['passes_development_gate'] for r in result['runs'])
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"METRIC: {min(r['delta85'] for r in result['runs']):.8f}")
    return result

if __name__ == '__main__':
    main()
