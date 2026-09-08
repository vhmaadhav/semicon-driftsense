#!/usr/bin/env python3
"""Decode a fresh jury-layout proxy set once; compare a frozen confidence head.

No fitting, threshold selection or model updates occur here. CPU threads are
capped; concurrent machine activity still prevents judge-latency claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from driftsense.confidence_experiment import predict
from driftsense.config import SHIPPED_BAND, SHIPPED_SUBPIXEL_ROWS, SHIPPED_VERIFICATION
from scripts.experiment_confidence85 import assess
from scripts.grade_emulation import rubric


def paired_interval(frame, candidate_score, threshold, draws=2000, seed=20260908):
    """Paired, within-set replacement bootstrap; one independent canvas/row."""
    baseline=frame.reset_index(drop=True).copy()
    candidate=baseline.copy(); candidate['score']=candidate_score
    groups=[np.flatnonzero(baseline['set'].to_numpy()==s) for s in ('A','B','C')]
    if any(len(g)==0 for g in groups):
        raise ValueError('paired interval needs all grayscale sets')
    rng=np.random.default_rng(seed); deltas=[]
    for _ in range(draws):
        idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        deltas.append(rubric(candidate.iloc[idx],threshold)['total']-rubric(baseline.iloc[idx],.18)['total'])
    if not np.isfinite(deltas).all():
        raise ValueError('a bootstrap draw has an undefined rubric; inspect denominators')
    return dict(draws=draws,seed=seed,delta_p025=float(np.percentile(deltas,2.5)),
                delta_p975=float(np.percentile(deltas,97.5)))


def decode(directory, weights, output):
    import torch
    import cv2
    import infer as I
    from driftsense.model import net_from_checkpoint
    from driftsense.matching import locate_phase2
    torch.set_num_threads(2);cv2.setNumThreads(2)
    # Explicit CPU load equivalent to infer.load_model's CPU branch. Avoid
    # its automatic MPS selection while another worktree is using the GPU.
    ckpt=torch.load(weights,map_location='cpu',weights_only=True)
    model=net_from_checkpoint(ckpt)
    model.load_state_dict(ckpt.get('model',ckpt));model.eval()
    model=I._fuse_conv_bn(model).to(memory_format=torch.channels_last)
    pairs=pd.read_csv(directory/'pairs.csv')
    gt=pd.read_csv(directory/'ground_truth.csv')
    manifest=pd.read_csv(directory/'manifest_jury.csv')
    if not pairs.pair_id.is_unique or len(pairs)!=200:
        raise ValueError('confirmation requires exactly 200 unique pairs')
    if manifest['set'].value_counts().to_dict()!={'A':70,'B':70,'C':40,'D':20}:
        raise ValueError('confirmation composition must be A70/B70/C40/D20')
    table=pairs.merge(gt,on='pair_id',validate='one_to_one').merge(
        manifest[['pair_id','set','architecture','severity','seed']],on='pair_id',validate='one_to_one')
    if len(table)!=200 or not table.seed.is_unique:
        raise ValueError('confirmation labels/groups must match every pair')
    rows=[]; fingerprints=[]
    for n,r in table.iterrows():
        refpath=directory/r.reference_path;seapath=directory/r.search_path
        fingerprints.append(dict(pair_id=r.pair_id,reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                                 search_sha256=hashlib.sha256(seapath.read_bytes()).hexdigest()))
        t0=time.perf_counter()
        result=locate_phase2(model,I.read_gray(str(refpath)),I.read_gray(str(seapath)),
            torch.device('cpu'),refine=True,band=SHIPPED_BAND,
            subpixel_rows=SHIPPED_SUBPIXEL_ROWS,verification=SHIPPED_VERIFICATION)
        record=dict(pair_id=r.pair_id,set=r['set'],architecture=r.architecture,severity=int(r.severity),
                    source_group=str(r.seed),gt_found=int(r.present),gt_x=r.x,gt_y=r.y,
                    gt_scale=r.scale,gt_rot=r.theta,x=result['x'],y=result['y'],
                    scale=result['scale'],theta=result['theta'],
                    score=result['confidence'],net_score=result['score'],secs=time.perf_counter()-t0)
        for key in ('zncc','peak_ratio','pose_peak','psr','apce','winner_margin'):
            record[key]=float(result.get(key,np.nan))
        rows.append(record)
        if (n+1)%10==0:
            pd.DataFrame(rows).to_csv(output/'confirmation_predictions.csv',index=False)
            print(f'decoded {n+1}/200',flush=True)
    pd.DataFrame(fingerprints).to_csv(output/'confirmation_hashes.csv',index=False)
    return pd.DataFrame(rows)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--head',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
    head_bytes=a.head.read_bytes();head=json.loads(head_bytes)
    d=decode(a.directory,a.weights,a.output)
    candidate_all=predict(d,head)
    gray=d[d['set'].isin(('A','B','C'))].copy()
    candidate=candidate_all[d['set'].isin(('A','B','C'))]
    baseline=assess(gray,gray.score.to_numpy(),.18)
    candidate_result=assess(gray,candidate,head['threshold'])
    interval=paired_interval(gray,candidate,head['threshold'])
    optical=d['set']=='D'
    def d_credit(stat,t):
        from driftsense.rubric import score
        temp=d.copy();temp['score']=stat
        _,per=score(temp,t,quiet=True)
        return float(per.loc[optical,'loc_credit'].mean())
    result=dict(domain='fresh independently seeded generator-proxy set; not official blind',
                model_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                head_sha256=hashlib.sha256(head_bytes).hexdigest(),
                gt_sha256=hashlib.sha256((a.directory/'ground_truth.csv').read_bytes()).hexdigest(),
                baseline=baseline,candidate=candidate_result,paired_interval=interval,
                delta85=candidate_result['total']-baseline['total'],
                baseline_D_credit=d_credit(d.score.to_numpy(),.18),
                candidate_D_credit=d_credit(candidate_all,head['threshold']),
                timing_note='2 CPU threads, concurrent activity; not a judge efficiency benchmark')
    result['passes_confirmation_gate']=bool(result['delta85']>=.35 and interval['delta_p025']>=0)
    d['candidate_score']=candidate_all;d['candidate_found']=(candidate_all>=head['threshold']).astype(int)
    d.to_csv(a.output/'confirmation_predictions.csv',index=False)
    (a.output/'confirmation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    print(f"METRIC: {result['delta85']:.8f}")
    return result

if __name__=='__main__':main()
