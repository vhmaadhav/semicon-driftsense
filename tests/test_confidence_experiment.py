"""Experimental confidence head: missingness, holdout isolation and inference parity."""
import numpy as np
import pandas as pd
import pytest


def _features():
    return pd.DataFrame(dict(net_score=[.8,.2,.7,.3], zncc=[.7,.1,.6,.2],
        peak_ratio=[.1,.9,.2,.8],pose_peak=[.8,.3,.7,.4],
        psr=[99.,9.,49.,4.],apce=[999.,99.,499.,49.],
        winner_margin=[.1,np.nan,0.,-.2]))


def test_missing_margin_is_not_conflated_with_a_measured_tie():
    from driftsense.confidence_experiment import feature_matrix
    x=feature_matrix(_features())
    assert x.shape == (4,8)
    assert x[1,6] == x[2,6] == 0
    assert x[1,7] == 1 and x[2,7] == 0
    assert x[0,4] == pytest.approx(np.log(100))
    assert x[0,5] == pytest.approx(np.log(1000))


def test_unavailable_base_evidence_is_rejected():
    from driftsense.confidence_experiment import feature_matrix
    d=_features();d.loc[0,'psr']=np.nan
    with pytest.raises(ValueError,match='finite'):
        feature_matrix(d)


def test_source_groups_cannot_cross_partitions_and_row_order_does_not_matter():
    from driftsense.confidence_experiment import partition
    groups=pd.Series([f'canvas{i}' for i in range(1000)]*2)
    a=partition(groups,20260908)
    assert np.array_equal(a[:1000],a[1000:])
    assert np.array_equal(a[::-1],partition(groups.iloc[::-1],20260908))
    assert set(a)=={'fit','tune','assess'}


def test_saved_model_predicts_without_labels_and_does_not_refit_standardisation():
    from driftsense.confidence_experiment import fit_head,predict
    train=_features();model=fit_head(train,np.array([1,0,1,0]))
    score=predict(train,model)
    assert np.isfinite(score).all() and (score>=0).all() and (score<=1).all()
    assert score[0]>score[1]
    # A second extreme row must not change the first row's preprocessing.
    other=train.iloc[[0,1]].copy();other.loc[other.index[1],'apce']=1e20
    assert predict(other,model)[0] == pytest.approx(score[0])


def test_assessment_score_matches_shared_rubric_for_declined_real_match():
    from scripts.experiment_confidence85 import assess
    from driftsense.rubric import score
    d=pd.DataFrame(dict(pair_id=['a','b','c','d'],set=['A','B','C','C'],
        gt_found=[1,1,0,0],x=[10.]*4,y=[10.]*4,gt_x=[10.]*4,gt_y=[10.]*4,
        scale=[10.]*4,gt_scale=[10.]*4,theta=[0.]*4,gt_rot=[0.]*4,score=[.9,.1,.1,.9]))
    actual=assess(d,d.score.to_numpy(),.18)
    expected,_=score(d,.18,quiet=True)
    assert actual['total']==pytest.approx(sum(v[1] for k,v in expected.items() if k!='calibration_submitted'))
    assert actual['declined_real']==1 and actual['accepted_absent']==1
    assert actual['loc_B']==0


def test_paired_interval_for_identical_candidate_is_exactly_zero():
    from scripts.confirm_confidence85 import paired_interval
    d=pd.DataFrame(dict(pair_id=['a','b','c','d'],set=['A','B','C','C'],
        gt_found=[1,1,0,0],x=[10.]*4,y=[10.]*4,gt_x=[10.]*4,gt_y=[10.]*4,
        scale=[10.]*4,gt_scale=[10.]*4,theta=[0.]*4,gt_rot=[0.]*4,score=[.9,.9,.1,.3]))
    r=paired_interval(d,d.score.to_numpy(),.18,draws=20)
    assert r['delta_p025']==r['delta_p975']==0


def test_assessment_mutations_cannot_change_fitted_head_or_threshold(tmp_path):
    from driftsense.confidence_experiment import partition
    from scripts.experiment_confidence85 import run
    n=120
    d=pd.concat([_features()]*30,ignore_index=True)
    d['pair_id']=[f'pair{i}' for i in range(n)]
    d['source_group']=[f'canvas{i}' for i in range(n)]
    d['set']=np.resize(['A','B','C'],n)
    d['gt_found']=(d['set']!='C').astype(int)
    d['score']=np.where(d.gt_found==1,.8,.1)
    for k in ('x','y','gt_x','gt_y','scale','gt_scale'):d[k]=10.
    d['theta']=d['gt_rot']=0.
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    run(d,a,20260908)
    changed=d.copy();mask=partition(d.source_group,20260908)=='assess'
    changed.loc[mask,'psr']*=1000
    # Change assessment labels too; fit/tune choices must stay identical.
    changed.loc[mask & (changed.index%2==0),'gt_x']+=20
    run(changed,b,20260908)
    assert (a/'candidate-20260908.json').read_bytes()==(b/'candidate-20260908.json').read_bytes()
