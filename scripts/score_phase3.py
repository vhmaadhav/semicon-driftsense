"""Phase 3 rubric scorer: the briefing's 85 measurable points.
loc 40 (1/2/3/5 px tiers) + scale 10 + rotation 10 + rejection F1 15 + AUC 10.
Masking rule matches driftsense/rubric.py: a declined present pair forfeits
localisation and pose (register.py zero-fills the row, and that is submitted)."""
import sys, numpy as np, pandas as pd
LOC=((1.,1.),(2.,.8),(3.,.6),(5.,.4)); SC=((.01,1.),(.02,.6),(.05,.3)); RO=((.25,1.),(.5,.6),(1.,.3))
def tier(v,t):
    for b,c in t:
        if v<=b: return c
    return 0.
def auc(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if not len(a) or not len(b): return float("nan")
    return float((a[:,None]>b[None,:]).mean()+.5*(a[:,None]==b[None,:]).mean())
def run(pred,gt,label,rot_sign=None):
    p=pd.read_csv(pred); g=pd.read_csv(gt)
    m=p.merge(g,on="pair_id",suffixes=("","_gt"))
    pres=m[m.present==1].copy(); absent=m[m.present==0]
    pres["err"]=np.hypot(pres.x-pres.x_gt,pres.y-pres.y_gt)
    ok=pres.found==1                       # declined present pair -> zero credit
    pres["loccred"]=np.where(ok,[tier(e,LOC) for e in pres.err],0.)
    pres["serr"]=np.abs(pres.scale-pres.scale_gt)/pres.scale_gt
    pres["sc"]=np.where(ok,[tier(e,SC) for e in pres.serr],0.)
    signs=[+1.,-1.] if rot_sign is None else [rot_sign]
    best=None
    for s in signs:
        rerr=np.abs(pres.theta*s-pres.theta_gt)
        cr=np.where(ok,[tier(e,RO) for e in rerr],0.).mean()
        if best is None or cr>best[0]: best=(cr,s,rerr)
    rotc,used_sign,rerr=best
    tp=int(((m.present==1)&(m.found==1)).sum()); fp=int(((m.present==0)&(m.found==1)).sum())
    fn=int(((m.present==1)&(m.found==0)).sum())
    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
    good=m[(m.present==1)&(np.hypot(m.x-m.x_gt,m.y-m.y_gt)<=5)&(m.found==1)].score
    bad=m[~m.index.isin(good.index)].score
    A=auc(good,bad)
    L,S,R,J,C=pres.loccred.mean()*40,pres.sc.mean()*10,rotc*10,f1*15,(0 if np.isnan(A) else A)*10
    print(f"\n=== {label} ===  n={len(m)} present={len(pres)} absent={len(absent)}")
    print(f"  loc      {L:6.2f}/40   credit {pres.loccred.mean():.4f}  med {pres.err.median():.3f}px  "
          f"<=1px {(pres.err<=1).sum()}/{len(pres)}  <=5px {(pres.err<=5).sum()}  >20px {(pres.err>20).sum()}")
    print(f"  scale    {S:6.2f}/10   med rel err {pres.serr.median()*100:.3f}%")
    print(f"  rotation {R:6.2f}/10   med |dtheta| {np.median(rerr):.3f} deg (sign {used_sign:+.0f})")
    print(f"  reject   {J:6.2f}/15   F1 {f1:.4f}  tp/fp/fn {tp}/{fp}/{fn}")
    print(f"  calib    {C:6.2f}/10   AUC {A:.4f}")
    print(f"  SUBTOTAL {L+S+R+J+C:6.2f}/85")
    return dict(loc=L,scale=S,rot=R,rej=J,cal=C,total=L+S+R+J+C,err=pres[["pair_id","err"]],
                theta_gt=pres.theta_gt.values, rerr=np.asarray(rerr), sign=used_sign)
if __name__=="__main__":
    run(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else "run")
