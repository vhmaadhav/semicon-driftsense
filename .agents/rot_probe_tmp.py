import os, sys, csv
import cv2, numpy as np
cv2.setNumThreads(2)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import infer as I
import driftsense.matching as M
import torch; torch.set_num_threads(4)

REF = os.path.join(HERE, ".agents", "ref_material")
GT = {r["pair_id"]: r for r in csv.DictReader(open(os.path.join(REF, "ground_truth.csv")))}
model, device = I.load_model(I.DEFAULT_WEIGHTS)

pid = "p010"
gt = GT[pid]
ref = I.read_gray(os.path.join(REF, "reference", f"{pid}.png"))
sea = I.read_gray(os.path.join(REF, "search", f"{pid}.png"))
res = M.locate_phase2(model, ref, sea, device, refine=True, verification="zncc", band=False)
print(f"p010: gt theta={gt['theta']} scale={gt['scale']} pos=({gt['x']},{gt['y']})")
print(f"  decode: theta={res['theta']:.4f} scale={res['scale']:.4f} err={np.hypot(res['x']-float(gt['x']),res['y']-float(gt['y'])):.3f}px")

# Is the polished rotation at least a local max of the polish objective?
x, y, m0, r0 = res["x"], res["y"], res["scale"], res["theta"]
def fit(r):
    mm, _, peak = M.polish_pose(ref, sea, x, y, m0, r)
    return peak
for r in [float(gt['theta'])-0.5, float(gt['theta'])-0.25, float(gt['theta']),
          float(gt['theta'])+0.25, float(gt['theta'])+0.5, r0]:
    print(f"  polish-peak at rot={r:7.3f}: {fit(r):.6f}")
# Raw correlation surface vs rotation at the located point (no polish):
tpl = lambda r: M.make_template(ref, m0, r)
def raw(r):
    t = tpl(r)
    win = sea[max(int(y-60),0):int(y+60), max(int(x-60),0):int(x+60)]
    return M._peak_score(win, t) if t.shape[0]<win.shape[0] else -1
print("  raw corr vs rotation (no polish):")
for r in np.arange(float(gt['theta'])-1.0, float(gt['theta'])+1.01, 0.25):
    print(f"    rot={r:6.2f}: {raw(float(r)):.6f}")
