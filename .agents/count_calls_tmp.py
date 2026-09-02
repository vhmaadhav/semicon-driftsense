import os, sys, time
import cv2, numpy as np
cv2.setNumThreads(2)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import infer as I
import driftsense.matching as M
import torch; torch.set_num_threads(4)

counts = {"make_template": 0, "matchTemplate": 0, "mt_px": 0, "locate_fwd": 0}
orig_mt = cv2.matchTemplate
def mt(*a, **k):
    counts["matchTemplate"] += 1
    im, tm = a[0], a[1]
    counts["mt_px"] += int(im.shape[0]*im.shape[1]*tm.shape[0]*tm.shape[1])
    return orig_mt(*a, **k)
cv2.matchTemplate = mt
orig_make = M.make_template
def make(*a, **k):
    counts["make_template"] += 1
    return orig_make(*a, **k)
M.make_template = make
orig_loc = M.locate
def loc(*a, **k):
    counts["locate_fwd"] += 1
    return orig_loc(*a, **k)
M.locate = loc

model, device = I.load_model(I.DEFAULT_WEIGHTS)
REF = os.path.join(HERE, ".agents", "ref_material")
import csv
n = 0
t0 = time.perf_counter()
with open(os.path.join(REF, "pairs.csv"), newline="") as f:
    for r in csv.DictReader(f):
        if n >= 6: break
        ref = I.read_gray(os.path.join(REF, r["reference_path"]))
        sea = I.read_gray(os.path.join(REF, r["search_path"]))
        M.locate_phase2(model, ref, sea, device, refine=True, verification="zncc", band=False)
        n += 1
tot = time.perf_counter() - t0
print(f"pairs={n} total={tot:.1f}s per-pair={tot/n:.2f}s")
for k, v in counts.items():
    print(f"  {k:16} {v:8d}  per-pair {v/n:8.1f}")
