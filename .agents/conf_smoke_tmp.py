import os, sys
import cv2
cv2.setNumThreads(2)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import infer as I
from driftsense.matching import locate_phase2
import torch; torch.set_num_threads(4)
REF = os.path.join(HERE, ".agents", "ref_material")
model, device = I.load_model(I.DEFAULT_WEIGHTS)
for pid in ["p001", "p015", "p018"]:
    ref = I.read_gray(os.path.join(REF, "reference", f"{pid}.png"))
    sea = I.read_gray(os.path.join(REF, "search", f"{pid}.png"))
    r = locate_phase2(model, ref, sea, device, refine=True, verification="zncc", band=False)
    print(f"{pid}: conf={r['confidence']:.4f} zncc={r.get('zncc', float('nan')):.4f} "
          f"net={r.get('score', float('nan')):.4f} found@0.487={int(r['confidence']>=0.487)}")
