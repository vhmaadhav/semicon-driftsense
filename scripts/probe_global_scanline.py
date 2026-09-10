"""Image-only global-versus-local row-motion correction on original development."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np, pandas as pd
from scripts.train_fresh_rows import split_groups
from scripts.train_setb85 import accuracy


def relative_shift(image, x, y, half):
    if (
        y < 3
        or y + 3 >= image.shape[0]
        or x - half - 8 < 0
        or x + half + 8 > image.shape[1]
    ):
        return None
    row = image[y : y + 1, x - half - 8 : x + half + 8].astype(np.float32)
    shifts = []
    for j in [-3, -2, -1, 1, 2, 3]:
        ref = image[y + j : y + j + 1, x - half : x + half].astype(np.float32)
        if ref.std() < 2:
            continue
        corr = cv2.matchTemplate(row, ref, cv2.TM_CCOEFF_NORMED)[0]
        k = int(corr.argmax())
        if corr[k] < 0.5 or k in (0, 16):
            continue
        a, b, c = corr[k - 1 : k + 2]
        den = a - 2 * b + c
        delta = np.clip(0.5 * (a - c) / den, -0.5, 0.5) if abs(den) > 1e-8 else 0
        shifts.append(k - 8 + delta)
    return float(np.median(shifts)) if len(shifts) >= 3 else None


def run(data, output):
    root = Path("experiments/setb85")
    d = pd.read_csv(root / "context/baseline.csv")
    c = d.copy()
    tables = []
    for file in sorted(data.glob("*/manifest.csv")):
        q = pd.read_csv(file)
        q["base"] = str(file.parent)
        tables.append(q)
    inputs = pd.concat(tables).set_index("pair_id")
    updates = 0
    for i, r in d.iterrows():
        if r.score < 0.18:
            continue
        v = inputs.loc[r.pair_id]
        im = cv2.imread(str(Path(v.base) / v.search_path), cv2.IMREAD_GRAYSCALE)
        global_shift = relative_shift(
            im, im.shape[1] // 2, int(round(r.y)), im.shape[1] // 2 - 20
        )
        local_shift = relative_shift(im, int(round(r.x)), int(round(r.y)), 40)
        if global_shift is not None and local_shift is not None:
            delta = global_shift - local_shift
            if abs(delta) <= 4:
                c.loc[i, "x"] = r.x + delta
                updates += 1
    group = split_groups(d)
    out = {
        "updates": updates,
        "note": "Fixed image-only correction; original historically seen development, not confirmation.",
        "splits": {},
    }
    for name, mask in [
        ("validation", (group >= 7) & (group < 9)),
        ("assessment", group == 9),
        ("all", np.ones(len(d), bool)),
    ]:
        out["splits"][name] = {
            k: {"baseline": accuracy(d, mask, k), "candidate": accuracy(c, mask, k)}
            for k in ["A", "B"]
        }
    output.mkdir(exist_ok=True)
    c.to_csv(output / "candidate.csv", index=False)
    (output / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.data, a.output)
    except Exception:
        import traceback

        a.output.mkdir(exist_ok=True)
        (a.output / "FAILED.txt").write_text(traceback.format_exc())
        raise
