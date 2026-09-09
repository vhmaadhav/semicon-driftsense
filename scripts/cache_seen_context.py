#!/usr/bin/env python3
"""Convert previously consumed confirmation shards into explicit development data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import infer as I
from driftsense.context_row import patches
from driftsense.matching import make_template


def run(source, output):
    frames, ts, ss = [], [], []
    for shard in sorted(source.glob("shard*")):
        if not shard.is_dir():
            continue
        d = pd.read_csv(shard / "baseline.csv")
        inputs = pd.read_csv(shard / "inputs.csv").set_index("pair_id")
        available = []
        for r in d.itertuples():
            v = inputs.loc[r.pair_id]
            reference = I.read_gray(str(Path(v.base) / v.reference_path))
            search = I.read_gray(str(Path(v.base) / v.search_path))
            pair = patches(search, make_template(reference, r.scale, r.theta), r.x, r.y)
            available.append(pair is not None)
            if pair is None:
                pair = np.zeros((17, 80), np.float32), np.zeros((17, 112), np.float32)
            ts.append(pair[0])
            ss.append(pair[1])
        d["available"] = available
        d["pair_id"] = shard.name + ":" + d.pair_id.astype(str)
        d["source_group"] = shard.name + ":" + d.source_group.astype(str)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    assert d.pair_id.is_unique
    output.mkdir(parents=True, exist_ok=True)
    d.to_csv(output / "baseline.csv", index=False)
    np.savez(output / "patches.npz", template=np.array(ts), search=np.array(ss))
    (output / "SEEN_DATA.txt").write_text(
        "Previously consumed confirmation; now development only. Never report as fresh holdout.\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)
