#!/usr/bin/env python3
"""Wait for a training run to finish, then judge it end to end, unattended.

Runs the decision protocol from NIGHT_LOG.md without a human in the loop:

1. Wait until the run's history has `--epochs` entries and no train.py is alive.
2. Stop pool generation, so the evaluation gets the whole machine (evaluation
   here is scene-generation bound, exactly like training was).
3. Shortlist candidate epochs from the in-loop history (free, already computed).
4. Separate them on freshly generated scenes with stream_eval -- the 300-scene
   val split cannot resolve differences this small, and the 100-scene in-loop
   metric once picked the *worst* of three candidates.
5. Average the tail checkpoints (SWA) and measure that too.
6. Paired bootstrap every candidate against the on-this-machine baseline.
7. Only if a candidate wins, measure it once on the three test splits.

Never writes weights/driftsense.pt.

    python scripts/judge_run.py --run weights/driftsense_v5f.pt --epochs 24
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, "venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    log("$ " + " ".join(os.path.basename(c) if c == PY else c for c in cmd))
    return subprocess.run(cmd, cwd=HERE, **kw)


def train_alive() -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*train.py*' }).Count"],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip()) > 0
    except ValueError:
        return False


def stop_generation():
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*build_pool*' -or "
         "$_.CommandLine -like '*gen_data*' } | "
         "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
         "-ErrorAction SilentlyContinue }"],
        capture_output=True, text=True)


def score(h: dict) -> float:
    """The selection score train.py uses: accuracy first, median as tiebreak."""
    return (1.0 - h["acc@5px"]) * 1000.0 + h["median_px"]


def stream_eval(weights: str, out_json: str, n: int, workers: int) -> dict | None:
    if os.path.exists(out_json):
        log(f"reusing {out_json}")
    else:
        r = run([PY, "scripts/stream_eval.py", "--weights", weights,
                 "-n", str(n), "--workers", str(workers), "--out", out_json])
        if r.returncode != 0:
            log(f"FAILED stream_eval on {weights}")
            return None
    with open(os.path.join(HERE, out_json)) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="weights/driftsense_v5f.pt")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--baseline", default="results/stream/driftsense_rtx.json")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--swa-tail", type=int, default=5)
    p.add_argument("--timeout-min", type=int, default=420)
    p.add_argument("--no-test", action="store_true",
                   help="skip the final test-split measurement")
    args = p.parse_args()

    tag = os.path.splitext(os.path.basename(args.run))[0]
    hist_path = os.path.join(HERE, args.run.replace(".pt", "_history.json"))

    # --- 1. wait -----------------------------------------------------------
    log(f"waiting for {args.epochs} epochs of {tag} (timeout {args.timeout_min} min)")
    deadline = time.time() + args.timeout_min * 60
    while time.time() < deadline:
        n_done = 0
        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    n_done = len(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        if n_done >= args.epochs and not train_alive():
            log(f"training finished: {n_done} epochs")
            break
        if n_done and not train_alive():
            log(f"train.py is gone after only {n_done}/{args.epochs} epochs -- "
                f"judging what exists")
            break
        time.sleep(60)
    else:
        log("TIMEOUT waiting for training; judging whatever exists")

    with open(hist_path) as f:
        history = json.load(f)
    if not history:
        raise SystemExit("no history to judge")

    # --- 2. give the evaluation the whole machine --------------------------
    log("stopping pool generation so evaluation is not competing with it")
    stop_generation()
    time.sleep(5)

    # --- 3. shortlist ------------------------------------------------------
    ranked = sorted(history, key=score)
    picks = []
    for h in ranked[:args.candidates]:
        picks.append(int(h["epoch"]))
    last = int(history[-1]["epoch"])
    if last not in picks:
        picks.append(last)
    picks = sorted(set(picks))
    log(f"shortlist by in-loop score: epochs {picks}")
    log("(the in-loop metric is 100 scenes and cannot separate these -- "
        "that is what stream_eval is for)")

    jobs = []  # (name, weights_path, json_path)
    for e in picks:
        w = args.run.replace(".pt", f"_e{e}.pt")
        if os.path.exists(os.path.join(HERE, w)):
            jobs.append((f"{tag}_e{e}", w, f"results/stream/{tag}_e{e}.json"))
        else:
            log(f"missing checkpoint {w}")

    # --- 4/5. SWA over the tail -------------------------------------------
    tail = sorted(int(h["epoch"]) for h in history)[-args.swa_tail:]
    tail_paths = [args.run.replace(".pt", f"_e{e}.pt") for e in tail]
    tail_paths = [q for q in tail_paths if os.path.exists(os.path.join(HERE, q))]
    if len(tail_paths) >= 2:
        swa = args.run.replace(".pt", "_swa.pt")
        r = run([PY, "scripts/average_checkpoints.py", *tail_paths, "--out", swa])
        if r.returncode == 0:
            jobs.append((f"{tag}_swa", swa, f"results/stream/{tag}_swa.json"))
            log(f"SWA over epochs {tail}")

    # --- evaluate every candidate on the same fresh scenes ------------------
    os.makedirs(os.path.join(HERE, "results", "stream"), exist_ok=True)
    results = {}
    for name, w, j in jobs:
        blob = stream_eval(w, j, args.n, args.workers)
        if blob:
            results[name] = (j, blob)
            log(f"{name}: acc@5 {blob['acc@5px']:.4f}  median {blob['median_px']:.2f}px")

    if not results:
        raise SystemExit("nothing evaluated")

    # --- 6. paired bootstrap ----------------------------------------------
    log("\npaired bootstrap against the baseline measured on this machine")
    run([PY, "scripts/compare_checkpoints.py", "--baseline", args.baseline,
         args.baseline, *[j for j, _ in results.values()]])

    # --- 7. the test splits, once ------------------------------------------
    with open(os.path.join(HERE, args.baseline)) as f:
        base_acc = json.load(f)["acc@5px"]
    best_name, (best_json, best_blob) = max(results.items(),
                                            key=lambda kv: kv[1][1]["acc@5px"])
    best_w = dict((n, w) for n, w, _ in jobs)[best_name]
    log(f"\nbest candidate: {best_name}  acc@5 {best_blob['acc@5px']:.4f} "
        f"vs baseline {base_acc:.4f}")

    if args.no_test:
        log("skipping test splits (--no-test)")
    elif best_blob["acc@5px"] <= base_acc:
        log("no candidate beats the shipped model on the large fresh set. "
            "Not touching the test splits -- they stay a measurement, not a "
            "selection. This is an acceptable outcome; the submission is "
            "unchanged and weights/driftsense.pt was never written.")
    else:
        log("candidate is ahead; measuring the three test splits ONCE")
        run([PY, "evaluate.py", "--splits", "data/test", "data/test_medium",
             "data/test_severe", "--weights", best_w, "--no-baseline",
             "--out", f"results/cmp_{tag}_test"])

    log("done")


if __name__ == "__main__":
    main()
