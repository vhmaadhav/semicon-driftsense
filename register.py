#!/usr/bin/env python3
"""Phase 2 batch entry point.

    python register.py --input pairs.csv --output predictions.csv

Writes one row per input pair, in input order, with the columns the Phase 2
contract names: pair_id, x, y, theta, scale, found, score.

Scale semantics, fixed by the Phase 2 task material (slide 5 + prompt section
2.3; see .agents/ORGANIZER_PHASE2_GROUND_TRUTH.md section 5): `scale` is the
recovered down-scaling factor z -- nominally in [8, 12], i.e. the search
image's nm/px -- NOT the reference-to-search linear factor 1/z (the two
readings differ by ~100x). theta is degrees, CCW positive as displayed,
about the match centre.

Two properties are treated as non-negotiable, because the scoring rules make
them expensive to get wrong:

* **Every pair_id appears exactly once.** A missing row scores zero, so a pair
  that raises, times out, or has unreadable images still emits a row -- a
  declined answer (`found=0`) rather than no answer.
* **No network, no downloads.** Weights load from the local `weights/`
  directory that ships inside the ZIP.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import infer as I  # noqa: E402
from driftsense.matching import locate_phase2  # noqa: E402

# The shipped Phase 2 operating point lives in driftsense.config (the ONE
# definition of the shipped decode config, so eval_ext.py and the parity tests
# consume the same value register.py does). Re-exported under the historical
# name for backwards compatibility -- every caller in this repo imports it
# from here.
from driftsense.config import SHIPPED_BAND, SHIPPED_THRESHOLD  # noqa: E402
from driftsense.config import SHIPPED_VERIFICATION  # noqa: E402
from driftsense.config import SHIPPED_SUBPIXEL_ROWS  # noqa: E402
from driftsense.config import LEGACY_FALLBACK_THRESHOLD  # noqa: E402,F401

DEFAULT_FOUND_THRESHOLD = SHIPPED_THRESHOLD

OUT_FIELDS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]

# Candidate spellings for the two image columns. The addendum fixes `pair_id`
# but publishes the rest of the pairs.csv layout separately, so accept the
# plausible spellings rather than guess one and fail the whole run.
REF_KEYS = ("reference", "reference_path", "ref", "ref_path", "reference_image",
            "template", "template_path", "high_res", "highres")
SEA_KEYS = ("search", "search_path", "sea", "search_image", "wide", "wide_path",
            "low_res", "lowres")


def cap_threads(requested=0):
    """Cap the torch and OpenCV thread pools to one sane value.

    The judge box is 4-core CPU-only. torch's intra-op default and OpenCV's
    pool both size themselves to every physical core, so without an explicit
    cap they oversubscribe 4 cores catastrophically (grader harness measured
    7.08 s/pair against 1.58 s in a tuned env on the same pairs). Default:
    min(4, os.cpu_count()); the --threads flag overrides for experiments.

    Restored after the PR #51 review: an interim revision defaulted to
    os.cpu_count(), which made a 10-core dev box measure a latency the 4-core
    reference machine can never reproduce -- the exact mismatch that produced
    the 7.08 s/pair surprise in the first place. Development and judging now
    share one default again.

    set_flush_denormal is x86-flavoured (avoids the denormal stalls of FP32
    near-zero activation outputs) and is best-effort: on platforms where it
    raises (e.g. ARM) we simply keep denormals.
    """
    n = requested if requested > 0 else min(4, os.cpu_count() or 4)
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_flush_denormal(True)
        except Exception:                        # noqa: BLE001
            pass
    except Exception:                            # noqa: BLE001
        pass
    try:
        import cv2
        cv2.setNumThreads(n)                     # noqa: N806 (cv2 spelling)
    except Exception:                            # noqa: BLE001
        pass
    return n


def pick_column(fieldnames, candidates, role):
    lowered = {f.lower().strip(): f for f in fieldnames}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    for f in fieldnames:                      # substring fallback
        if any(c.split("_")[0] in f.lower() for c in candidates):
            return f
    raise SystemExit(f"pairs.csv: could not find the {role} column among {fieldnames}")


# --------------------------------------------------------------------------
# Live terminal display
#
# Cosmetic only, and deliberately fenced off from everything the judge reads:
#
#   * predictions.csv is written by the same code either way -- the display
#     never touches a row, a value or a flush point.
#   * stdout keeps the contract lines the harness greps for ("wrote N rows
#     to <path>", the runtime summary) whether or not it is a terminal.
#   * the machine-readable per-pair records ("# per-pair seconds",
#     "# t,<pair_id>,<secs>") are NEVER lost, but they are not spammed at a
#     human either. When stderr is redirected -- every harness, every CI job,
#     every `2> log.txt` -- they go to stderr exactly as before. When stderr
#     is an interactive terminal they would just scroll the dashboard away,
#     so they are written to a sidecar `<output>.timing` file instead and the
#     summary card says where. An earlier revision simply dropped them on a
#     tty; that loses the audit trail for anything running under a pty.
#   * "# runtime: median X p90 Y max Z n=N" always goes to stderr. It is one
#     line, it is the summary a harness greps for, and it prints after the
#     dashboard is done.
#
# Everything below degrades to plain periodic lines when stdout is not a
# terminal, and every ANSI write is wrapped so a display failure can never
# take down a run.
# --------------------------------------------------------------------------

BOX_W = 78          # inner width of the header/summary cards

# Two ASCII mascots watch the run: a cat on the left, a panda on the right,
# passing a patch between them (which is, roughly, what the pipeline does).
# ASCII only -- no emoji, so the cards render the same in a pty, a CI log and
# a screenshot.
_CAT = ("  /\\_/\\  ", " ( o.o ) ", " ( -.- ) ")     # top, open eyes, blink
_PANDA = (" (@)_(@) ", " ( '~' ) ", " ( '-' ) ")     # top, chewing, closed
_SPIN = ("|", "/", "-", "\\")


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = int(seconds // 60), int(seconds % 60)
    return f"{m:02d}m {s:02d}s" if m else f"{s:02d}s"


class _LiveDisplay:
    """Three-line animated status block, redrawn in place on a terminal."""

    LINES = 3
    TRACK = 19          # width of the patch-travel track between the mascots

    def __init__(self, stream, total, quiet=False, animate=True):
        self.stream = stream
        self.total = max(1, int(total))
        self.tty = bool(getattr(stream, "isatty", lambda: False)()) and not quiet
        self.quiet = quiet
        self.animate = animate and self.tty
        self.frame = 0
        self.drawn = False
        self.state = {"n": 0, "pid": "", "dt": 0.0, "med": 0.0,
                      "elapsed": 0.0, "eta": 0.0, "found": 0}
        self._lock = __import__("threading").Lock()
        self._stop = None
        self._ticker = None

    # -- geometry ---------------------------------------------------------
    def _cols(self):
        try:
            import shutil
            return shutil.get_terminal_size((100, 24)).columns
        except Exception:                        # noqa: BLE001
            return 100

    def _card(self, title, rows):
        """A bordered card. Colour is applied to already-padded text, so the
        layout is identical with and without ANSI."""
        top = "+" + "-" * BOX_W + "+"
        edge = self._accent(top)
        pipe = self._accent("|")
        out = [edge, pipe + self._head(title.center(BOX_W)) + pipe, edge]
        for label, value in rows:
            text = f"  {label}: {value}"
            if len(text) > BOX_W:
                text = text[:BOX_W - 3] + "..."
            padded = text.ljust(BOX_W)
            if self._colour_ok():
                padded = padded.replace(f"{label}:", f"\033[1m{label}:\033[0m", 1)
            out.append(pipe + padded + pipe)
        out.append(edge)
        return "\n".join(out)

    # -- painting ---------------------------------------------------------
    def _scene(self):
        """The two animated mascot lines, or None when the terminal is narrow."""
        cols = self._cols()
        if cols < 96:
            return None
        f = self.frame
        cat = _CAT[2] if f % 11 == 0 else _CAT[1]
        panda = _PANDA[2] if f % 7 in (0, 1) else _PANDA[1]
        pos = f % (2 * self.TRACK)
        if pos >= self.TRACK:                    # ping-pong back to the cat
            pos = 2 * self.TRACK - pos - 1
        track = ["."] * self.TRACK
        track[pos] = "o"
        label = f"registering {_SPIN[f % len(_SPIN)]}"
        return (self._dim(f"  {_CAT[0]}  {''.join(track)}  {_PANDA[0]}"),
                self._dim(f"  {cat}  {label.center(self.TRACK)}  {panda}"))

    # -- colour -----------------------------------------------------------
    def _colour_ok(self):
        """ANSI colour only on a terminal that has not opted out."""
        return self.tty and not os.environ.get("NO_COLOR")

    def _dim(self, text):
        return f"\033[2m{text}\033[0m" if self._colour_ok() else text

    def _accent(self, text):
        return f"\033[36m{text}\033[0m" if self._colour_ok() else text

    def _c(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self._colour_ok() else text

    def _head(self, text):
        return f"\033[1;36m{text}\033[0m" if self._colour_ok() else text

    def _bar(self):
        s = self.state
        cols, n = self._cols(), s["n"]
        pct = 100.0 * n / self.total
        width = 26 if cols >= 110 else (16 if cols >= 90 else 10)
        filled = int(round(width * n / self.total))
        # Built as (plain, coloured) pairs so the truncation below counts
        # visible characters and never slices an escape sequence in half.
        def stat(label, value, colour):
            """One '  label value' segment as (plain, painted).

            The label stays dim and the value carries the colour, so the eye
            lands on the numbers rather than on the words between them.
            """
            plain = f"  {label} {value}"
            return plain, f"  {self._dim(label)} {self._c(value, colour)}"

        parts = [
            ("  [", "  ["),
            ("#" * filled, self._c("#" * filled, "1;32")),
            ("-" * (width - filled), self._dim("-" * (width - filled))),
            ("] ", self._accent("] ")),
            (f"{pct:5.1f}%", self._c(f"{pct:5.1f}%", "1;97")),
            *[stat(*a) for a in (
                ("pair", f"{n}/{self.total}", "1;36"),
                ("found", f"{s['found']}", "1;32"),
                ("med", f"{s['med']:.2f}s", "1;35"),
                ("elapsed", _fmt_time(s["elapsed"]), "1;33"),
                ("eta", _fmt_time(s["eta"]), "1;34"),
            )],
        ]
        if cols >= 122 and s["pid"]:
            parts.append(stat("last", f"{s['pid']} {s['dt']:.2f}s", "0;36"))
        out, visible, budget = [], 0, max(20, cols - 1)
        for plain, painted in parts:
            if visible + len(plain) > budget:
                out.append(plain[:budget - visible])
                break
            out.append(painted if self._colour_ok() else plain)
            visible += len(plain)
        return "".join(out)

    def _write(self, text):
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:                        # noqa: BLE001
            self.tty = self.animate = False

    def _paint(self):
        """Draw (or redraw in place) the status block. Caller holds the lock."""
        if not self.tty:
            return
        scene = self._scene()
        lines = list(scene) if scene else ["", ""]
        lines.append(self._bar())
        buf = "\033[%dA" % self.LINES if self.drawn else ""
        buf += "".join(f"\r\033[K{ln}\n" for ln in lines)
        self._write(buf)
        self.drawn = True

    # -- public API -------------------------------------------------------
    def header(self, rows):
        if self.quiet:
            return
        self._write(self._card("DRIFT-SENSE  PHASE 2  REGISTRATION", rows) + "\n\n")

    def update(self, **state):
        self.state.update(state)
        with self._lock:
            self.frame += 1
            self._paint()

    def erase(self):
        """Clear the block so a log line can scroll above it."""
        with self._lock:
            if self.tty and self.drawn:
                self._write("\033[%dA" % self.LINES + "\r\033[K\n" * self.LINES
                            + "\033[%dA" % self.LINES)
                self.drawn = False

    def start(self):
        """Animate between pairs -- a pair takes ~1.5 s, so without this the
        mascots would only move once per pair. Daemon thread, display-only."""
        if not self.animate:
            return
        import threading
        self._stop = threading.Event()

        def tick():
            while not self._stop.wait(0.18):
                with self._lock:
                    if self.drawn:
                        self.frame += 1
                        self._paint()

        self._ticker = threading.Thread(target=tick, daemon=True)
        self._ticker.start()

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=1.0)
        self.erase()

    def summary(self, rows):
        if self.quiet:
            return
        self._write("\n" + self._card("RUN COMPLETE", rows) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pairs.csv")
    ap.add_argument("--output", required=True, help="predictions.csv")
    ap.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_FOUND_THRESHOLD,
                    help="confidence at or above which a pair is reported found. "
                         "Applies to the LEARNED path only: the statistic and the "
                         "threshold are one unit system, so when the weights cannot "
                         "load and the ZNCC fallback runs, this value is ignored and "
                         "the fallback uses its own calibrated "
                         "LEGACY_FALLBACK_THRESHOLD instead -- a fused6 threshold "
                         "applied to a raw NCC would decide nothing meaningful")
    ap.add_argument("--verification", default=SHIPPED_VERIFICATION,
                    help="hypothesis selector: zncc (default) | consensus | majority. "
                         "consensus overrides the native-ZNCC winner only when the rank "
                         "and band scores pick the same different hypothesis; it was "
                         "measured +2/0 and +1/0 rescued/broken on the PR #3 proxy; "
                         "full 2,250-pair A/B (issue #9): +0.11 total, paired CI "
                         "spans zero, 5 broken / 6 rescued -- real but under the "
                         "promotion gate, so zncc stays the default")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch/OpenCV thread cap. 0 (default) auto-caps to "
                         "min(4, CPU cores) to match the 4-core reference "
                         "machine -- it does NOT leave the library defaults, "
                         "because an uncapped pool oversubscribes the grader's "
                         "box and inflates every per-pair time. Pass a positive "
                         "value to override.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    # Thread sanity before any heavy work (model load touches torch and
    # conv2d; the coarse sweep touches cv2). --threads overrides the default.
    avail_cores = os.cpu_count() or 1
    active_threads = cap_threads(a.threads)

    with open(a.input, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{a.input}: no rows")

    fields = list(rows[0].keys())
    id_col = pick_column(fields, ("pair_id", "id", "pair"), "pair_id")
    ref_col = pick_column(fields, REF_KEYS, "reference")
    sea_col = pick_column(fields, SEA_KEYS, "search")
    base = os.path.dirname(os.path.abspath(a.input))

    def resolve(p):
        p = (p or "").strip()
        return p if os.path.isabs(p) else os.path.join(base, p)

    model, device = I.load_model(a.weights) or (None, None)

    # The judge may name an output path in a directory that does not exist
    # yet; creating it here is the difference between a run and a crash.
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)

    times = []
    t_start = time.perf_counter()
    found_count = 0
    total_rows = len(rows)

    disp = _LiveDisplay(sys.stdout, total_rows, quiet=a.quiet)
    disp.header([
        ("Pairs", f"{total_rows} from {os.path.basename(a.input)}"),
        ("Decode", f"{'learned + ZNCC verify' if model is not None else 'ZNCC fallback (no weights)'}"
                   f", threshold {a.threshold if model is not None else LEGACY_FALLBACK_THRESHOLD}"),
        ("Threads", f"{active_threads} (of {avail_cores} cores detected)"
                    f"{'' if a.threads else ' -- default min(4, cores)'}"),
        ("Output", os.path.abspath(a.output)),
    ])

    # Per-pair timing metadata never touches stdout: stdout stays the human
    # progress stream and the predictions file stays byte-identical. It goes
    # to stderr when stderr is redirected (the machine-consumption case), and
    # to a sidecar file when stderr is a terminal, where it would otherwise
    # scroll the dashboard away one line per pair.
    stderr_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    timing_path = (a.output + ".timing") if stderr_tty else None
    timing_fh = None
    if timing_path:
        try:
            timing_fh = open(timing_path, "w")
        except OSError:                          # noqa: BLE001
            timing_path = None                   # unwritable: fall back below
    trace = timing_fh if timing_fh is not None else sys.stderr
    print("# per-pair seconds", file=trace)
    disp.start()

    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for n, r in enumerate(rows):
            pid = r[id_col]
            # Declined answer, overwritten below on success. Constructed first
            # so that any failure path still has a complete row to write.
            out = {"pair_id": pid, "x": 0, "y": 0, "theta": 0, "scale": 0,
                   "found": 0, "score": 0.0}
            t0 = time.perf_counter()
            try:
                ref = I.read_gray(resolve(r[ref_col]))
                sea = I.read_gray(resolve(r[sea_col]))
                if model is None:
                    res = I.zncc_fallback(ref, sea)
                    res.setdefault("scale", 10.0)
                    res.setdefault("theta", 0.0)
                    # The fallback's score is raw ZNCC from a single template
                    # sweep, not the learned path's confidence statistic, so it
                    # gates at driftsense.config.LEGACY_FALLBACK_THRESHOLD
                    # rather than at --threshold. Only reachable when the
                    # weights/torch are unavailable -- on the grader box they
                    # ship inside the ZIP, so this is a degraded-mode guard.
                    threshold = LEGACY_FALLBACK_THRESHOLD
                else:
                    threshold = a.threshold
                    # band=False: the difference-of-Gaussians pre-filter on
                    # the coarse sweep costs points on both architectures.
                    # Measured independently here at +0.439 (95% CI
                    # [+0.132, +0.767], P 99.8%) on the 0.456M model and +0.509
                    # (P 94.2%) on the shipped 1.02M one; PR #18 reached the
                    # same conclusion separately. The value is the shipped
                    # decode config (driftsense.config), shared with
                    # eval_ext.py so the evaluator decodes identically.
                    res = locate_phase2(model, ref, sea, device, refine=True,
                                        verification=a.verification,
                                        band=SHIPPED_BAND,
                                        subpixel_rows=SHIPPED_SUBPIXEL_ROWS)
                # The reported confidence (see locate_phase2): the shipped
                # legacy min(network score, native ZNCC) on the model path,
                # raw ZNCC on the fallback path.
                score = float(res.get("confidence", res.get("score", 0.0)))
                found = int(score >= threshold)
                out.update({
                    "x": f'{float(res["x"]):.4f}' if found else 0,
                    "y": f'{float(res["y"]):.4f}' if found else 0,
                    "theta": f'{float(res.get("theta", 0.0)):.4f}' if found else 0,
                    "scale": f'{float(res.get("scale", 10.0)):.4f}' if found else 0,
                    "found": found,
                    "score": f"{score:.6f}",
                })
            # Never let one bad pair cost the rest of the run, and never
            # drop the row. SystemExit is caught too: read_gray raises
            # SystemExit for an unreadable image, and that must zero-fill
            # THIS row only -- not kill the whole batch.
            except Exception as e:                      # noqa: BLE001
                disp.erase()
                print(f"[warn] pair {pid}: {type(e).__name__}: {e}", file=sys.stderr)
            except SystemExit as e:
                disp.erase()
                print(f"[warn] pair {pid}: SystemExit: {e}", file=sys.stderr)
            w.writerow(out)
            if out.get("found"):
                found_count += 1
            dt = time.perf_counter() - t0
            times.append(dt)

            # Machine-readable per-pair record, always: it is the audit trail
            # the harness parses. Only erase the live block when the record is
            # actually going to the terminal -- writing to the sidecar must
            # leave the dashboard alone.
            if trace is sys.stderr:
                disp.erase()
            # flush per pair: the audit trail must survive a kill, and a
            # sidecar file is block-buffered where stderr was not.
            print(f"# t,{pid},{dt:.3f}", file=trace, flush=True)

            cur_n = n + 1
            elapsed = time.perf_counter() - t_start
            rate = cur_n / elapsed if elapsed > 0 else 0.0
            if not a.quiet:
                med = float(np.median(times))
                disp.update(n=cur_n, pid=str(pid), dt=dt, med=med,
                            elapsed=elapsed, found=found_count,
                            eta=(total_rows - cur_n) / rate if rate > 0 else 0.0)
                if not disp.tty and (cur_n % 25 == 0 or cur_n == total_rows):
                    print(f"  {cur_n}/{total_rows}  median {med:.2f}s  "
                          f"elapsed {_fmt_time(elapsed)}  "
                          f"eta {_fmt_time((total_rows - cur_n) / rate if rate else 0)}",
                          flush=True)
                f.flush()

    disp.stop()
    if timing_fh is not None:
        timing_fh.close()
    t_total = time.perf_counter() - t_start
    t = np.array(times)
    if not a.quiet:
        # The two contract lines the harness greps for. They are printed on
        # every path -- terminal or pipe, animated or not.
        print(f"wrote {len(rows)} rows to {a.output}")
        print(f"runtime: median {np.median(t):.2f}s  p90 {np.percentile(t,90):.2f}s  "
              f"max {t.max():.2f}s  total {t.sum()/60:.1f} min")
        disp.summary([
            ("Pairs", f"{total_rows} in {_fmt_time(t_total)} wall "
                      f"({total_rows / t_total:.2f} pairs/s)"),
            ("Latency", f"median {np.median(t):.2f}s  mean {np.mean(t):.2f}s  "
                        f"p90 {np.percentile(t, 90):.2f}s  max {t.max():.2f}s"),
            ("Reported found", f"{found_count}/{total_rows} "
                               f"({100.0 * found_count / total_rows:.1f}%)"),
            ("Predictions", os.path.abspath(a.output)),
        ] + ([("Per-pair timings", os.path.abspath(timing_path))]
             if timing_path else []))
        if t.max() > 20:
            print(f"WARNING: {int((t>20).sum())} pair(s) exceeded the 20 s hard timeout",
                  file=sys.stderr)

    # Machine-readable runtime summary: stderr, same numbers as the stdout
    # line, for the judge harness to parse without scraping progress text.
    print(f"# runtime: median {np.median(t):.2f} p90 {np.percentile(t,90):.2f} "
          f"max {t.max():.2f} n={len(t)}", file=sys.stderr)


if __name__ == "__main__":
    main()
