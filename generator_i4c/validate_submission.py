#!/usr/bin/env python3
"""
Organizer-side first-pass QA for student-submitted Drift-Sense datasets.

Reads a submission's manifest.csv and runs cheap per-sample sanity checks,
then renders low-res "contact sheet" images tiling every sample (reference +
GT-annotated search thumbnail, color-coded by status) so an organizer can
eyeball an entire submission in seconds before any deeper manual or
automated grading.

Manifest columns are flexible:
  - reference image: `reference_path` or `ref_path`
  - search image: `search_path`
  - ground truth: either an explicit box (`gt_box_x/y/w/h`) or just a center
    point (`gt_x`, `gt_y`) -- a point gets a `--gt-box-size` (default 100px)
    square derived around it, matching this repo's own 1000px/10x-downsample
    convention.
  - `script_path` (optional): path to the script that generated that sample.
    Checked for existence by default; pass --run-scripts to actually execute
    it (see the big warning on that flag -- it's untrusted code execution).

Checks (all cheap -- no template matching, no model inference):
  - files exist and load
  - image size matches expected (default 1000x1000)
  - not blank / not saturated (std-dev + extreme-pixel-fraction thresholds)
  - looks grayscale (SEM images are single-channel; flags accidental RGB
    content that isn't just a duplicated-channel grayscale save)
  - ground-truth box lies within the search image bounds
  - ground-truth patch consistency: crop the search image at gt_box and
    compare it (mean abs pixel diff) against the reference resized to the
    same size -- the one check that validates the *claim* itself, not just
    file formatting. Tries a small set of candidate rotation angles on the
    reference first (Phase 2's reference_rotation_deg means it may have
    been captured at an unknown small angle) and keeps the best match.
  - script_path exists / is non-empty (and, with --run-scripts, exits 0)

Usage:
    # one submission with a manifest.csv (reference_path/search_path/gt_box_*)
    python validate_submission.py --manifest output/train/manifest.csv

    # every submission under a root (recursively finds each manifest.csv)
    python validate_submission.py --root ./submissions --out-dir ./review

    # flat folder, no manifest: data/sample_0_ref.png + data/sample_0_search.png
    python validate_submission.py --data-dir data --out-dir ./review

    # same, with a separate ground-truth CSV to enable the GT-consistency check
    # (columns: id, gt_box_x, gt_box_y, gt_box_w, gt_box_h -- id must match the
    # sample_N part of the filenames)
    python validate_submission.py --data-dir data --gt-csv data/ground_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPECTED_SIZE = (1000, 1000)  # (width, height)
MIN_STD = 3.0                        # near-blank / flat image
MAX_EXTREME_FRACTION = 0.98          # fraction of pixels pinned at 0 or 255
GT_PATCH_MAX_MEAN_ABS_DIFF = 40.0    # generous vs. the 30.0 used in tests/,
                                      # since we don't know a student's noise settings
COLOR_CHANNEL_DIFF_THRESHOLD = 6.0   # mean abs diff between B/G/R channels

# The generator can rotate the captured Reference by up to ~10-15deg
# (Phase 2's reference_rotation_deg) relative to the Search, and the
# manifest doesn't carry the actual angle used. Rather than assume no
# rotation (which would flag correctly-generated rotated samples as FAIL),
# try a small set of candidate angles and keep the best match -- mirrors
# how the ZNCC baseline already searches a few candidate scales instead of
# assuming the ratio is exactly known.
GT_ROTATION_CANDIDATES_DEG = [0, -15, -10, -5, 5, 10, 15]

STATUS_ORDER = {"OK": 0, "WARN": 1, "FAIL": 2}
STATUS_COLOR = {"OK": (90, 200, 120), "WARN": (230, 180, 60), "FAIL": (225, 80, 80)}


@dataclass
class SampleResult:
    row_id: str
    reference_path: str
    search_path: str
    gt_box: tuple | None = None
    script_path: str | None = None
    status: str = "OK"
    reasons: list = field(default_factory=list)

    def flag(self, level: str, reason: str):
        if STATUS_ORDER[level] > STATUS_ORDER[self.status]:
            self.status = level
        self.reasons.append(reason)


def _resolve(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_gray(path: str):
    """Returns (grayscale ndarray or None, is_real_color bool)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, False
    if img.ndim == 2:
        return img, False
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    channel_diff = (np.abs(b - g).mean() + np.abs(g - r).mean() + np.abs(b - r).mean()) / 3.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, channel_diff > COLOR_CHANNEL_DIFF_THRESHOLD


def _first(row: dict, *keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


def _best_rotated_patch_diff(ref: np.ndarray, patch: np.ndarray, wi: int, hi: int) -> float:
    """Mean abs diff between `patch` and `ref` (downsampled to wi x hi),
    minimized over GT_ROTATION_CANDIDATES_DEG -- tolerant of the Reference
    having been captured at an unknown small rotation relative to Search."""
    h, w = ref.shape[:2]
    center = (w / 2.0, h / 2.0)
    best = None
    for angle in GT_ROTATION_CANDIDATES_DEG:
        if angle == 0:
            candidate = ref
        else:
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            candidate = cv2.warpAffine(ref, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ref_down = cv2.resize(candidate, (wi, hi), interpolation=cv2.INTER_AREA)
        diff = float(np.abs(patch.astype(int) - ref_down.astype(int)).mean())
        if best is None or diff < best:
            best = diff
    return best


def _resolve_gt_box(row: dict, box_size: float):
    """Accept either an explicit box (gt_box_x/y/w/h) or a center point
    (gt_x/gt_y), deriving a box_size x box_size box around the point when
    only a point is given -- matches this repo's own convention (100x100 at
    the standard 1000px/10x-downsample calibration)."""
    box_fields = ["gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h"]
    if all(row.get(f) not in (None, "") for f in box_fields):
        return tuple(float(row[f]) for f in box_fields)

    gt_x, gt_y = _first(row, "gt_x"), _first(row, "gt_y")
    if gt_x is not None and gt_y is not None:
        cx, cy = float(gt_x), float(gt_y)
        return (cx - box_size / 2, cy - box_size / 2, box_size, box_size)

    return None


def check_sample(row: dict, base_dir: str, gt_box_size: float = 100.0,
                  run_scripts: bool = False, script_timeout: int = 60) -> SampleResult:
    ref_raw = _first(row, "reference_path", "ref_path")
    search_raw = _first(row, "search_path")
    if ref_raw is None or search_raw is None:
        result = SampleResult(row_id=str(row.get("id", "?")), reference_path="", search_path="")
        result.flag("FAIL", "row missing a reference/ref_path or search_path column")
        return result

    ref_path = _resolve(ref_raw, base_dir)
    search_path = _resolve(search_raw, base_dir)
    result = SampleResult(row_id=str(row.get("id", "?")), reference_path=ref_path, search_path=search_path)

    ref, ref_is_color = load_gray(ref_path)
    search, search_is_color = load_gray(search_path)

    if ref is None:
        result.flag("FAIL", f"reference not found/readable: {ref_raw}")
    if search is None:
        result.flag("FAIL", f"search not found/readable: {search_raw}")
    if ref is None or search is None:
        return result

    if (ref.shape[1], ref.shape[0]) != EXPECTED_SIZE:
        result.flag("WARN", f"reference size {ref.shape[1]}x{ref.shape[0]} != expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")
    if (search.shape[1], search.shape[0]) != EXPECTED_SIZE:
        result.flag("WARN", f"search size {search.shape[1]}x{search.shape[0]} != expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}")

    for name, img in (("reference", ref), ("search", search)):
        std = float(img.std())
        if std < MIN_STD:
            result.flag("FAIL", f"{name} looks blank (std={std:.2f})")
        extreme_frac = float(np.mean((img <= 1) | (img >= 254)))
        if extreme_frac > MAX_EXTREME_FRACTION:
            result.flag("FAIL", f"{name} looks saturated ({extreme_frac * 100:.0f}% pixels at 0/255)")

    if ref_is_color:
        result.flag("WARN", "reference is a real color image, not grayscale")
    if search_is_color:
        result.flag("WARN", "search is a real color image, not grayscale")

    match_found_raw = _first(row, "match_found")
    is_intentional_no_match = match_found_raw is not None and str(match_found_raw).strip().lower() in ("false", "0", "no")

    gt_box = _resolve_gt_box(row, gt_box_size)
    if gt_box is not None:
        x0, y0, w, h = gt_box
        result.gt_box = gt_box
        sh, sw = search.shape[:2]
        if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0 + w > sw or y0 + h > sh:
            result.flag("FAIL", f"gt_box ({x0:.0f},{y0:.0f},{w:.0f},{h:.0f}) out of search bounds ({sw}x{sh})")
        else:
            x0i, y0i, wi, hi = int(round(x0)), int(round(y0)), int(round(w)), int(round(h))
            patch = search[y0i:y0i + hi, x0i:x0i + wi]
            diff = _best_rotated_patch_diff(ref, patch, wi, hi)
            if diff > GT_PATCH_MAX_MEAN_ABS_DIFF:
                result.flag("FAIL", f"search patch at gt_box doesn't resemble reference (mean abs diff={diff:.1f})")
    elif is_intentional_no_match:
        pass  # match_found=False row with no gt_box -- expected, not a data problem
    else:
        result.flag("WARN", "manifest missing gt_box_x/y/w/h or gt_x/gt_y -- skipped ground-truth consistency check")

    script_raw = _first(row, "script_path")
    if script_raw is not None:
        script_path = _resolve(script_raw, base_dir)
        result.script_path = script_path
        if not os.path.isfile(script_path):
            result.flag("FAIL", f"script_path not found: {script_raw}")
        elif os.path.getsize(script_path) == 0:
            result.flag("WARN", f"script_path is an empty file: {script_raw}")
        elif run_scripts:
            ok, msg = _run_script(script_path, script_timeout)
            if not ok:
                result.flag("WARN", f"script did not run cleanly: {msg}")

    return result


def _run_script(script_path: str, timeout: int) -> tuple:
    """Best-effort execution of a student's generation script, sandboxed only
    by a subprocess + timeout -- NOT a real sandbox. Only run this (--run-scripts)
    against submissions you're prepared to treat as untrusted code, ideally in
    a container/VM. There's no standard output contract to check the script's
    images against yet, so this only confirms the script runs without error;
    it does not (currently) verify it reproduces reference_path/search_path.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["python3", script_path],
            cwd=os.path.dirname(script_path) or ".",
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as e:
        return False, str(e)
    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no stderr"
        return False, f"exit code {proc.returncode}: {stderr_tail}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Contact sheet rendering
# ---------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_tile(result: SampleResult, thumb_size: int) -> Image.Image:
    label_h = 34
    border = 3
    pad = 4
    tile_w = thumb_size * 2 + pad * 3
    tile_h = thumb_size + label_h + pad * 2

    canvas = Image.new("RGB", (tile_w + border * 2, tile_h + border * 2), STATUS_COLOR[result.status])
    inner = Image.new("RGB", (tile_w, tile_h), (18, 18, 20))

    ref = cv2.imread(result.reference_path, cv2.IMREAD_GRAYSCALE) if result.reference_path else None
    search = cv2.imread(result.search_path, cv2.IMREAD_GRAYSCALE) if result.search_path else None

    def to_thumb(img):
        if img is None:
            blank = Image.new("L", (thumb_size, thumb_size), 40)
            d = ImageDraw.Draw(blank)
            d.line((0, 0, thumb_size, thumb_size), fill=120, width=2)
            d.line((0, thumb_size, thumb_size, 0), fill=120, width=2)
            return blank.convert("RGB")
        resized = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized).convert("RGB")

    ref_thumb = to_thumb(ref)
    search_thumb = to_thumb(search)

    if search is not None and result.gt_box is not None:
        sh, sw = search.shape[:2]
        sx = thumb_size / sw
        sy = thumb_size / sh
        x0, y0, w, h = result.gt_box
        d = ImageDraw.Draw(search_thumb)
        d.rectangle(
            [x0 * sx, y0 * sy, (x0 + w) * sx, (y0 + h) * sy],
            outline=(255, 90, 90), width=2,
        )

    inner.paste(ref_thumb, (pad, pad))
    inner.paste(search_thumb, (pad * 2 + thumb_size, pad))

    d = ImageDraw.Draw(inner)
    font = _font(13)
    label = f"#{result.row_id} {result.status}"
    d.text((pad, thumb_size + pad + 2), label, fill=(230, 230, 230), font=font)
    if result.reasons:
        reason_line = result.reasons[0]
        if len(reason_line) > 56:
            reason_line = reason_line[:53] + "..."
        d.text((pad, thumb_size + pad + 17), reason_line, fill=(170, 170, 170), font=font)

    canvas.paste(inner, (border, border))
    return canvas


def build_contact_sheets(results: list, out_dir: str, name: str, thumb_size: int, cols: int, per_sheet: int):
    os.makedirs(out_dir, exist_ok=True)
    tiles = [render_tile(r, thumb_size) for r in results]
    if not tiles:
        return []

    tile_w, tile_h = tiles[0].size
    paths = []
    for page_start in range(0, len(tiles), per_sheet):
        page_tiles = tiles[page_start:page_start + per_sheet]
        rows = (len(page_tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (tile_w * cols, tile_h * rows), (8, 8, 10))
        for i, tile in enumerate(page_tiles):
            r, c = divmod(i, cols)
            sheet.paste(tile, (c * tile_w, r * tile_h))
        page_idx = page_start // per_sheet
        out_path = os.path.join(out_dir, f"{name}_contact_sheet_{page_idx:02d}.png")
        sheet.save(out_path)
        paths.append(out_path)
    return paths


def write_report_csv(results: list, out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "status", "reasons", "reference_path", "search_path", "script_path"])
        for r in results:
            writer.writerow([r.row_id, r.status, " | ".join(r.reasons), r.reference_path, r.search_path, r.script_path or ""])


def discover_pairs(data_dir: str, ref_suffix: str, search_suffix: str) -> tuple[list, list]:
    """Pair up files in a flat folder by filename, e.g.
    sample_0_ref.png + sample_0_search.png -> id "sample_0".

    Returns (rows, unmatched) where rows is a list of dicts shaped like a
    manifest row (id/reference_path/search_path, no gt_box_* unless merged
    in separately), and unmatched is a list of files that had no partner --
    a real failure mode with this layout that a manifest can't have.
    """
    files = sorted(os.listdir(data_dir))
    ref_by_id = {}
    search_by_id = {}

    for fname in files:
        stem, ext = os.path.splitext(fname)
        if not ext.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            continue
        if stem.endswith(ref_suffix):
            ref_by_id[stem[: -len(ref_suffix)]] = fname
        elif stem.endswith(search_suffix):
            search_by_id[stem[: -len(search_suffix)]] = fname

    ids = sorted(set(ref_by_id) | set(search_by_id))
    rows, unmatched = [], []
    for sample_id in ids:
        ref_fname = ref_by_id.get(sample_id)
        search_fname = search_by_id.get(sample_id)
        if ref_fname is None:
            unmatched.append({
                "id": sample_id, "reference_path": "", "search_path": search_fname,
                "reason": f"search file '{search_fname}' has no matching '*{ref_suffix}' file",
            })
            continue
        if search_fname is None:
            unmatched.append({
                "id": sample_id, "reference_path": ref_fname, "search_path": "",
                "reason": f"reference file '{ref_fname}' has no matching '*{search_suffix}' file",
            })
            continue
        rows.append({"id": sample_id, "reference_path": ref_fname, "search_path": search_fname})

    return rows, unmatched


def load_gt_csv(gt_csv_path: str) -> dict:
    """id -> dict of gt_box_x/y/w/h (and anything else present), for merging
    into rows discovered by discover_pairs()."""
    with open(gt_csv_path, newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def validate_rows(rows: list, unmatched: list, base_dir: str, name: str, out_dir: str, opts: dict) -> dict:
    thumb_size, cols, per_sheet = opts["thumb_size"], opts["cols"], opts["per_sheet"]
    max_samples, seed = opts["max_samples"], opts["seed"]

    if max_samples is not None and len(rows) > max_samples:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_samples]

    results = [
        check_sample(row, base_dir, opts["gt_box_size"], opts["run_scripts"], opts["script_timeout"])
        for row in rows
    ]
    for u in unmatched:
        results.append(SampleResult(
            row_id=u["id"],
            reference_path=_resolve(u["reference_path"], base_dir) if u["reference_path"] else "",
            search_path=_resolve(u["search_path"], base_dir) if u["search_path"] else "",
            status="FAIL", reasons=[u["reason"]],
        ))

    sheet_paths = build_contact_sheets(results, out_dir, name, thumb_size, cols, per_sheet)
    report_path = os.path.join(out_dir, f"{name}_report.csv")
    write_report_csv(results, report_path)

    counts = {"OK": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r.status] += 1

    print(f"[{name}] {len(results)} samples -> OK={counts['OK']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
    if unmatched:
        print(f"  {len(unmatched)} unmatched file(s):")
        for u in unmatched:
            print(f"    {u['id']}: {u['reason']}")
    for p in sheet_paths:
        print(f"  wrote {p}")
    print(f"  wrote {report_path}")

    return {"name": name, "source": base_dir, "total": len(results), **counts}


def validate_manifest(manifest_path: str, out_dir: str, opts: dict) -> dict:
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    name = os.path.basename(os.path.dirname(manifest_path)) or "submission"
    return validate_rows(rows, [], base_dir, name, out_dir, opts)


def validate_data_dir(data_dir: str, ref_suffix: str, search_suffix: str, gt_csv: str | None,
                       out_dir: str, opts: dict) -> dict:
    rows, unmatched = discover_pairs(data_dir, ref_suffix, search_suffix)
    if gt_csv:
        gt_by_id = load_gt_csv(gt_csv)
        for row in rows:
            row.update(gt_by_id.get(row["id"], {}))
    name = os.path.basename(os.path.abspath(data_dir)) or "submission"
    return validate_rows(rows, unmatched, data_dir, name, out_dir, opts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", help="path to a single submission's manifest.csv")
    p.add_argument("--root", help="root directory to search recursively for manifest.csv files (one per submission)")
    p.add_argument("--data-dir", help="flat folder of paired images, no manifest (e.g. sample_0_ref.png + sample_0_search.png)")
    p.add_argument("--ref-suffix", default="_ref", help="filename suffix (before extension) marking a reference image, for --data-dir")
    p.add_argument("--search-suffix", default="_search", help="filename suffix (before extension) marking a search image, for --data-dir")
    p.add_argument("--gt-csv", default=None, help="optional CSV with columns id,gt_box_x,gt_box_y,gt_box_w,gt_box_h to enable the GT-consistency check under --data-dir")
    p.add_argument("--out-dir", default="./review", help="where to write contact sheets + CSV reports")
    p.add_argument("--thumb-size", type=int, default=140, help="px per thumbnail (reference/search each)")
    p.add_argument("--cols", type=int, default=8, help="tiles per row")
    p.add_argument("--per-sheet", type=int, default=96, help="max tiles per contact-sheet image before paginating")
    p.add_argument("--max-samples", type=int, default=None, help="randomly subsample a submission if it has more rows than this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gt-box-size", type=float, default=100.0, help="box side length to derive around a gt_x/gt_y center point when the manifest has no explicit gt_box_w/h")
    p.add_argument("--run-scripts", action="store_true",
                    help="actually execute each row's script_path (python3 script_path) to confirm it runs cleanly. "
                         "UNTRUSTED CODE EXECUTION -- only use this against submissions you're prepared to sandbox "
                         "(container/VM), and note it only checks the script exits 0, not that it reproduces the images.")
    p.add_argument("--script-timeout", type=int, default=60, help="seconds before a --run-scripts execution is killed")
    args = p.parse_args()

    modes = [bool(args.manifest), bool(args.root), bool(args.data_dir)]
    if sum(modes) != 1:
        p.error("pass exactly one of --manifest, --root, or --data-dir")

    if args.run_scripts:
        print("WARNING: --run-scripts executes student-submitted code on this machine. "
              "Make sure you're running this in a sandbox/container, not your main machine.\n")

    opts = dict(
        thumb_size=args.thumb_size, cols=args.cols, per_sheet=args.per_sheet,
        max_samples=args.max_samples, seed=args.seed, gt_box_size=args.gt_box_size,
        run_scripts=args.run_scripts, script_timeout=args.script_timeout,
    )

    if args.data_dir:
        summary = [validate_data_dir(args.data_dir, args.ref_suffix, args.search_suffix, args.gt_csv, args.out_dir, opts)]
    else:
        manifests = [args.manifest] if args.manifest else sorted(glob.glob(os.path.join(args.root, "**", "manifest.csv"), recursive=True))
        if not manifests:
            p.error(f"no manifest.csv found under {args.root}")
        summary = [validate_manifest(m, args.out_dir, opts) for m in manifests]

    if len(summary) > 1:
        summary_path = os.path.join(args.out_dir, "_all_submissions_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "source", "total", "OK", "WARN", "FAIL"])
            writer.writeheader()
            for row in summary:
                writer.writerow(row)
        print(f"\nwrote {summary_path}")
        total_fail = sum(s["FAIL"] for s in summary)
        print(f"{len(summary)} submissions, {total_fail} samples with FAIL across all of them")


if __name__ == "__main__":
    main()
