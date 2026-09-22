"""Phase 3 ``pairs.csv`` reader (``driftsense/pairs3.py``) and the column
resolution it exists to make safe.

Two things are pinned here, and they are different in kind:

1. **The reader's own contract** -- the six-column Phase 3 layout parses, the
   blind split's withheld fields are genuinely unreachable, aliases still
   work, and a bad header raises something actionable.

2. **The regression that motivated the module.** ``register.py`` resolved its
   image columns by substring fallback. On the Phase 3 header that fallback
   picked ``reference_gds_path`` for the "reference" role, which reached
   ``cv2.imread``, returned ``None``, raised ``SystemExit`` inside
   ``read_gray``, and was swallowed by the per-pair handler into a *declined
   row* -- a well-formed, exit-0, all-declined ``predictions.csv``. These
   tests assert the ambiguity is now an error, and that every real Phase 1/2
   header still resolves, so the fix cannot quietly break scoring.
"""

import csv
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.pairs3 import (  # noqa: E402
    PHASE3_FIELDS,
    WITHHELD_FIELDS,
    PairsRow,
    PairsSchemaError,
    inference_fields,
    read_pairs,
    resolve_column,
    resolve_schema,
)


def _load_register():
    """Import register.py without executing its CLI.

    register.py puts its own directory on sys.path and imports infer, so it is
    loaded the same way test_register_runtime_meta.py reaches it: by path, as
    a module, never as a subprocess, because pick_column is what is under test.
    """
    spec = importlib.util.spec_from_file_location(
        "register_under_test", os.path.join(REPO_ROOT, "register.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# 1. The reader's contract
# --------------------------------------------------------------------------

TRAIN_HEADER = list(PHASE3_FIELDS)
TRAIN_ROW = {
    "pair_id": "p001",
    "search_path": "search/p001.png",
    "reference_gds_path": "reference/p001.gds",
    "search_gds_path": "search/p001.gds",
    "reference_sem_path": "reference_sem/p001.png",
    "params_json_path": "params/p001.json",
}


def _write(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(path)


def test_training_split_reads_all_six_columns(tmp_path):
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [TRAIN_ROW])
    rows = read_pairs(p)
    assert len(rows) == 1
    r = rows[0]
    assert r.pair_id == "p001"
    assert r.reference_gds_path.endswith("reference/p001.gds")
    assert r.search_gds_path.endswith("search/p001.gds")
    assert r.present_in_training is True
    assert not r.is_blind


def test_blind_split_reads_and_loses_the_withheld_columns(tmp_path):
    """The blind split's last two fields are empty; everything else is equal."""
    row = dict(TRAIN_ROW, reference_sem_path="", params_json_path="")
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [row])
    r = read_pairs(p)[0]
    assert r.is_blind and not r.present_in_training
    # The half that must keep working.
    assert r.pair_id == "p001"
    assert r.reference_gds_path.endswith("reference/p001.gds")
    assert r.search_gds_path.endswith("search/p001.gds")
    assert r.search_path.endswith("search/p001.png")


def test_withheld_accessor_is_empty_on_a_blind_split(tmp_path):
    """The organizer's warning, as a test: needing those fields fails on data
    we hold, not on the scored run. So the accessor never surfaces them."""
    row = dict(TRAIN_ROW, reference_sem_path="", params_json_path="")
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [row])
    withheld = read_pairs(p)[0].training_only_fields
    assert withheld == {"reference_sem_path": "", "params_json_path": ""}


def test_withheld_accessor_refuses_to_leak_even_if_a_blind_row_carries_values(
        tmp_path):
    """Split is detected from content, never trusted from a caller flag. A
    blind row that *does* carry values (a mislabelled file) still reports
    nothing, so a training path cannot leak into inference."""
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [TRAIN_ROW])
    r = read_pairs(p)[0]
    # Same row, forced to the blind side: the accessor must still blank it.
    forced_blind = PairsRow(**{**r.__dict__, "present_in_training": False})
    assert forced_blind.training_only_fields == {
        "reference_sem_path": "", "params_json_path": ""}
    assert forced_blind.is_blind


def test_inference_fields_exclude_the_withheld_columns():
    """The disjointness the blind-split guarantee rests on."""
    assert set(inference_fields()).isdisjoint(set(WITHHELD_FIELDS))
    assert set(inference_fields()) | set(WITHHELD_FIELDS) == set(PHASE3_FIELDS)


def test_paths_resolve_relative_to_the_csv(tmp_path):
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [TRAIN_ROW])
    r = read_pairs(p)[0]
    assert os.path.isabs(r.reference_gds_path)
    assert os.path.dirname(r.reference_gds_path) == os.path.join(
        str(tmp_path), "reference")
    # ...and can be left as-written for manifest round-trips.
    raw = read_pairs(p, absolute_paths=False)[0]
    assert raw.reference_gds_path == "reference/p001.gds"


def test_absolute_paths_are_left_alone(tmp_path):
    row = dict(TRAIN_ROW, search_path="/abs/search/p001.png")
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [row])
    assert read_pairs(p)[0].search_path == "/abs/search/p001.png"


def test_header_matching_is_case_and_space_insensitive(tmp_path):
    """The organizer publishes the header separately from the addendum, so a
    Pairs.CSV-style spelling must still resolve. The *values* pass through
    untouched."""
    header = [f"  {f.upper()} " for f in PHASE3_FIELDS]
    # DictWriter matches on the exact header string, so key the row by it.
    row = {f"  {k.upper()} ": v for k, v in TRAIN_ROW.items()}
    p = _write(tmp_path / "pairs.csv", header, [row])
    r = read_pairs(p)[0]
    assert r.pair_id == "p001"
    assert r.reference_gds_path.endswith("reference/p001.gds")
    assert r.search_path.endswith("search/p001.png")


def test_duplicate_pair_id_is_rejected(tmp_path):
    """One row per pair_id exactly once is the output contract; a duplicate in
    the input becomes a duplicate in the output."""
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [TRAIN_ROW, TRAIN_ROW])
    with pytest.raises(PairsSchemaError, match="duplicate pair_id"):
        read_pairs(p)


def test_empty_pair_id_is_rejected(tmp_path):
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER,
               [dict(TRAIN_ROW, pair_id="")])
    with pytest.raises(PairsSchemaError, match="empty pair_id"):
        read_pairs(p)


def test_no_rows_is_a_hard_stop(tmp_path):
    p = _write(tmp_path / "pairs.csv", TRAIN_HEADER, [])
    with pytest.raises(SystemExit, match="no rows"):
        read_pairs(p)


def test_explicit_mapping_overrides_a_role(tmp_path):
    """An override must be honoured -- and still exist."""
    header = [f for f in PHASE3_FIELDS]
    p = _write(tmp_path / "pairs.csv", header, [TRAIN_ROW])
    r = read_pairs(p, mapping={"pair_id": "PAIR_ID"})[0]
    assert r.pair_id == "p001"

    with pytest.raises(PairsSchemaError, match="not a column"):
        read_pairs(p, mapping={"pair_id": "nope"})


def test_extra_columns_are_tolerated(tmp_path):
    """Phase 2 manifests carry many more columns (gt_x, magnification, ...);
    Phase 3 readers must not choke on the superset."""
    header = TRAIN_HEADER + ["magnification", "rotation_deg", "gt_x"]
    row = dict(TRAIN_ROW, magnification="10.0", rotation_deg="0.0", gt_x="1.0")
    p = _write(tmp_path / "pairs.csv", header, [row])
    assert read_pairs(p)[0].pair_id == "p001"


# --------------------------------------------------------------------------
# 2. Aliases, and the ambiguity that is now an error
# --------------------------------------------------------------------------

def test_canonical_is_the_only_accepted_spelling():
    """Phase 3 publishes its six column names on the slide, so those are the
    contract. A Phase 1/2-style header is resolved via an explicit mapping,
    never by guessing that `reference_path` (an SEM image in Phase 2) means
    `reference_gds_path` (a design file in Phase 3)."""
    assert resolve_column(["pair_id", "search_path", "reference_gds_path"],
                          "reference_gds_path") == "reference_gds_path"
    with pytest.raises(PairsSchemaError):
        resolve_column(["pair_id", "search_path", "reference_path"],
                       "reference_gds_path")


def test_phase2_style_header_resolves_via_explicit_mapping():
    """The supported path for a differently-spelled header: say so. Every
    Phase 3 role must be mapped, because the reader needs all six."""
    mapping = {
        "pair_id": "id",
        "search_path": "search_path",
        "reference_gds_path": "reference_path",
        "search_gds_path": "search_gds_path",
        "reference_sem_path": "reference_sem_path",
        "params_json_path": "params_json_path",
    }
    schema = resolve_schema(["id", "search_path", "reference_path",
                             "search_gds_path", "reference_sem_path",
                             "params_json_path"], mapping=mapping)
    assert schema["pair_id"] == "id"
    assert schema["reference_gds_path"] == "reference_path"


def test_phase3_header_resolves_cleanly_by_canonical_names():
    """The whole reason the Phase 3 layout is readable at all: its six columns
    are the canonical spellings, so every role resolves by exact match. This
    is what register.py's substring fallback could not do -- it had no
    canonical `reference` column to match and fell through to a prefix scan."""
    schema = resolve_schema(list(PHASE3_FIELDS))
    assert schema == {role: role for role in PHASE3_FIELDS}


def test_a_mixed_header_raises_rather_than_guessing():
    """A header with a Phase 2 reference and no canonical GDS column must not
    have `reference_path` silently promoted to the design-file role."""
    with pytest.raises(PairsSchemaError) as ei:
        resolve_schema(["pair_id", "search_path", "reference_path",
                        "reference_sem_path"])
    msg = str(ei.value)
    assert "reference_gds_path" in msg, msg
    assert "reference_path" in msg, msg


def test_near_miss_diagnostic_names_the_candidates():
    """A failed lookup should say which columns were close, not just 'not
    found' -- otherwise the fix is a guessing game."""
    with pytest.raises(PairsSchemaError) as ei:
        resolve_column(["pair_id", "search_path", "reference_gds_path"],
                       "reference_sem_path")
    msg = str(ei.value)
    assert "resembling this role" in msg
    assert "reference_gds_path" in msg


def test_duplicate_header_columns_are_rejected(tmp_path):
    header = list(PHASE3_FIELDS) + ["reference_gds_path"]
    p = _write(tmp_path / "pairs.csv", header, [TRAIN_ROW])
    with pytest.raises(PairsSchemaError, match="duplicate column"):
        read_pairs(p)


@pytest.mark.parametrize("bad", ["", "pair_id", "pair_id,search_path"])
def test_missing_columns_raise_not_silently_default(tmp_path, bad):
    if not bad:
        (tmp_path / "empty.csv").write_text("")
        with pytest.raises(PairsSchemaError, match="no header row"):
            read_pairs(str(tmp_path / "empty.csv"))
        return
    p = _write(tmp_path / "pairs.csv", bad.split(","), [{}])
    with pytest.raises(PairsSchemaError):
        read_pairs(p)


# --------------------------------------------------------------------------
# 3. register.py -- the fix, and the Phase 1/2 headers it must not break
# --------------------------------------------------------------------------

# Every real layout this repo feeds register.py. Collected from the tests and
# scripts that build pairs/manifest files; all of them must keep resolving.
PHASE2_HEADERS = [
    (["pair_id", "search_path", "reference_path"], "reference_path", "search_path"),
    (["pair_id", "reference", "search"], "reference", "search"),
    (["pair_id", "reference_path", "search_path"], "reference_path", "search_path"),
    (["id", "reference_path", "search_path", "gt_x", "gt_y"],
     "reference_path", "search_path"),
    (["id", "architecture", "reference_path", "search_path"],
     "reference_path", "search_path"),
]

# The header that made register.py's fallback dangerous: NO column is an exact
# match for REF_KEYS, and TWO columns share the `reference` prefix. The old
# code returned the first, which is a .gds path.
AMBIGUOUS_HEADER = ["pair_id", "search_path", "reference_gds_path",
                    "search_gds_path", "reference_sem_path", "params_json_path"]


@pytest.mark.parametrize("fields,ref,sea", PHASE2_HEADERS)
def test_register_pick_column_still_resolves_every_phase2_header(
        fields, ref, sea):
    """The fix must be invisible on Phase 2 data: same column, same answer."""
    reg = _load_register()
    assert reg.pick_column(fields, reg.REF_KEYS, "reference") == ref
    assert reg.pick_column(fields, reg.SEA_KEYS, "search") == sea


def test_register_pick_column_rejects_the_ambiguous_phase3_reference_role():
    """The regression. Two `reference*` columns and no exact match: resolving
    the reference role must abort the batch, not guess and decline every pair."""
    reg = _load_register()
    with pytest.raises(SystemExit) as ei:
        reg.pick_column(AMBIGUOUS_HEADER, reg.REF_KEYS, "reference")
    msg = str(ei.value)
    assert "ambiguous" in msg
    assert "reference_gds_path" in msg and "reference_sem_path" in msg


def test_register_pick_column_still_finds_the_unambiguous_search_role():
    """`search_path` is an exact match on the Phase 3 header, so it keeps
    working -- the failure is specific to the role that became ambiguous."""
    reg = _load_register()
    assert reg.pick_column(AMBIGUOUS_HEADER, reg.SEA_KEYS, "search") == "search_path"


def test_register_pick_column_missing_column_still_reports_not_found():
    """The original not-found path must survive: it never used the fallback
    and its message is relied on by other tests."""
    reg = _load_register()
    with pytest.raises(SystemExit, match="could not find the reference column"):
        reg.pick_column(["pair_id", "search_path"], reg.REF_KEYS, "reference")


def test_register_pick_column_is_exact_match_first():
    """An exact canonical spelling wins even when a prefix-sharing column is
    also present -- the Phase 2 `reference_path` vs `reference_sem_path` case."""
    reg = _load_register()
    fields = ["pair_id", "reference_path", "reference_sem_path", "search_path"]
    assert reg.pick_column(fields, reg.REF_KEYS, "reference") == "reference_path"
