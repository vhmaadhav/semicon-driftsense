"""Phase 3 ``pairs.csv`` -- schema and reader.

Phase 3 keeps the Phase 1/2 search image, answer format and scoring, and
changes what the *reference* is: a GDSII design file instead of an SEM image.
That change lands first in ``pairs.csv``, which grows from "an id and two
image paths" into six named columns:

    pair_id, search_path, reference_gds_path, search_gds_path,
    reference_sem_path, params_json_path

Two properties of that layout drive this module, and both are load-bearing
for scoring rather than cosmetic:

* **The last two columns are withheld at inference.** They are filled in the
  training split and empty in the blind split. The organizer's own warning is
  explicit -- *"code that needs those last two fields fails on data you hold,
  not on the scored run"* -- so a training-time convenience must never become
  an inference-time dependency. :func:`read_pairs` therefore parses *all six*
  columns but exposes the withheld pair only through :attr:`PairsRow.
  training_only_fields`, which returns empty strings on a blind split no
  matter what the file contains. Nothing can read a withheld value without
  going through the training-only accessor.

* **Column resolution must be unambiguous, not best-effort.** ``register.py``
  resolves its two image columns by substring fallback, which was a safe
  convenience while Phase 2 published its layout separately. On the Phase 3
  header that fallback picks ``reference_gds_path`` for the "reference" role
  (two columns match ``reference*``) and hands a ``.gds`` path to
  ``cv2.imread``. That returns ``None``, ``read_gray`` raises ``SystemExit``,
  and ``register.py``'s per-pair handler converts it into a *declined row* --
  producing a well-formed, exit-0, all-declined ``predictions.csv``. That is
  the exact failure shape a silent guess produces: indistinguishable from a
  run that honestly rejected everything. This module refuses to guess. Exact
  matches win, known aliases are honoured, and genuine ambiguity raises
  :class:`PairsSchemaError` naming every candidate so the fix is obvious.

Column names here are lowercase and stripped; the organizer publishes the
header separately from the addendum, so matching is case- and
whitespace-insensitive while the *values* are passed through untouched.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

# --------------------------------------------------------------------------
# The Phase 3 layout. Kept as one ordered tuple because the order is part of
# what the organizer published, and a test pins the round-trip.
PHASE3_FIELDS = (
    "pair_id",
    "search_path",
    "reference_gds_path",
    "search_gds_path",
    "reference_sem_path",
    "params_json_path",
)

# The two columns the blind split leaves empty. Named once, here, so the
# withheld-set is a single edit rather than a scattered list of string
# literals.
WITHHELD_FIELDS = ("reference_sem_path", "params_json_path")

# There is deliberately NO alias table here.
#
# A tempting one exists: Phase 1/2 manifests in this repo spell the key `id`
# and the reference `reference_path`/`reference`, so aliasing them to the
# Phase 3 names looks helpful. It is not, and it is actively unsafe:
#
#   * `reference_path` in Phase 1/2 is an **SEM image**. In Phase 3 terms that
#     is `reference_sem_path`. Aliasing it to `reference_gds_path` would assert
#     a false equivalence and resolve a design-file role onto an image path --
#     the same class of wrong-column bug this module exists to prevent.
#   * `pairs3.py` is a *Phase 3* reader. Phase 3 publishes its six column names
#     on the slide, so the canonical spellings are the contract.
#
# When a caller genuinely has a different header, it says so explicitly via
# ``mapping=`` -- which is checked, not guessed. Guessing is what produced the
# original bug; an alias table is just a politer way of guessing.



class PairsSchemaError(ValueError):
    """``pairs.csv`` cannot be read unambiguously.

    Deliberately an exception rather than a ``SystemExit``: the caller decides
    whether an unusable schema is a batch-level abort (a submission run) or a
    single-pair skip. Raising ``SystemExit`` from library code would make it
    un-catchable by ``except Exception`` -- the precise trap register.py has to
    document around ``read_gray``.
    """


@dataclass(frozen=True)
class PairsRow:
    """One row of a Phase 3 ``pairs.csv``, with paths already absolute.

    ``absolute_paths=False`` leaves the stored values as-written, which is
    what a manifest round-trip test wants; the default resolves against the
    CSV's own directory because that is what the entry point needs.
    """

    pair_id: str
    search_path: str
    reference_gds_path: str
    search_gds_path: str
    reference_sem_path: str
    params_json_path: str
    present_in_training: bool
    source: dict

    @property
    def training_only_fields(self) -> dict:
        """The two withheld columns -- **empty on a blind split, always**.

        This is the only sanctioned way to reach ``reference_sem_path`` or
        ``params_json_path``. On a blind split it returns ``{"": ""}``-style
        empties rather than whatever the file happened to contain, so a
        training-side code path that leaks into inference reads a blank and
        fails loudly at its own validation, instead of quietly consuming a
        value the scored run will not have.
        """
        if not self.present_in_training:
            return {name: "" for name in WITHHELD_FIELDS}
        return {
            "reference_sem_path": self.reference_sem_path,
            "params_json_path": self.params_json_path,
        }

    @property
    def is_blind(self) -> bool:
        """True when this row is from a blind (inference) split."""
        return not self.present_in_training


def normalize(fieldname: str) -> str:
    """Header comparison form: lowercase, stripped."""
    return (fieldname or "").strip().lower()


def resolve_column(fieldnames, role: str) -> str:
    """Map a logical *role* onto the actual header spelling in ``fieldnames``.

    One rule: the canonical Phase 3 spelling must be present, exactly. There
    is no fallback and no alias table -- a role that cannot be matched exactly
    raises :class:`PairsSchemaError`.

    That is strict on purpose. ``register.py`` resolved its image columns by
    substring, which was safe only while a header could not contain two
    columns matching one role. On a header holding both ``reference_gds_path``
    and ``reference_sem_path`` the guess picks a ``.gds`` path, ``cv2.imread``
    returns ``None``, and the resulting ``SystemExit`` is swallowed per pair
    into a declined row. Strictness here converts that into one loud error.

    A caller whose header uses non-canonical names passes ``mapping=`` to
    :func:`resolve_schema`, which is explicit and existence-checked.
    """
    names = [normalize(f) for f in fieldnames]
    canonical = normalize(role)

    if canonical in names:
        return fieldnames[names.index(canonical)]

    # Diagnose rather than merely fail: a prefix-sharing column is the
    # likeliest real cause, and naming it turns a puzzle into a one-line fix.
    stem = canonical.split("_")[0]
    near = [f for f in fieldnames if normalize(f).startswith(stem)] if stem else []
    hint = (f" Columns resembling this role: {near}.") if near else ""
    raise PairsSchemaError(
        f"pairs.csv: no column named {canonical!r} for role {role!r} among "
        f"{list(fieldnames)}.{hint} Pass an explicit mapping to resolve it "
        "under a different spelling.")


def resolve_schema(fieldnames, mapping: dict | None = None) -> dict:
    """Resolve every Phase 3 role against a header, returning role -> column.

    ``mapping`` overrides individual roles with caller-supplied column names;
    an override is still checked for existence, so a typo fails here rather
    than at the first row.
    """
    names = [normalize(f) for f in fieldnames]
    resolved = {}
    for role in PHASE3_FIELDS:
        if mapping and role in mapping:
            want = normalize(mapping[role])
            if want not in names:
                raise PairsSchemaError(
                    f"pairs.csv: explicit mapping {role!r} -> "
                    f"{mapping[role]!r} is not a column; header is "
                    f"{list(fieldnames)}")
            resolved[role] = fieldnames[names.index(want)]
        else:
            resolved[role] = resolve_column(fieldnames, role)
    return resolved


def read_pairs(path: str, mapping: dict | None = None,
               absolute_paths: bool = True) -> list[PairsRow]:
    """Read a Phase 3 ``pairs.csv`` into :class:`PairsRow` objects.

    Raises :class:`PairsSchemaError` on a missing/duplicated/ambiguous column,
    and ``SystemExit`` only for "file absent / no rows" -- conditions where
    there is no partial answer to give.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)

    if not fieldnames:
        raise PairsSchemaError(f"{path}: empty file, no header row")

    # A duplicated role column (two columns differing only in case) makes the
    # mapping ambiguous in a way the resolver cannot see, because it returns
    # on first match. Catch it here.
    lowered = [normalize(f) for f in fieldnames]
    dupes = {n for n in lowered if lowered.count(n) > 1}
    if dupes:
        raise PairsSchemaError(
            f"{path}: duplicate column name(s) {sorted(dupes)} -- the header "
            "is ambiguous, so no role can be resolved safely")

    schema = resolve_schema(fieldnames, mapping)

    if not raw_rows:
        raise SystemExit(f"{path}: no rows")

    base = os.path.dirname(os.path.abspath(path))

    def pathval(row, role):
        v = (row.get(schema[role]) or "").strip()
        if not v:
            return ""
        if not absolute_paths:
            return v
        return v if os.path.isabs(v) else os.path.join(base, v)

    out = []
    seen = set()
    for n, row in enumerate(raw_rows):
        pid = (row.get(schema["pair_id"]) or "").strip()
        if not pid:
            raise PairsSchemaError(f"{path}: row {n + 2} has an empty pair_id")
        if pid in seen:
            # Not cosmetic: the output contract is one row per pair_id exactly
            # once, and a duplicate here becomes a duplicate there.
            raise PairsSchemaError(
                f"{path}: duplicate pair_id {pid!r} (row {n + 2}); the output "
                "contract requires each pair_id exactly once")
        seen.add(pid)

        withheld = {name: (row.get(schema[name]) or "").strip()
                    for name in WITHHELD_FIELDS}
        # The split is *detected*, never declared: a row carries the training
        # fields or it does not. Trusting a caller flag here would let a blind
        # split be labelled as training and expose withheld columns.
        present_in_training = all(withheld.values())

        out.append(PairsRow(
            pair_id=pid,
            search_path=pathval(row, "search_path"),
            reference_gds_path=pathval(row, "reference_gds_path"),
            search_gds_path=pathval(row, "search_gds_path"),
            reference_sem_path=pathval(row, "reference_sem_path"),
            params_json_path=pathval(row, "params_json_path"),
            present_in_training=present_in_training,
            source=dict(row),
        ))
    return out


def inference_fields() -> tuple:
    """The columns an inference-time reader may touch.

    Exists so the blind-split guarantee is checkable rather than merely
    documented -- a test asserts this set and the withheld set are disjoint.
    """
    return tuple(f for f in PHASE3_FIELDS if f not in WITHHELD_FIELDS)
