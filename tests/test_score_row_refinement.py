import pandas as pd
import pytest
from scripts.score_row_refinement import compare


def frame():
    rows = []
    for i, s in enumerate(["A", "A", "B", "B", "C", "C"]):
        rows.append(
            dict(
                pair_id=str(i),
                source_group=str(i),
                set=s,
                gt_found=int(s != "C"),
                gt_x=10.0,
                gt_y=10.0,
                x=10.0,
                y=10.0,
                gt_scale=10.0,
                scale=10.0,
                gt_rot=0.0,
                theta=0.0,
                score=0.9 if s != "C" else 0.1,
            )
        )
    return pd.DataFrame(rows)


def test_pair_order_does_not_change_comparison():
    b = frame()
    result = compare(b, b.iloc[::-1], draws=20)
    assert result["delta85"] == 0
    assert result["paired95"] == [0, 0]
    assert result["baseline"]["submitted_correctness_auc"] is None


def test_changed_confidence_is_not_a_row_only_experiment():
    b = frame()
    c = b.copy()
    c.loc[0, "score"] = 0.8
    with pytest.raises(AssertionError):
        compare(b, c, draws=2)


def test_missing_pair_is_not_silently_dropped():
    with pytest.raises(ValueError, match="pair IDs differ"):
        compare(frame(), frame().iloc[:-1], draws=2)


def test_repeated_source_requires_clustered_bootstrap():
    b = frame()
    b.loc[1, "source_group"] = b.loc[0, "source_group"]
    with pytest.raises(ValueError, match="clustered"):
        compare(b, b, draws=2)
