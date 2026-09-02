from generator.src.phase2_audit import AUDIT_SPECS, validate_audit_specs


def test_issue45_fixture_contract() -> None:
    validate_audit_specs()
    assert len(AUDIT_SPECS) == 20
    assert sum(spec.present for spec in AUDIT_SPECS) == 16
    assert {spec.family for spec in AUDIT_SPECS} == {"dram", "finfet"}
