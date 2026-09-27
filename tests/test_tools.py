import pytest
from tools import calculate_clinical_risk_score, check_patient_triage_status

def test_clinical_risk_score_formula():
    assert calculate_clinical_risk_score(0, False) == 0.0
    assert calculate_clinical_risk_score(30, False) == 0.5
    assert calculate_clinical_risk_score(30, True) == 1.0
    assert calculate_clinical_risk_score(15, True) == 0.75

def test_valid_patient_lookup():
    res = check_patient_triage_status('PLS-PAT-1004')
    assert res['status'] == 'success'
    assert res['record_id'] == 'PLS-PAT-1004'
    assert res['clinical_risk_score'] >= 0.65
    assert res['requires_icu_escalation'] is True

def test_invalid_patient_lookup():
    res = check_patient_triage_status('PLS-PAT-9999')
    assert res['status'] == 'error'