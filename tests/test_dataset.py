import pytest
from dataset import generate_patient_dataset, validate_dataset

def test_dataset_generation_and_validation():
    records = generate_patient_dataset()
    stats = validate_dataset(records)
    assert stats['total_records'] == 50
    assert 10.0 <= stats['critical_review_rate_pct'] <= 30.0
    for d, c in stats['department_counts'].items():
        assert c >= 3
    for s, c in stats['status_counts'].items():
        assert c >= 1