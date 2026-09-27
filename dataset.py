import json, random
from typing import Dict, List, Any
SEED = 42
DEPARTMENTS = ['Cardiology', 'Neurology', 'Pulmonology', 'Orthopedics', 'General Medicine']
STATUSES = ['In Triage', 'Admitted', 'ICU Transferred', 'Under Observation', 'Discharged']
DIAGNOSES = {'Cardiology': ['Acute Coronary Syndrome', 'Atrial Fibrillation', 'Hypertensive Emergency', 'Congestive Heart Failure'], 'Neurology': ['Transient Ischemic Attack', 'Migraine with Aura', 'Acute Ischemic Stroke', 'Epileptic Seizure'], 'Pulmonology': ['Acute COPD Exacerbation', 'Severe Bronchial Asthma', 'Bilateral Pneumonia', 'Pulmonary Embolism'], 'Orthopedics': ['Femur Neck Fracture', 'Acute Lumbar Radiculopathy', 'Compound Tibia Fracture', 'Ligament Tear'], 'General Medicine': ['Severe Sepsis', 'Diabetic Ketoacidosis', 'Acute Gastroenteritis', 'Dengue with Thrombocytopenia']}

def generate_patient_dataset(num_records: int=50, seed: int=SEED) -> List[Dict[str, Any]]:
    random.seed(seed)
    records = []
    for i in range(1, num_records + 1):
        record_id = f'PLS-PAT-{1000 + i}'
        department = random.choice(DEPARTMENTS)
        diagnosis = random.choice(DIAGNOSES[department])
        status = random.choice(STATUSES)
        days_admitted = random.randint(0, 30)
        sp_o2 = random.randint(84, 99)
        heart_rate = random.randint(55, 145)
        systolic_bp = random.randint(90, 190)
        is_critical_flag = i in [4, 11, 19, 27, 34, 42, 49]
        records.append({'record_id': record_id, 'patient_name': f'Patient-{1000 + i}', 'department': department, 'diagnosis': diagnosis, 'status': status, 'sp_o2_percentage': sp_o2, 'heart_rate_bpm': heart_rate, 'systolic_bp_mmhg': systolic_bp, 'days_admitted': days_admitted, 'flagged_for_critical_icu_review': is_critical_flag, 'assigned_ward': f'Ward-{department[:4].upper()}-{i % 8 + 1}'})
    return records

def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(records) >= 40, f'Expected >= 40 records, got {len(records)}'
    dept_counts = {}
    for r in records:
        d = r['department']
        dept_counts[d] = dept_counts.get(d, 0) + 1
    for d, c in dept_counts.items():
        assert c >= 3, f'Department {d} has {c} records (< 3 required)'
    status_counts = {}
    for r in records:
        s = r['status']
        status_counts[s] = status_counts.get(s, 0) + 1
    for s, c in status_counts.items():
        assert c >= 1, f'Status {s} has {c} records (< 1 required)'
    critical_count = sum((1 for r in records if r['flagged_for_critical_icu_review']))
    critical_rate = critical_count / len(records) * 100.0
    assert 10.0 <= critical_rate <= 30.0, f'Critical rate {critical_rate}% outside [10%, 30%]'
    return {'total_records': len(records), 'department_counts': dept_counts, 'status_counts': status_counts, 'critical_review_count': critical_count, 'critical_review_rate_pct': round(critical_rate, 2)}

def save_dataset_to_file(filepath: str=None) -> List[Dict[str, Any]]:
    if filepath is None:
        filepath = os.path.join(base_dir, 'data/patients.json')
    records = generate_patient_dataset()
    with open(filepath, 'w') as f:
        json.dump(records, f, indent=2)
    return records
if __name__ == '__main__':
    recs = save_dataset_to_file()
    stats = validate_dataset(recs)
    print('Dataset created and validated successfully!')