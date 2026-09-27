import os, json
from typing import Dict, Any
PATIENT_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data/patients.json')

def load_patients_data() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(PATIENT_DATA_FILE):
        from dataset import save_dataset_to_file
        save_dataset_to_file(PATIENT_DATA_FILE)
    with open(PATIENT_DATA_FILE, 'r') as f:
        records = json.load(f)
    return {r['record_id']: r for r in records}
PATIENTS_MAP = load_patients_data()

def calculate_clinical_risk_score(days_admitted: int, flagged_for_critical_icu_review: bool) -> float:
    critical_component = 0.5 if flagged_for_critical_icu_review else 0.0
    aging_component = min(days_admitted, 30) / 30.0 * 0.5
    return round(critical_component + aging_component, 4)

def check_patient_triage_status(record_id: str) -> Dict[str, Any]:
    global PATIENTS_MAP
    if not PATIENTS_MAP or record_id not in PATIENTS_MAP:
        PATIENTS_MAP = load_patients_data()
    if record_id not in PATIENTS_MAP:
        return {'status': 'error', 'error_message': f'Patient record {record_id} not found in hospital admission database.'}
    patient = PATIENTS_MAP[record_id]
    days_admitted = patient.get('days_admitted', 0)
    critical_flag = patient.get('flagged_for_critical_icu_review', False)
    risk_score = calculate_clinical_risk_score(days_admitted, critical_flag)
    requires_icu_escalation = risk_score >= 0.65
    return {'status': 'success', 'record_id': record_id, 'patient_name': patient.get('patient_name'), 'department': patient.get('department'), 'diagnosis': patient.get('diagnosis'), 'admission_status': patient.get('status'), 'sp_o2_percentage': patient.get('sp_o2_percentage'), 'heart_rate_bpm': patient.get('heart_rate_bpm'), 'systolic_bp_mmhg': patient.get('systolic_bp_mmhg'), 'days_admitted': days_admitted, 'flagged_for_critical_icu_review': critical_flag, 'assigned_ward': patient.get('assigned_ward'), 'clinical_risk_score': risk_score, 'requires_icu_escalation': requires_icu_escalation, 'action_summary': 'HIGH PRIORITY: Critical ICU Triage Escalation triggered for Chief Medical Officer review.' if requires_icu_escalation else 'Standard clinical observation protocol active within assigned ward.'}
if __name__ == '__main__':
    print('Testing patient lookup...')
    res = check_patient_triage_status('PLS-PAT-1004')
    print(json.dumps(res, indent=2))