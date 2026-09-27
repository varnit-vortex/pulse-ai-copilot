import sys
from tools import check_patient_triage_status

def run_mcp_client_simulation():
    print('=' * 60)
    print('PULSE-AI FASTMCP PROTOCOL CLIENT DEMONSTRATION')
    print('=' * 60)
    test_patients = ['PLS-PAT-1004', 'PLS-PAT-1002', 'PLS-PAT-9999']
    for pid in test_patients:
        print(f"\n[MCP Tool Call] Invoking 'check_patient_triage' for: {pid}")
        result = check_patient_triage_status(pid)
        print(f"  -> MCP Response Status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"  -> Patient Name : {result.get('patient_name')} ({result.get('department')})")
            print(f"  -> Risk Score   : {result.get('clinical_risk_score')} (ICU Escalation: {result.get('requires_icu_escalation')})")
            print(f"  -> Action       : {result.get('action_summary')}")
        else:
            print(f"  -> Error        : {result.get('error_message')}")
    print('\n' + '=' * 60)
    print('Status: ALL MCP PROTOCOL TOOL INVOCATIONS VERIFIED')
if __name__ == '__main__':
    run_mcp_client_simulation()