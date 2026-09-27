import pytest
from graph import run_pulse_agent

def test_polite_greetings():
    res = run_pulse_agent('Hi, can you help me?')
    assert 'PulseAI' in res['final_response']
    assert 'How may I assist you today?' in res['final_response']

def test_critical_patient_escalation_flow():
    res = run_pulse_agent('Check status for PLS-PAT-1004')
    assert 'CRITICAL' in res['final_response']
    assert 'Escalate' in res['final_response'] or 'ICU' in res['final_response']

def test_out_of_scope_fallback():
    res = run_pulse_agent('How do I cook Italian carbonara pasta?')
    assert 'outside PulseAI' in res['final_response']
import pytest
from graph import run_pulse_agent
from knowledge_base import KB_DOCUMENTS

def test_all_12_kb_topics_retrieval():
    for doc in KB_DOCUMENTS:
        res = run_pulse_agent(doc['title'])
        assert res['intent'] == 'CLINICAL_RAG'
        assert doc['doc_id'] in res['final_response']

def test_multi_turn_session_continuity():
    s_id = 'test_multi_turn_continuity_session'
    res1 = run_pulse_agent('What are the OPD consultation timings?', session_id=s_id)
    assert res1['turn_count'] >= 1
    res2 = run_pulse_agent('Check status for patient PLS-PAT-1004', session_id=s_id)
    assert res2['turn_count'] >= 2

def test_patient_normal_sla_observation():
    res = run_pulse_agent('Check status for patient PLS-PAT-1002')
    assert 'NORMAL CLINICAL OBSERVATION' in res['final_response'] or 'Standard' in res['final_response']