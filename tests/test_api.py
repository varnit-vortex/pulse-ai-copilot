import pytest
from fastapi.testclient import TestClient
from server import app
client = TestClient(app)

def test_health_endpoint():
    res = client.get('/health')
    assert res.status_code == 200
    assert res.json()['status'] == 'healthy'

def test_ask_endpoint():
    res = client.post('/ask', json={'query': 'What is the emergency triage protocol?', 'session_id': 'test_s1'})
    assert res.status_code == 200
    assert res.json()['status'] == 'success'
    assert 'KB-DOC-001' in res.json()['response']

def test_patient_triage_api():
    res = client.get('/patient-triage/PLS-PAT-1004')
    assert res.status_code == 200
    assert res.json()['status'] == 'success'