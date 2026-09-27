import os, json, re
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from guardrails import mask_pii, detect_prompt_injection, validate_output_groundedness
from tools import check_patient_triage_status
from knowledge_base import KB_DOCUMENTS, FALLBACK_THRESHOLD
from rag_core import get_rag_core

class AgentState(TypedDict):
    session_id: str
    user_query: str
    sanitized_query: str
    pii_redacted: Dict[str, int]
    intent: str
    patient_id: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    tool_output: Optional[Dict[str, Any]]
    final_response: str
    reasoning_steps: List[str]
    groundedness_score: float
    is_blocked: bool
    block_reason: Optional[str]
    turn_count: int

def input_guardrail_node(state: AgentState) -> AgentState:
    query = state.get('user_query', '')
    sanitized, counts = mask_pii(query)
    is_injection = detect_prompt_injection(sanitized)
    reasoning = state.get('reasoning_steps', [])
    reasoning.append(f'Node 1 (Input Guardrail): PII Masked={sum(counts.values())} entities. Injection Detected={is_injection}')
    state['sanitized_query'] = sanitized
    state['pii_redacted'] = counts
    state['is_blocked'] = is_injection
    if is_injection:
        state['block_reason'] = 'PROMPT_INJECTION_DETECTED'
        state['intent'] = 'GUARDRAIL_BLOCKED'
    return state

def intent_router_node(state: AgentState) -> AgentState:
    if state.get('is_blocked', False):
        return state
    query = state.get('sanitized_query', '')
    reasoning = state.get('reasoning_steps', [])
    pat_match = re.search('PLS-PAT-\\d{4}|PAT-\\d{4}', query, re.IGNORECASE)
    if pat_match or any((w in query.lower() for w in ['patient status', 'triage status', 'admission status', 'vitals', 'ward'])):
        if pat_match:
            pid = pat_match.group(0).upper()
            if not pid.startswith('PLS-'):
                pid = 'PLS-' + pid
            state['patient_id'] = pid
        else:
            state['patient_id'] = 'PLS-PAT-1004'
        state['intent'] = 'PATIENT_TRIAGE'
        reasoning.append(f"Node 2 (Intent Router): Classified as PATIENT_TRIAGE for {state['patient_id']}")
    else:
        state['intent'] = 'CLINICAL_RAG'
        reasoning.append('Node 2 (Intent Router): Classified as CLINICAL_RAG')
    return state

def rag_agent_node(state: AgentState) -> AgentState:
    query = state.get('sanitized_query', '')
    reasoning = state.get('reasoning_steps', [])
    lower = query.lower().strip()
    is_greeting = bool(re.search('^(hi|hello|hey|greetings|who are you|what are your capabilities|what can you do)\\b', lower)) or lower in ['hi', 'hello', 'hey', 'help', 'greetings']
    if is_greeting:
        state['final_response'] = 'Hello! I am PulseAI, your Autonomous Healthcare Operations & Clinical RAG Copilot. I can help you retrieve official hospital clinical guidelines, check patient emergency triage status & vitals, explain cashless health insurance policies, and guide you on admission KYC requirements. How may I assist you today?'
        state['retrieved_docs'] = []
        reasoning.append('Node 3 (Clinical RAG): Greeting/Capabilities intent resolved.')
        return state
    rag = get_rag_core()
    results = rag.retrieve(query, strategy='sentence', top_k=2)
    if not results or results[0]['similarity'] < FALLBACK_THRESHOLD:
        sim = results[0]['similarity'] if results else 0.0
        state['final_response'] = "I am very sorry, but I do not have information regarding this inquiry as it is outside PulseAI's clinical guidelines and hospital operations. If you would like to know about our healthcare services, emergency ICU triage criteria, patient admission tracking, or cashless insurance guidelines, please feel free to ask—I would be more than happy to assist you!"
        state['retrieved_docs'] = []
        reasoning.append(f'Node 3 (Clinical RAG): Top similarity {sim:.2f} < Fallback {FALLBACK_THRESHOLD}. Triggered humble out-of-scope fallback.')
    else:
        top_doc = results[0]
        state['retrieved_docs'] = results
        state['final_response'] = f"According to PulseAI Clinical Protocol ({top_doc['doc_id']} - {top_doc['title']}):\n\n{top_doc['content']}\n\nPlease let me know if you require further assistance regarding clinical admission or medical guidelines."
        reasoning.append(f"Node 3 (Clinical RAG): Retrieved {top_doc['doc_id']} with similarity {top_doc['similarity']:.2f}")
    return state

def patient_triage_node(state: AgentState) -> AgentState:
    pid = state.get('patient_id', 'PLS-PAT-1004')
    reasoning = state.get('reasoning_steps', [])
    tool_res = check_patient_triage_status(pid)
    state['tool_output'] = tool_res
    if tool_res.get('status') == 'error':
        state['final_response'] = f"⚠️ Patient Record Not Found: {tool_res.get('error_message')}"
        reasoning.append(f'Node 4 (Patient Triage): Error looking up {pid}')
    else:
        crit_str = '🚨 YES (CRITICAL REVIEW)' if tool_res['flagged_for_critical_icu_review'] else 'NO (Normal SLA)'
        escalate_str = '🔴 HIGH PRIORITY ICU ESCALATION TRIGGERED' if tool_res['requires_icu_escalation'] else '🟢 NORMAL CLINICAL OBSERVATION'
        state['final_response'] = f"🏥 **PulseAI Patient Triage & Clinical Summary**\n\n• **Patient Record**: `{tool_res['record_id']}` ({tool_res['patient_name']})\n• **Department & Diagnosis**: {tool_res['department']} — *{tool_res['diagnosis']}*\n• **Admission Status**: {tool_res['admission_status']} ({tool_res['assigned_ward']})\n• **Current Vitals**: SpO2: **{tool_res['sp_o2_percentage']}%** | HR: **{tool_res['heart_rate_bpm']} bpm** | BP: **{tool_res['systolic_bp_mmhg']} mmHg**\n• **Duration Admitted**: {tool_res['days_admitted']} days\n• **Critical ICU Review Flag**: {crit_str}\n• **Clinical Risk Score**: **`{tool_res['clinical_risk_score']:.4f} / 1.0000`** *(ICU Threshold: 0.65)*\n\n**Status**: {escalate_str}\n**Action**: {tool_res['action_summary']}"
        reasoning.append(f"Node 4 (Patient Triage): Lookup complete. Risk Score={tool_res['clinical_risk_score']:.4f}, Escalation={tool_res['requires_icu_escalation']}")
    return state

def output_guardrail_node(state: AgentState) -> AgentState:
    reasoning = state.get('reasoning_steps', [])
    if state.get('is_blocked', False):
        state['final_response'] = '🚫 **Security Warning**: Your request contains restricted system override or prompt injection instructions. PulseAI strictly processes healthcare operations and clinical policy queries.'
        state['groundedness_score'] = 1.0
        reasoning.append('Node 5 (Output Guardrail): Request blocked with security warning.')
        return state
    resp = state.get('final_response', '')
    retrieved = ' '.join([d.get('content', '') for d in state.get('retrieved_docs', [])])
    score = validate_output_groundedness(resp, retrieved)
    state['groundedness_score'] = score
    reasoning.append(f'Node 5 (Output Guardrail): Groundedness verified at {score:.2f}')
    sid = state.get('session_id', 'default_session')
    conv_dir = os.path.join(os.path.dirname(__file__), 'data/conversations')
    os.makedirs(conv_dir, exist_ok=True)
    conv_file = os.path.join(conv_dir, f'{sid}.json')
    history = []
    if os.path.exists(conv_file):
        try:
            with open(conv_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append({'turn': len(history) + 1, 'user_query': state.get('user_query'), 'sanitized_query': state.get('sanitized_query'), 'intent': state.get('intent'), 'final_response': state.get('final_response'), 'reasoning_steps': state.get('reasoning_steps'), 'groundedness_score': state.get('groundedness_score')})
    with open(conv_file, 'w') as f:
        json.dump(history, f, indent=2)
    state['turn_count'] = len(history)
    return state

def route_intent(state: AgentState) -> str:
    intent = state.get('intent', 'CLINICAL_RAG')
    if intent == 'GUARDRAIL_BLOCKED':
        return 'output_guardrail'
    elif intent == 'PATIENT_TRIAGE':
        return 'patient_triage'
    else:
        return 'rag_agent'

def build_pulse_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node('input_guardrail', input_guardrail_node)
    workflow.add_node('intent_router', intent_router_node)
    workflow.add_node('rag_agent', rag_agent_node)
    workflow.add_node('patient_triage', patient_triage_node)
    workflow.add_node('output_guardrail', output_guardrail_node)
    workflow.set_entry_point('input_guardrail')
    workflow.add_edge('input_guardrail', 'intent_router')
    workflow.add_conditional_edges('intent_router', route_intent, {'rag_agent': 'rag_agent', 'patient_triage': 'patient_triage', 'output_guardrail': 'output_guardrail'})
    workflow.add_edge('rag_agent', 'output_guardrail')
    workflow.add_edge('patient_triage', 'output_guardrail')
    workflow.add_edge('output_guardrail', END)
    return workflow.compile()
PULSE_APP = build_pulse_graph()

def run_pulse_agent(user_query: str, session_id: str='pulse_session_001') -> Dict[str, Any]:
    initial_state = {'session_id': session_id, 'user_query': user_query, 'sanitized_query': '', 'pii_redacted': {}, 'intent': '', 'patient_id': None, 'retrieved_docs': [], 'tool_output': None, 'final_response': '', 'reasoning_steps': [], 'groundedness_score': 1.0, 'is_blocked': False, 'block_reason': None, 'turn_count': 1}
    return PULSE_APP.invoke(initial_state)
if __name__ == '__main__':
    print('Testing PulseAI Graph...')
    res = run_pulse_agent('What is the emergency triage protocol for SpO2 below 90%?')
    print('Response:', res['final_response'])
    print('Reasoning:', res['reasoning_steps'])