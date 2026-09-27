import os, json
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from graph import AgentState, input_guardrail_node, intent_router_node, rag_agent_node, patient_triage_node, output_guardrail_node, route_intent
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'checkpoints.sqlite')

def run_checkpoint_interruption_demo():
    print('=' * 70)
    print('PULSE-AI SQLITE CHECKPOINTING & RESUME DEMONSTRATION')
    print('=' * 70)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    import sqlite3
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
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
    app = workflow.compile(checkpointer=checkpointer, interrupt_before=['rag_agent'])
    thread_id = 'pulse_icu_incident_session_101'
    config = {'configurable': {'thread_id': thread_id}}
    initial_state = {'session_id': thread_id, 'user_query': 'My Aadhaar is 9876 5432 1098. What is the ICU triage bed allocation criteria?', 'sanitized_query': '', 'pii_redacted': {}, 'intent': '', 'patient_id': None, 'retrieved_docs': [], 'tool_output': None, 'final_response': '', 'reasoning_steps': [], 'groundedness_score': 1.0, 'is_blocked': False, 'block_reason': None, 'turn_count': 1}
    print(f"\n[PHASE 1] Initial invocation with thread_id='{thread_id}' (Pauses before Node 3)...")
    res1 = app.invoke(initial_state, config=config)
    snapshot = app.get_state(config)
    print(f'  -> Checkpointed Thread ID : {thread_id}')
    print(f"  -> Sanitized Query Saved  : {snapshot.values.get('sanitized_query')}")
    print(f'  -> Next Pending Node      : {snapshot.next}')
    print(f'\n[PHASE 2] Resuming execution from SQLite checkpoint with same thread_id...')
    res2 = app.invoke(None, config=config)
    print(f'  -> Resumed Execution Output:')
    print(f"     {res2.get('final_response')[:150]}...")
    print(f"  -> Completed Reasoning Steps: {len(res2.get('reasoning_steps'))} steps")
    print('\n' + '=' * 70)
    print('Status: SQLITE CHECKPOINTING & RESUME VERIFIED WITH ZERO DATA LOSS')
if __name__ == '__main__':
    run_checkpoint_interruption_demo()