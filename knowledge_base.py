import os, json, re
from typing import List, Dict, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
KB_DOCUMENTS = [{'doc_id': 'KB-DOC-001', 'title': 'Emergency Triage and ICU Bed Allocation Criteria', 'category': 'Clinical Guidelines', 'content': 'Emergency Triage Protocol (KB-DOC-001): Patients presenting with acute respiratory distress (SpO2 < 90%), severe hypotension (Systolic BP < 90 mmHg), or acute cardiac ischemia are categorized as Level 1 Critical Triage. Immediate ICU bed allocation is mandated within 15 minutes of clinical presentation. Level 2 Urgent patients (severe pain, fever > 103F, uncontrolled hypertension) are stabilized in the Emergency Observation Ward within 45 minutes. Routine non-emergency admissions are routed to general wards subject to standard bed availability.'}, {'doc_id': 'KB-DOC-002', 'title': 'OPD Consultation Timings and Doctor Appointment Protocols', 'category': 'Hospital Operations', 'content': 'OPD Consultation Rules (KB-DOC-002): Outpatient department consultations operate Monday through Saturday from 08:00 AM to 08:00 PM IST across Super-Specialty wings. Patients may book appointments up to 14 days in advance via the PulseAI portal or hospital front desk. A grace period of 15 minutes is permitted after the scheduled slot time; beyond 15 minutes, the slot is reallocated to walk-in emergency queues. Follow-up consultations within 7 days of the primary visit are complimentary for standard reviews.'}, {'doc_id': 'KB-DOC-003', 'title': 'Cashless Health Insurance Pre-Authorization and Claim Process', 'category': 'Billing & Insurance', 'content': 'Cashless Insurance Policy (KB-DOC-003): PulseAI Healthcare is empaneled with all major IRDAI-licensed Third Party Administrators (TPAs). For planned hospital admissions, pre-authorization requests must be submitted at least 48 hours prior to admission with medical estimate certificates. For emergency admissions, cashless pre-authorization is initiated within 4 hours of bed allocation. Co-payment clauses, non-medical consumables, and room rent capping above policy limits must be settled directly by the patient at the time of final discharge.'}, {'doc_id': 'KB-DOC-004', 'title': 'Patient Admission KYC and ABHA Digital Health ID Requirements', 'category': 'Regulatory Compliance', 'content': 'Patient Admission KYC Requirements (KB-DOC-004): Under National Health Authority guidelines, all inpatient admissions require valid government photo identification (Aadhaar Card, Passport, or Voter ID) and creation or linking of an Ayushman Bharat Health Account (ABHA ID). For international or non-resident patients, a valid Passport with Medical Visa endorsements is mandatory. Emergency life-saving treatment is never delayed for lack of identity documentation; KYC verification is completed within 24 hours post-stabilization.'}, {'doc_id': 'KB-DOC-005', 'title': 'Critical Incident and Medical Dispute Grievance Redressal', 'category': 'Patient Safety', 'content': 'Medical Grievance & Dispute Redressal (KB-DOC-005): Patient grievances regarding clinical care, diagnostic delays, or billing discrepancies may be formally lodged through the Patient Relations Office within 30 days of discharge. All clinical disputes are reviewed by an independent Medical Audit Committee comprising the Chief Medical Officer and external specialists. A formal written resolution report is furnished within 7 business days, adhering to standard Clinical Establishments Act regulations.'}, {'doc_id': 'KB-DOC-006', 'title': 'Hospital Discharge Summary and Post-Operative Follow-Up', 'category': 'Clinical Guidelines', 'content': 'Discharge & Follow-Up Protocol (KB-DOC-006): Planned patient discharges are processed between 10:00 AM and 02:00 PM daily. A comprehensive Discharge Summary detailing surgical notes, medication schedules, dietary instructions, and emergency warning signs is handed over to the patient. A mandatory post-operative follow-up appointment is scheduled between 5 to 10 days post-discharge. Emergency helpline assistance is available 24x7 for any post-discharge complications.'}, {'doc_id': 'KB-DOC-007', 'title': 'Diagnostic Lab Test and Radiology Slabs Pricing Policy', 'category': 'Billing & Insurance', 'content': 'Diagnostic & Radiology Pricing Policy (KB-DOC-007): Standard routine biochemistry and hematology tests are processed with a 4-hour turnaround time at standardized institutional rates. Advanced imaging (128-Slice CT Scans, 3.0 Tesla MRI) requires prior radiologist screening and kidney function tests (Serum Creatinine) before contrast administration. In-house admitted patients receive a 15% institutional discount on all diagnostic investigations not covered by third-party health insurance.'}, {'doc_id': 'KB-DOC-008', 'title': 'Medication Administration and Pharmacy Dispensation Rules', 'category': 'Pharmacy & Safety', 'content': 'Pharmacy Dispensation Policy (KB-DOC-008): Prescription medications are dispensed exclusively by registered pharmacists against valid digital doctor prescriptions in the Electronic Health Record (EHR). High-alert medications (Narcotics, Schedule X drugs, Intravenous Insulin, Chemotherapy agents) require dual-nurse verification before administration. Patients may return unused, unsealed non-refrigerated medications within 48 hours of discharge for a full credit refund.'}, {'doc_id': 'KB-DOC-009', 'title': 'Visitor Guidelines and Intensive Care Isolation Protocols', 'category': 'Hospital Operations', 'content': 'Visitor Access & Infection Control Rules (KB-DOC-009): General ward visiting hours are strictly restricted to 04:00 PM - 06:00 PM daily, with a maximum of two visitors permitted per patient pass. In Intensive Care Units (ICU / CCU / NICU), visiting is limited to 15 minutes per day between 05:00 PM - 06:00 PM for immediate family only. Hand hygiene and sterile PPE gowns are mandatory before entering all isolation and negative-pressure respiratory chambers.'}, {'doc_id': 'KB-DOC-010', 'title': 'Organ Donation Consent and Brain-Death Certification Guidelines', 'category': 'Regulatory Compliance', 'content': 'Organ Donation Protocol (KB-DOC-010): Deceased organ donation is conducted under the Transplantation of Human Organs and Tissues Act (THOTA). Brain stem death certification requires two separate clinical evaluations conducted 6 hours apart by a board of 4 authorized medical practitioners, including a Neurologist or Neurosurgeon. Written informed consent is obtained from the legal next-of-kin before organ retrieval coordination with the National Organ & Tissue Transplant Organization (NOTTO).'}, {'doc_id': 'KB-DOC-011', 'title': 'Pediatric and Neonatal Emergency Care Guidelines', 'category': 'Clinical Guidelines', 'content': 'Pediatric Emergency Protocols (KB-DOC-011): All pediatric emergency cases (patients under 18 years) are triaged by specialized Pediatric Intensivists. Neonatal Intensive Care Unit (NICU) admissions are prioritized for preterm infants < 34 weeks gestation or severe neonatal jaundice. At least one parent or authorized guardian is permitted 24-hour continuous bedside stay in pediatric inpatient suites.'}, {'doc_id': 'KB-DOC-012', 'title': 'Tele-Consultation and Home Healthcare Service Protocols', 'category': 'Hospital Operations', 'content': 'Telemedicine & Home Care Services (KB-DOC-012): PulseAI Tele-Health connects patients with board-certified physicians via encrypted video calls adhering to the Telemedicine Practice Guidelines. Digital prescriptions issued via tele-consultation are valid across all licensed retail pharmacies for non-scheduled drugs. Home nursing, post-surgical dressing, and sample collection services are available within a 25 km radius of the main hospital facility.'}]

def save_kb_documents_to_disk(target_dir: str=None) -> None:
    if target_dir is None:
        target_dir = os.path.join(os.path.dirname(__file__), 'data/knowledge_base')
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, 'knowledge_base.json'), 'w') as f:
        json.dump(KB_DOCUMENTS, f, indent=2)
    for doc in KB_DOCUMENTS:
        filename = doc['doc_id'] + '_' + doc['title'].lower().replace(' ', '_') + '.md'
        filepath = os.path.join(target_dir, filename)
        with open(filepath, 'w') as f:
            f.write('# ' + doc['title'] + '\n**Document ID:** ' + doc['doc_id'] + ' | **Category:** ' + doc['category'] + '\n\n' + doc['content'] + '\n')

def chunk_fixed_size_with_overlap(text: str, chunk_size: int=220, overlap: int=40) -> List[str]:
    chunks = []
    stride = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks

def chunk_sentence_based(text: str) -> List[str]:
    sentences = re.split('(?<=[.!?])\\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def build_vector_collections(chroma_dir: str=None) -> Tuple[Any, Any]:
    if chroma_dir is None:
        chroma_dir = os.path.join(os.path.dirname(__file__), 'data/chroma_db')
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    fixed_col = client.get_or_create_collection('pulse_fixed_collection')
    sent_col = client.get_or_create_collection('pulse_sentence_collection')
    if fixed_col.count() == 0:
        for doc in KB_DOCUMENTS:
            f_chunks = chunk_fixed_size_with_overlap(doc['content'])
            for idx, c in enumerate(f_chunks):
                cid = doc['doc_id'] + '_FIXED_' + f'{idx:02d}'
                emb = model.encode(c).tolist()
                fixed_col.add(ids=[cid], embeddings=[emb], documents=[c], metadatas=[{'doc_id': doc['doc_id'], 'title': doc['title']}])
    if sent_col.count() == 0:
        for doc in KB_DOCUMENTS:
            s_chunks = chunk_sentence_based(doc['content'])
            for idx, c in enumerate(s_chunks):
                cid = doc['doc_id'] + '_SENT_' + f'{idx:02d}'
                emb = model.encode(c).tolist()
                sent_col.add(ids=[cid], embeddings=[emb], documents=[c], metadatas=[{'doc_id': doc['doc_id'], 'title': doc['title']}])
    return (fixed_col, sent_col)
FALLBACK_THRESHOLD = 0.31
if __name__ == '__main__':
    save_kb_documents_to_disk()
    f_col, s_col = build_vector_collections()
    print('Knowledge base created and indexed successfully in ChromaDB!')