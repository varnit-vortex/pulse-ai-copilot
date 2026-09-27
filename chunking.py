import re
from typing import List, Dict, Any

def chunk_fixed_size_with_overlap(text: str, chunk_size: int=220, overlap: int=40) -> List[str]:
    chunks = []
    stride = chunk_size - overlap
    start = 0
    while start < len(text):
        c = text[start:start + chunk_size].strip()
        if c:
            chunks.append(c)
        start += stride
    return chunks

def chunk_sentence_based(text: str) -> List[str]:
    sentences = re.split('(?<=[.!?])\\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_all_documents(docs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    fixed_chunks = []
    sent_chunks = []
    for doc in docs:
        fc = chunk_fixed_size_with_overlap(doc['content'])
        for idx, chunk_text in enumerate(fc):
            fixed_chunks.append({'chunk_id': f"{doc['doc_id']}_FIXED_{idx:02d}", 'doc_id': doc['doc_id'], 'title': doc['title'], 'category': doc.get('category', 'General'), 'content': chunk_text})
        sc = chunk_sentence_based(doc['content'])
        for idx, chunk_text in enumerate(sc):
            sent_chunks.append({'chunk_id': f"{doc['doc_id']}_SENT_{idx:02d}", 'doc_id': doc['doc_id'], 'title': doc['title'], 'category': doc.get('category', 'General'), 'content': chunk_text})
    return {'fixed_chunks': fixed_chunks, 'sentence_chunks': sent_chunks}