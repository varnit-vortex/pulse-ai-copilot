import pytest
from knowledge_base import KB_DOCUMENTS, chunk_fixed_size_with_overlap, chunk_sentence_based
from rag_core import get_rag_core

def test_kb_document_count():
    assert len(KB_DOCUMENTS) == 12

def test_chunking_strategies():
    sample = KB_DOCUMENTS[0]['content']
    fc = chunk_fixed_size_with_overlap(sample)
    sc = chunk_sentence_based(sample)
    assert len(fc) >= 1
    assert len(sc) >= 1

def test_rag_retrieval_and_threshold():
    rag = get_rag_core()
    res = rag.retrieve('What is the emergency triage protocol?', strategy='sentence', top_k=2)
    assert len(res) > 0
    assert res[0]['similarity'] > 0.31