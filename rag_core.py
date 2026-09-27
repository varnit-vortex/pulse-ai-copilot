import os, shutil
from typing import List, Dict, Any, Tuple
import chromadb
import numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from sentence_transformers import SentenceTransformer
from knowledge_base import KB_DOCUMENTS
from chunking import chunk_all_documents
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'chroma_db')
_EMBED_MODEL = None

def get_embedding_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        except Exception:
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL

class PulseRAGCore:

    def __init__(self, persist_dir: str=CHROMA_DIR, force_reindex: bool=False):
        self.persist_dir = persist_dir
        self.model = get_embedding_model()
        if force_reindex and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.fixed_col = self.client.get_or_create_collection(name='pulse_fixed_collection', metadata={'hnsw:space': 'cosine'})
        self.sent_col = self.client.get_or_create_collection(name='pulse_sentence_collection', metadata={'hnsw:space': 'cosine'})
        if self.sent_col.count() == 0 or self.fixed_col.count() == 0:
            self.index_all_documents()

    def index_all_documents(self):
        chunks_map = chunk_all_documents(KB_DOCUMENTS)
        fc = chunks_map['fixed_chunks']
        if fc and self.fixed_col.count() == 0:
            texts = [c['content'] for c in fc]
            embs = self.model.encode(texts).tolist()
            ids = [c['chunk_id'] for c in fc]
            metas = [{'doc_id': c['doc_id'], 'title': c['title']} for c in fc]
            self.fixed_col.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
        sc = chunks_map['sentence_chunks']
        if sc and self.sent_col.count() == 0:
            texts = [c['content'] for c in sc]
            embs = self.model.encode(texts).tolist()
            ids = [c['chunk_id'] for c in sc]
            metas = [{'doc_id': c['doc_id'], 'title': c['title']} for c in sc]
            self.sent_col.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)

    def retrieve(self, query: str, strategy: str='sentence', top_k: int=3) -> List[Dict[str, Any]]:
        collection = self.sent_col if strategy == 'sentence' else self.fixed_col
        q_emb = self.model.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=top_k)
        retrieved = []
        if results and results.get('documents') and (len(results['documents']) > 0):
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            ids = results['ids'][0]
            distances = results['distances'][0] if 'distances' in results else [0.0] * len(docs)
            for cid, doc_text, meta, dist in zip(ids, docs, metas, distances):
                sim = max(0.0, 1.0 - float(dist))
                retrieved.append({'chunk_id': cid, 'doc_id': meta.get('doc_id', 'KB-DOC-001'), 'title': meta.get('title', ''), 'content': doc_text, 'similarity': round(sim, 4)})
        return retrieved
_GLOBAL_RAG = None

def get_rag_core() -> PulseRAGCore:
    global _GLOBAL_RAG
    if _GLOBAL_RAG is None:
        _GLOBAL_RAG = PulseRAGCore()
    return _GLOBAL_RAG