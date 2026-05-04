"""RAG module - Retrieval-Augmented Generation system"""

from app.rag.chain import (
    generate_rag_response,
    generate_rag_response_async,
    rag_chain
)
from app.rag.lcel_chain import (
    rag_chain_with_sources,
    extract_sources_metadata,
    LCELRetriever,
)
from app.rag.document_processor import document_processor
from app.rag.vector_store import vector_store
from app.rag.embeddings import embeddings_service
from app.rag.retriever import retriever
from app.rag.generator import generator

__all__ = [
    # ── Chain (sekarang berbasis LCEL) ──
    'generate_rag_response',
    'generate_rag_response_async',
    'rag_chain',
    # ── LCEL chain ──
    'rag_chain_with_sources',
    'extract_sources_metadata',
    'LCELRetriever',
    # ── Core services ──
    'document_processor',
    'vector_store',
    'embeddings_service',
    'retriever',
    'generator',
]
