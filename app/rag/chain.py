"""Main RAG pipeline chain — menggunakan LCEL dari lcel_chain.py"""

from typing import List, Dict, Any, Optional
import time
import logging

# ── Import LCEL chain sebagai implementasi utama ──────────────────────────────
from app.rag.lcel_chain import (
    RAGChainWithSources,
    rag_chain_with_sources,
    extract_sources_metadata,
    LCELRetriever,
)

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Main RAG pipeline orchestrator.
    Sekarang mendelegasikan ke RAGChainWithSources (LCEL) untuk mendapatkan
    answer + source_documents + metadata sekaligus.
    Interface lama tetap kompatibel 100%.
    """
    
    def __init__(self):
        # Gunakan LCEL chain sebagai backend
        self._lcel = rag_chain_with_sources

    def _preprocess_query(self, query: str) -> str:
        """Preprocess user query"""
        return ' '.join(query.strip().split())

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate RAG response (synchronous).

        Returns:
            {
              'text':             str,            ← jawaban LLM
              'answer':           str,            ← alias
              'source_documents': List[Document], ← LangChain Documents lengkap
              'sources_metadata': List[Dict],     ← [{file_name, page_number, url, score, ...}]
              'sources':          List[Dict],     ← alias (backward-compat)
              'scores':           List[float],    ← backward-compat
              'docs_retrieved':   int,
              'tokens':           int,
              'total_time_ms':    int,
            }
        """
        processed_query = self._preprocess_query(query)
        logger.info(f"[User {user_id}] Processing query: {processed_query[:100]}")

        result = self._lcel.generate_response(
            query=processed_query,
            conversation_history=conversation_history,
            user_id=user_id,
            filters=filters,
        )

        # Tambah field backward-compat yang dipakai kode lama
        result.setdefault('scores', [
            s.get('score', 0) for s in result.get('sources_metadata', [])
        ])
        result.setdefault('tokens', 0)
        result.setdefault('retrieval_time_ms', 0)
        result.setdefault('generation_time_ms', 0)

        return result
    
    async def generate_response_async(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate RAG response (asynchronous) via LCEL chain."""
        processed_query = self._preprocess_query(query)
        logger.info(f"[User {user_id}] Processing query (async): {processed_query[:100]}")

        result = await self._lcel.generate_response_async(
            query=processed_query,
            conversation_history=conversation_history,
            user_id=user_id,
            filters=filters,
        )

        result.setdefault('scores', [
            s.get('score', 0) for s in result.get('sources_metadata', [])
        ])
        result.setdefault('tokens', 0)
        result.setdefault('retrieval_time_ms', 0)
        result.setdefault('generation_time_ms', 0)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Global instance
# ─────────────────────────────────────────────────────────────────────────────
rag_chain = RAGChain()


# ── Convenience functions (backward-compatible) ───────────────────────────────
def generate_rag_response(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Generate RAG response (sync) — sekarang mengembalikan source_documents juga."""
    return rag_chain.generate_response(query, conversation_history, user_id)


async def generate_rag_response_async(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Generate RAG response (async) — sekarang mengembalikan source_documents juga."""
    return await rag_chain.generate_response_async(query, conversation_history, user_id)
