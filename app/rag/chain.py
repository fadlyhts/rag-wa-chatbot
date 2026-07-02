"""
LCEL-based RAG Chain with Source Documents & Metadata Extraction

RAG chain using LangChain Expression Language (LCEL) that returns both:
  - answer           : LLM text response
  - source_documents : list of source documents with full metadata
                       (file_name, page_number, url, score, etc.)

Integration:
  - Uses LCELRetriever from app.rag.retriever
  - Uses GeminiLCELWrapper from app.rag.generator_gemini
  - Uses LCEL_RAG_PROMPT from app.rag.prompt_templates
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import time

# LangChain Core (LCEL)
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
)

# Project internals
from app.rag.config import rag_config
from app.rag.prompt_templates import LCEL_RAG_PROMPT
from app.rag.retriever import LCELRetriever
from app.rag.generator_gemini import GeminiLCELWrapper

logger = logging.getLogger(__name__)

# Singleton instances for the chain
_llm_instance = GeminiLCELWrapper()


def _format_docs(docs: List[Document]) -> str:
    """
    Format list of Documents into context text for LLM prompt.
    Each chunk is labeled with source number and file name.
    """
    if not docs:
        return "Tidak ada informasi relevan yang ditemukan dalam knowledge base."

    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        title      = meta.get("title", "Document")
        file_name  = meta.get("file_name", "-")
        page_num   = meta.get("page_number")
        score      = meta.get("score", 0.0)
        doc_metadata = meta.get("doc_metadata", {})
        
        doc_type = doc_metadata.get("Jenis Dokumen", "Dokumen")
        doc_number = doc_metadata.get("No. Dokumen", "")
        doc_rev = doc_metadata.get("No. Revisi", "")
        doc_title = doc_metadata.get("Judul", title)

        page_info = f" | p.{page_num}" if page_num is not None else ""
        
        header_parts = [f"--- SUMBER {i} ---"]
        header_parts.append(f"Judul: {doc_title}")
        if doc_number:
            header_parts.append(f"Nomor Dokumen: {doc_number}")
        if doc_type:
            header_parts.append(f"Jenis Dokumen: {doc_type}")
        if doc_rev:
            header_parts.append(f"Revisi: {doc_rev}")
        header_parts.append(f"Nama File: {file_name}{page_info}")
        header_parts.append(f"Relevansi: {score:.2f}")
        
        header = "\n".join(header_parts)
        
        parts.append(
            f"{header}\n\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


def extract_sources_metadata(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract key metadata from each source document.

    Returns list of dicts with file_name, page_number, url, title,
    content_type, score, chunk_index, and a text snippet.
    """
    sources: List[Dict[str, Any]] = []
    for doc in docs:
        m = doc.metadata
        sources.append({
            "file_name":    m.get("file_name"),
            "page_number":  m.get("page_number"),
            "url":          m.get("url"),
            "title":        m.get("title"),
            "document_id":  m.get("document_id"),
            "content_type": m.get("content_type"),
            "score":        round(float(m.get("score", 0.0)), 4),
            "chunk_index":  m.get("chunk_index"),
            # Text snippet for debugging/display
            "snippet":      doc.page_content[:150].replace("\n", " ") + "...",
        })
    return sources


def build_rag_chain_with_sources(
    top_k: int = rag_config.top_k,
    min_score: float = rag_config.min_score,
):
    """
    Build LCEL chain that returns:
        {
            "answer":           str,
            "source_documents": List[Document],
            "sources_metadata": List[Dict],
        }
    """
    retriever = LCELRetriever(top_k=top_k, min_score=min_score)
    llm       = _llm_instance

    # Helper for Step 3: Retrieve relevant documents
    retrieve_docs = RunnableLambda(
        lambda inputs: retriever.retrieve(inputs["question"])
    )

    # Helper for Step 5 & 6: Run LLM
    def _run_llm(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Step 5: Build messages for LLM
        prompt_value = LCEL_RAG_PROMPT.invoke({
            "context":              inputs["context"],
            "conversation_history": inputs.get("conversation_history", "No history yet."),
            "question":             inputs["question"],
        })
        # Step 6: Generate response with LLM
        return llm.invoke(prompt_value.to_messages())

    # Main LCEL chain assembly (Execution Flow)
    chain = (
        # Step 2 & 3: Pass inputs through and Retrieve documents
        RunnableParallel(
            source_documents=retrieve_docs,
            question=RunnableLambda(lambda x: x["question"]),
            conversation_history=RunnableLambda(
                lambda x: x.get("conversation_history", "No history yet.")
            ),
        )
        # Step 4: Format context from retrieved documents
        | RunnableLambda(lambda x: {
            **x,
            "context": _format_docs(x["source_documents"]),
        })
        # Step 5, 6 & 7: Run LLM and Extract source metadata for WhatsApp
        | RunnableLambda(lambda x: {
            "llm_result":       _run_llm(x),
            "source_documents": x["source_documents"],
            "sources_metadata": extract_sources_metadata(x["source_documents"]),
        })
        | RunnableLambda(lambda x: {
            "answer":           x["llm_result"].get("content", ""),
            "tokens_used":      x["llm_result"].get("tokens_used", 0),
            "source_documents": x["source_documents"],
            "sources_metadata": x["sources_metadata"],
        })
    )

    return chain


class RAGChainWithSources:
    """
    Wrapper class for easy integration with chain.py and existing jobs.
    """

    def __init__(self):
        self._chain = build_rag_chain_with_sources()

    # Format conversation history
    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        if not history:
            return "No conversation history yet."
        recent = history[-5:]  # last 5 messages
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    # Preprocess user query
    @staticmethod
    def _preprocess_query(query: str) -> str:
        query = query.strip()
        query = ' '.join(query.split())
        return query

    # Sync
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate complete RAG response."""
        start = time.time()

        # Step 1: Preprocess query
        processed_query = self._preprocess_query(query)

        # Step 2: Format conversation history
        history_str = self._format_history(conversation_history or [])

        try:
            # Step 3 to 7: LCEL Pipeline Execution
            # (Retrieves docs -> Formats context -> Builds prompt -> Runs LLM -> Extracts metadata)
            result = self._chain.invoke({
                "question":             processed_query,
                "conversation_history": history_str,
            })

            answer            = result["answer"]
            tokens_used       = result.get("tokens_used", 0)
            source_documents  = result["source_documents"]
            sources_metadata  = result["sources_metadata"]

            logger.info(
                f"[User {user_id}] LCEL RAG completed: "
                f"{int((time.time()-start)*1000)}ms, "
                f"{len(source_documents)} docs, {tokens_used} tokens"
            )

            return {
                "text":             answer,          # backward-compatible
                "answer":           answer,
                "source_documents": source_documents,
                "sources_metadata": sources_metadata,
                "sources":          sources_metadata,  # alias
                "docs_retrieved":   len(source_documents),
                "total_time_ms":    int((time.time() - start) * 1000),
                "scores":           [s.get("score", 0) for s in sources_metadata],
                "tokens":           tokens_used,
                "retrieval_time_ms":0,
                "generation_time_ms":0,
            }

        except Exception as exc:
            logger.error(f"[User {user_id}] LCEL RAG error: {exc}", exc_info=True)
            return {
                "text":             "Maaf, saya sedang mengalami gangguan. Mohon coba lagi sebentar. 🙏",
                "answer":           "",
                "source_documents": [],
                "sources_metadata": [],
                "sources":          [],
                "docs_retrieved":   0,
                "error":            str(exc),
                "total_time_ms":    int((time.time() - start) * 1000),
                "scores":           [],
                "tokens":           0,
                "retrieval_time_ms":0,
                "generation_time_ms":0,
            }

    # Async
    async def generate_response_async(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of generate_response (for FastAPI endpoints)."""
        start = time.time()

        # Step 1: Preprocess query
        processed_query = self._preprocess_query(query)

        # Step 2: Format conversation history
        history_str = self._format_history(conversation_history or [])
        retriever   = LCELRetriever()

        try:
            # Step 3: Retrieve relevant documents (async)
            docs = await retriever.aretrieve(processed_query, filters)
            
            # Step 4: Format context
            context = _format_docs(docs)

            # Step 5: Build messages for LLM
            prompt_value = LCEL_RAG_PROMPT.invoke({
                "context":              context,
                "conversation_history": history_str,
                "question":             processed_query,
            })
            
            # Step 6: Generate response with LLM
            llm_result = await _llm_instance.ainvoke(prompt_value.to_messages())
            answer = llm_result.get("content", "")
            tokens_used = llm_result.get("tokens_used", 0)

            # Step 7: Extract source metadata for WhatsApp formatting
            sources_metadata = extract_sources_metadata(docs)

            logger.info(
                f"[User {user_id}] LCEL RAG async completed: "
                f"{int((time.time()-start)*1000)}ms, "
                f"{len(docs)} docs, {tokens_used} tokens"
            )

            return {
                "text":             answer,
                "answer":           answer,
                "source_documents": docs,
                "sources_metadata": sources_metadata,
                "sources":          sources_metadata,
                "docs_retrieved":   len(docs),
                "total_time_ms":    int((time.time() - start) * 1000),
                "scores":           [s.get("score", 0) for s in sources_metadata],
                "tokens":           tokens_used,
                "retrieval_time_ms":0,
                "generation_time_ms":0,
            }

        except Exception as exc:
            logger.error(f"[User {user_id}] LCEL RAG async error: {exc}", exc_info=True)
            return {
                "text":             "Maaf, saya sedang mengalami gangguan. Mohon coba lagi sebentar. 🙏",
                "answer":           "",
                "source_documents": [],
                "sources_metadata": [],
                "sources":          [],
                "docs_retrieved":   0,
                "error":            str(exc),
                "total_time_ms":    int((time.time() - start) * 1000),
                "scores":           [],
                "tokens":           0,
                "retrieval_time_ms":0,
                "generation_time_ms":0,
            }


# Global singleton instance
rag_chain_with_sources = RAGChainWithSources()

# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility Wrappers
# ─────────────────────────────────────────────────────────────────────────────

# In case old code refers to RAGChain or rag_chain
RAGChain = RAGChainWithSources
rag_chain = rag_chain_with_sources

def generate_rag_response(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate RAG response (sync) - convenience function"""
    return rag_chain.generate_response(query, conversation_history, user_id, filters)

async def generate_rag_response_async(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate RAG response (async) - convenience function"""
    return await rag_chain.generate_response_async(query, conversation_history, user_id, filters)
