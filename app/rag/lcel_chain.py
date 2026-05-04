"""
LCEL-based RAG Chain with Source Documents & Metadata Extraction

RAG chain using LangChain Expression Language (LCEL) that returns both:
  - answer           : LLM text response
  - source_documents : list of source documents with full metadata
                       (file_name, page_number, url, score, etc.)

Integration:
  - Uses existing Qdrant VectorStore (app.rag.vector_store)
  - Uses existing GeminiGenerator (app.rag.generator_gemini)
  - Qdrant payload already contains required metadata
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging

# LangChain Core (LCEL)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Project internals
from app.rag.vector_store import vector_store
from app.rag.factory import get_embeddings_service, get_generator_service
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


def _qdrant_result_to_document(result: Dict[str, Any]) -> Document:
    """
    Convert a single Qdrant search result to a LangChain Document.

    Qdrant payload (from document_processor.py) contains:
        - content       : chunk text
        - title         : document title
        - document_id   : document hash ID
        - chunk_index   : chunk order
        - total_chunks  : total chunks for this document
        - content_type  : type (faq, policy, product, etc.)
        - (optional) file_name, page_number, source_url, url
    """
    payload: Dict[str, Any] = result.get("payload", {})

    # Normalize metadata fields
    # file_name: use 'file_name' field or fallback to 'title'
    file_name: str = (
        payload.get("file_name")
        or payload.get("title", "unknown_file")
    )

    # page_number: use 'page_number' field or fallback to chunk_index
    page_number: Optional[int] = (
        payload.get("page_number")
        or payload.get("chunk_index")
    )

    # url/source_url
    source_url: Optional[str] = (
        payload.get("url")
        or payload.get("source_url")
    )

    metadata: Dict[str, Any] = {
        # Primary fields
        "file_name":    file_name,
        "page_number":  page_number,
        "url":          source_url,
        # Additional fields for debugging/analytics
        "document_id":  payload.get("document_id"),
        "title":        payload.get("title", "Untitled"),
        "content_type": payload.get("content_type", "unknown"),
        "chunk_index":  payload.get("chunk_index"),
        "total_chunks": payload.get("total_chunks"),
        "score":        result.get("score", 0.0),        # similarity score
        "qdrant_id":    result.get("id"),                # Qdrant point ID
    }

    return Document(
        page_content=payload.get("content", ""),
        metadata=metadata,
    )


class LCELRetriever:
    """
    Thin wrapper around the existing Qdrant VectorStore
    to be used as a Runnable in the LCEL chain.
    """

    def __init__(
        self,
        top_k: int = rag_config.top_k,
        min_score: float = rag_config.min_score,
    ):
        self.top_k = top_k
        self.min_score = min_score

    # Sync retrieve
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve and convert to LangChain Documents."""
        embeddings_svc = get_embeddings_service()

        query_vector = embeddings_svc.generate_embedding(query)
        raw_results = vector_store.search(
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.min_score,
            filter_conditions=filters,
        )

        docs = [_qdrant_result_to_document(r) for r in raw_results]
        logger.info(f"[LCELRetriever] Retrieved {len(docs)} documents")
        return docs

    # Async retrieve
    async def aretrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        embeddings_svc = get_embeddings_service()
        query_vector = await embeddings_svc.generate_embedding_async(query)
        raw_results = vector_store.search(
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.min_score,
            filter_conditions=filters,
        )
        docs = [_qdrant_result_to_document(r) for r in raw_results]
        logger.info(f"[LCELRetriever] Retrieved {len(docs)} documents (async)")
        return docs


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

        page_info = f" | p.{page_num}" if page_num is not None else ""
        parts.append(
            f"[Source {i}] {title} ({file_name}{page_info}) "
            f"[Relevance: {score:.2f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


class GeminiLCELWrapper:
    """
    Wraps GeminiGenerator into an LCEL-compatible callable
    (accepts LangChain message list, returns string).
    """

    def invoke(self, messages: list) -> str:
        generator = get_generator_service()

        # Convert LangChain messages to dict format used by GeminiGenerator
        converted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})

        result = generator.generate(converted)
        # GeminiGenerator returns dict with 'content' key
        return result.get("content") or result.get("text", "")

    async def ainvoke(self, messages: list) -> str:
        generator = get_generator_service()
        converted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})

        result = await generator.generate_async(converted)
        return result.get("content") or result.get("text", "")


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


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Prompt template
# ─────────────────────────────────────────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Kamu adalah asisten AI cerdas untuk chatbot WhatsApp PTPN 1. \
Jawab pertanyaan pengguna dengan akurat berdasarkan konteks yang tersedia.

Panduan:
- Gunakan HANYA informasi dari konteks di bawah
- Jika konteks tidak memuat jawaban, sampaikan dengan sopan bahwa kamu tidak memiliki informasi tersebut
- Gunakan bahasa yang ramah, singkat, dan mudah dibaca di WhatsApp
- Sertakan emoji yang relevan agar pesan lebih hidup
- Jangan mengarang informasi

Konteks dari knowledge base:
{context}

Riwayat percakapan:
{conversation_history}""",
    ),
    ("human", "{question}"),
])

# Singleton instances
_llm_instance = GeminiLCELWrapper()


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

    Usage:
        chain = build_rag_chain_with_sources()
        result = chain.invoke({
            "question": "What is the leave policy?",
            "conversation_history": "User: hello\\nAssistant: hi!",
        })
        print(result["answer"])
        print(result["sources_metadata"])

    For async:
        result = await chain.ainvoke({...})
    """
    retriever = LCELRetriever(top_k=top_k, min_score=min_score)
    llm       = _llm_instance

    # Step A: retrieve documents based on question
    retrieve_docs = RunnableLambda(
        lambda inputs: retriever.retrieve(inputs["question"])
    )

    # Step B: run LLM with context
    def _run_llm(inputs: Dict[str, Any]) -> str:
        prompt_value = _RAG_PROMPT.invoke({
            "context":              inputs["context"],
            "conversation_history": inputs.get("conversation_history", "No history yet."),
            "question":             inputs["question"],
        })
        return llm.invoke(prompt_value.to_messages())

    # Main LCEL chain
    chain = (
        # 1. Retrieve documents in parallel with input passthrough
        RunnableParallel(
            source_documents=retrieve_docs,
            question=RunnableLambda(lambda x: x["question"]),
            conversation_history=RunnableLambda(
                lambda x: x.get("conversation_history", "No history yet.")
            ),
        )
        # 2. Add context string from source_documents
        | RunnableLambda(lambda x: {
            **x,
            "context": _format_docs(x["source_documents"]),
        })
        # 3. Run LLM and package final result
        | RunnableLambda(lambda x: {
            "answer":           _run_llm(x),
            "source_documents": x["source_documents"],
            "sources_metadata": extract_sources_metadata(x["source_documents"]),
        })
    )

    return chain


class RAGChainWithSources:
    """
    Wrapper class for easy integration with chain.py and existing jobs.

    Usage:
        from app.rag.lcel_chain import rag_chain_with_sources

        result = rag_chain_with_sources.generate_response(
            query="What is the overtime policy?",
            conversation_history=[
                {"role": "user",      "content": "hello"},
                {"role": "assistant", "content": "hi, how can I help?"},
            ],
            user_id=42
        )

        # Access answer
        print(result["text"])

        # Access source metadata
        for src in result["sources_metadata"]:
            print(f"- {src['file_name']} p.{src['page_number']} (score: {src['score']})")
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

    # Sync
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete RAG response.

        Returns:
            {
                "text":             str,            <- text answer (backward-compatible)
                "answer":           str,            <- alias for "text"
                "source_documents": List[Document], <- LangChain Document objects
                "sources_metadata": List[Dict],     <- extracted metadata
                "sources":          List[Dict],     <- alias for sources_metadata
                "docs_retrieved":   int,
                "total_time_ms":    int,
            }
        """
        import time
        start = time.time()

        history_str = self._format_history(conversation_history or [])

        try:
            result = self._chain.invoke({
                "question":             query,
                "conversation_history": history_str,
            })

            answer            = result["answer"]
            source_documents  = result["source_documents"]
            sources_metadata  = result["sources_metadata"]

            logger.info(
                f"[User {user_id}] LCEL RAG completed: "
                f"{int((time.time()-start)*1000)}ms, "
                f"{len(source_documents)} docs"
            )

            return {
                "text":             answer,          # backward-compatible
                "answer":           answer,
                "source_documents": source_documents,
                "sources_metadata": sources_metadata,
                "sources":          sources_metadata,  # alias
                "docs_retrieved":   len(source_documents),
                "total_time_ms":    int((time.time() - start) * 1000),
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
        import time
        start = time.time()

        history_str = self._format_history(conversation_history or [])
        retriever   = LCELRetriever()

        try:
            # Retrieve async
            docs = await retriever.aretrieve(query, filters)
            context = _format_docs(docs)

            # LLM (GeminiGenerator async)
            prompt_value = _RAG_PROMPT.invoke({
                "context":              context,
                "conversation_history": history_str,
                "question":             query,
            })
            answer = await _llm_instance.ainvoke(prompt_value.to_messages())

            sources_metadata = extract_sources_metadata(docs)

            logger.info(
                f"[User {user_id}] LCEL RAG async completed: "
                f"{int((time.time()-start)*1000)}ms, "
                f"{len(docs)} docs"
            )

            return {
                "text":             answer,
                "answer":           answer,
                "source_documents": docs,
                "sources_metadata": sources_metadata,
                "sources":          sources_metadata,
                "docs_retrieved":   len(docs),
                "total_time_ms":    int((time.time() - start) * 1000),
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
            }


# Global singleton instance
rag_chain_with_sources = RAGChainWithSources()
