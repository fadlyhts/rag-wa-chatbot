"""
LCEL-based RAG Chain dengan Source Documents & Metadata Extraction
==================================================================

Implementasi RAG chain menggunakan LangChain Expression Language (LCEL)
yang mengembalikan BOTH:
  - answer   : teks jawaban dari LLM
  - source_documents : list dokumen sumber beserta metadata lengkap
                       (file_name, page_number, url, score, dll.)

Cara integrasi dengan project ini:
  - Menggunakan VectorStore Qdrant yang sudah ada (app.rag.vector_store)
  - Menggunakan GeminiGenerator yang sudah ada (app.rag.generator_gemini)
  - Payload Qdrant sudah berisi metadata yang dibutuhkan
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging

# ─── LangChain Core (LCEL) ───────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─── Project internals ────────────────────────────────────────────────────────
from app.rag.vector_store import vector_store
from app.rag.factory import get_embeddings_service, get_generator_service
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Helper: konversi hasil Qdrant → LangChain Document
# ─────────────────────────────────────────────────────────────────────────────

def _qdrant_result_to_document(result: Dict[str, Any]) -> Document:
    """
    Konversi satu hasil pencarian Qdrant ke LangChain Document.

    Payload Qdrant (dari document_processor.py) berisi:
        - content       : isi teks chunk
        - title         : judul dokumen
        - document_id   : ID hash dokumen
        - chunk_index   : urutan chunk
        - total_chunks  : total chunk dokumen ini
        - content_type  : tipe (faq, policy, product, …)
        - (opsional) file_name, page_number, source_url, url
    """
    payload: Dict[str, Any] = result.get("payload", {})

    # ── Normalisasi field metadata ────────────────────────────────────────────
    # file_name: ambil dari field 'file_name' atau fallback ke 'title'
    file_name: str = (
        payload.get("file_name")
        or payload.get("title", "unknown_file")
    )

    # page_number: ambil dari field 'page_number' atau turunkan dari chunk_index
    page_number: Optional[int] = (
        payload.get("page_number")
        or payload.get("chunk_index")  # fallback: chunk index sebagai proxy halaman
    )

    # url/source_url
    source_url: Optional[str] = (
        payload.get("url")
        or payload.get("source_url")
    )

    metadata: Dict[str, Any] = {
        # ── Tiga field utama yang diminta ─────────────────────────────────────
        "file_name":    file_name,
        "page_number":  page_number,
        "url":          source_url,
        # ── Field tambahan berguna untuk debugging/analytics ──────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Retriever: query → List[Document]
# ─────────────────────────────────────────────────────────────────────────────

class LCELRetriever:
    """
    Wrapper tipis di atas VectorStore Qdrant yang sudah ada,
    agar bisa dipakai sebagai Runnable dalam LCEL chain.
    """

    def __init__(
        self,
        top_k: int = rag_config.top_k,
        min_score: float = rag_config.min_score,
    ):
        self.top_k = top_k
        self.min_score = min_score

    # ── sync ──────────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve + konversi ke LangChain Documents."""
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

    # ── async ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Context formatter: List[Document] → str untuk prompt
# ─────────────────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """
    Format daftar Document menjadi teks konteks untuk LLM prompt.
    Setiap chunk dilabeli dengan nomor sumber dan nama file.
    """
    if not docs:
        return "Tidak ada informasi relevan yang ditemukan dalam knowledge base."

    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        title      = meta.get("title", "Dokumen")
        file_name  = meta.get("file_name", "-")
        page_num   = meta.get("page_number")
        score      = meta.get("score", 0.0)

        page_info = f" | Hal. {page_num}" if page_num is not None else ""
        parts.append(
            f"[Sumber {i}] {title} ({file_name}{page_info}) "
            f"[Relevansi: {score:.2f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LLM wrapper: messages → str  (menggunakan GeminiGenerator yang ada)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiLCELWrapper:
    """
    Membungkus GeminiGenerator menjadi callable yang kompatibel
    dengan LCEL (menerima list pesan LangChain, mengembalikan string).
    """

    def invoke(self, messages: list) -> str:
        generator = get_generator_service()

        # Konversi LangChain messages → format dict yang dipakai GeminiGenerator
        converted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})

        result = generator.generate(converted)
        # GeminiGenerator mengembalikan dict dengan key 'content'
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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Metadata extractor: List[Document] → List[Dict]
# ─────────────────────────────────────────────────────────────────────────────

def extract_sources_metadata(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Ekstrak metadata penting dari setiap source document.

    Returns list of dicts, contoh satu item:
    {
        "file_name":    "kebijakan_penggajian.pdf",
        "page_number":  3,
        "url":          "https://ptpn1.co.id/docs/penggajian.pdf",
        "title":        "Kebijakan Penggajian PTPN 1",
        "content_type": "policy",
        "score":        0.87,
        "chunk_index":  2,
        "snippet":      "...150 karakter pertama chunk..."
    }
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
            # Cuplikan teks untuk keperluan debug/display
            "snippet":      doc.page_content[:150].replace("\n", " ") + "…",
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

# ── Singleton instances ───────────────────────────────────────────────────────
_llm_instance = GeminiLCELWrapper()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LCEL Chain builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_chain_with_sources(
    top_k: int = rag_config.top_k,
    min_score: float = rag_config.min_score,
):
    """
    Bangun LCEL chain yang mengembalikan:
        {
            "answer":           str,
            "source_documents": List[Document],   <- objek Document lengkap
            "sources_metadata": List[Dict],        <- metadata yang sudah diekstrak
        }

    Cara penggunaan:
        chain = build_rag_chain_with_sources()
        result = chain.invoke({
            "question": "Apa kebijakan cuti tahunan PTPN 1?",
            "conversation_history": "User: halo\\nAssistant: halo juga!",
        })
        print(result["answer"])
        print(result["sources_metadata"])

    Untuk async:
        result = await chain.ainvoke({...})
    """
    retriever = LCELRetriever(top_k=top_k, min_score=min_score)
    llm       = _llm_instance

    # ── Step A: retrieve documents berdasarkan question ───────────────────────
    retrieve_docs = RunnableLambda(
        lambda inputs: retriever.retrieve(inputs["question"])
    )

    # ── Step B: jalankan LLM dengan context ──────────────────────────────────
    def _run_llm(inputs: Dict[str, Any]) -> str:
        prompt_value = _RAG_PROMPT.invoke({
            "context":              inputs["context"],
            "conversation_history": inputs.get("conversation_history", "Belum ada riwayat."),
            "question":             inputs["question"],
        })
        return llm.invoke(prompt_value.to_messages())

    # ── Chain LCEL utama ─────────────────────────────────────────────────────
    # RunnableParallel mengambil source_documents sekaligus passthrough input
    chain = (
        # 1. Ambil dokumen secara paralel dengan passthrough input
        RunnableParallel(
            source_documents=retrieve_docs,
            question=RunnableLambda(lambda x: x["question"]),
            conversation_history=RunnableLambda(
                lambda x: x.get("conversation_history", "Belum ada riwayat.")
            ),
        )
        # 2. Tambahkan context string dari source_documents
        | RunnableLambda(lambda x: {
            **x,
            "context": _format_docs(x["source_documents"]),
        })
        # 3. Jalankan LLM + kemas hasil akhir
        | RunnableLambda(lambda x: {
            "answer":           _run_llm(x),
            "source_documents": x["source_documents"],
            "sources_metadata": extract_sources_metadata(x["source_documents"]),
        })
    )

    return chain


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Convenience class (drop-in untuk chain.py yang sudah ada)
# ─────────────────────────────────────────────────────────────────────────────

class RAGChainWithSources:
    """
    Kelas pembungkus agar mudah digunakan dari chain.py / jobs yang ada.

    Contoh pemakaian di process_message.py:
        from app.rag.lcel_chain import rag_chain_with_sources

        result = rag_chain_with_sources.generate_response(
            query="Apa kebijakan lembur di PTPN 1?",
            conversation_history=[
                {"role": "user",      "content": "halo"},
                {"role": "assistant", "content": "halo, ada yang bisa saya bantu?"},
            ],
            user_id=42
        )

        # Akses jawaban
        print(result["text"])

        # Akses metadata sumber
        for src in result["sources_metadata"]:
            print(f"- {src['file_name']} hal.{src['page_number']} (skor: {src['score']})")
    """

    def __init__(self):
        self._chain = build_rag_chain_with_sources()

    # ── Format conversation history ───────────────────────────────────────────
    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        if not history:
            return "Belum ada riwayat percakapan."
        recent = history[-5:]  # ambil 5 pesan terakhir
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Asisten"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    # ── Sync ─────────────────────────────────────────────────────────────────
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate RAG response lengkap.

        Returns:
            {
                "text":             str,            <- jawaban teks (kompatibel dengan chain.py lama)
                "answer":           str,            <- alias untuk "text"
                "source_documents": List[Document], <- objek Document LangChain
                "sources_metadata": List[Dict],     <- metadata yang sudah diekstrak
                "sources":          List[Dict],     <- alias singkat untuk sources_metadata
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
                f"[User {user_id}] LCEL RAG selesai: "
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

    # ── Async ────────────────────────────────────────────────────────────────
    async def generate_response_async(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version dari generate_response (untuk FastAPI endpoints)."""
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
                f"[User {user_id}] LCEL RAG async selesai: "
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


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Singleton global (import langsung dari modul lain)
# ─────────────────────────────────────────────────────────────────────────────

rag_chain_with_sources = RAGChainWithSources()
