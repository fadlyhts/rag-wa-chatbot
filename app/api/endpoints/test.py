"""Test endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.waha_client import WAHAClient
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class SendMessageRequest(BaseModel):
    """Send message request"""
    to: str
    text: str
    session: str = "default"


@router.post("/test/send-message")
async def test_send_message(request: SendMessageRequest):
    """
    Test sending a WhatsApp message via WAHA
    
    Example:
    ```json
    {
        "to": "6285156121852",
        "text": "Hello from backend!",
        "session": "default"
    }
    ```
    """
    try:
        waha = WAHAClient(session=request.session)
        result = waha.send_message(to=request.to, text=request.text)
        
        return {
            "status": "sent",
            "result": result,
            "session_used": request.session
        }
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/sessions")
async def test_get_sessions():
    """Get all WAHA sessions"""
    try:
        waha = WAHAClient()
        sessions = waha.get_sessions()
        
        return {
            "total": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ───────────────────────────────────────────────────────────────
# RAG Chain Test — Test langsung jawaban + source_documents
# ───────────────────────────────────────────────────────────────

class RAGTestRequest(BaseModel):
    """Request body untuk test RAG chain"""
    question: str
    conversation_history: Optional[List[dict]] = []
    user_id: Optional[int] = 0


@router.post("/test/rag")
async def test_rag_chain(request: RAGTestRequest):
    """
    Test RAG chain secara langsung — mengembalikan jawaban + source_documents.

    Contoh request:
    ```json
    {
        "question": "Apa kebijakan cuti tahunan PTPN 1?",
        "conversation_history": [],
        "user_id": 1
    }
    ```

    Response:
    ```json
    {
        "answer": "Kebijakan cuti tahunan...",
        "sources_metadata": [
            {
                "file_name": "PKB_PTPN1.pdf",
                "page_number": 12,
                "url": null,
                "title": "Perjanjian Kerja Bersama",
                "score": 0.87,
                "snippet": "..."
            }
        ],
        "docs_retrieved": 3,
        "total_time_ms": 1200
    }
    ```
    """
    try:
        from app.rag.lcel_chain import rag_chain_with_sources

        logger.info(f"[RAG Test] Query: {request.question[:80]}")

        result = await rag_chain_with_sources.generate_response_async(
            query=request.question,
            conversation_history=request.conversation_history,
            user_id=request.user_id,
        )

        return {
            "answer":           result["answer"],
            "sources_metadata": result["sources_metadata"],
            "docs_retrieved":   result["docs_retrieved"],
            "total_time_ms":    result["total_time_ms"],
            # Sertakan snippet source_documents untuk debugging
            "source_documents_preview": [
                {
                    "page_content_snippet": doc.page_content[:200],
                    "metadata": doc.metadata,
                }
                for doc in result["source_documents"]
            ]
        }

    except Exception as e:
        logger.error(f"[RAG Test] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/rag")
async def test_rag_chain_get(question: str = "Halo, siapa kamu?"):
    """
    Test RAG chain via GET (mudah dicoba langsung dari browser / Swagger).
    Contoh: GET /api/test/rag?question=Apa+kebijakan+cuti
    """
    try:
        from app.rag.lcel_chain import rag_chain_with_sources

        result = await rag_chain_with_sources.generate_response_async(
            query=question,
            conversation_history=[],
            user_id=0,
        )

        return {
            "question":         question,
            "answer":           result["answer"],
            "docs_retrieved":   result["docs_retrieved"],
            "total_time_ms":    result["total_time_ms"],
            "sources_metadata": result["sources_metadata"],
        }

    except Exception as e:
        logger.error(f"[RAG Test GET] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
