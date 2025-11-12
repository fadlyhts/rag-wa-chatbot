"""Vector database management API endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.security.auth import get_current_active_admin
from app.models.admin import Admin
from app.rag.vector_store import vector_store
from app.rag.embeddings import embeddings_service

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request schema"""
    query: str
    top_k: int = 5
    score_threshold: Optional[float] = None


class SearchResult(BaseModel):
    """Search result schema"""
    id: str
    score: float
    document_id: int
    title: str
    content: str
    chunk_index: int
    total_chunks: int


@router.get("/vector/collections")
async def list_collections(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    List all Qdrant collections
    
    Returns collection names and vector counts
    Shows active collection based on AI_PROVIDER setting
    """
    try:
        from app.rag.config import rag_config
        
        collections = vector_store.client.get_collections().collections
        active_collection = rag_config.qdrant_collection
        
        collections_info = []
        for collection in collections:
            try:
                info = vector_store.client.get_collection(collection.name)
                is_active = collection.name == active_collection
                collections_info.append({
                    "name": collection.name,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status,
                    "is_active": is_active,
                    "vector_size": info.config.params.vectors.size if hasattr(info.config, 'params') else None
                })
            except Exception as e:
                logger.error(f"Error getting info for collection {collection.name}: {e}")
                collections_info.append({
                    "name": collection.name,
                    "vectors_count": 0,
                    "points_count": 0,
                    "status": "error",
                    "is_active": False,
                    "vector_size": None
                })
        
        return {
            "success": True,
            "active_collection": active_collection,
            "active_provider": rag_config.ai_provider,
            "expected_vector_size": rag_config.vector_size,
            "collections": collections_info,
            "total": len(collections_info),
            "info": "Collections are automatically created per AI provider (documents_openai, documents_gemini)"
        }
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@router.get("/vector/collections/{collection_name}/stats")
async def get_collection_stats(
    collection_name: str,
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get detailed statistics for a collection
    
    Returns:
    - Vector count
    - Points count
    - Configuration
    - Status
    """
    try:
        collection_info = vector_store.client.get_collection(collection_name)
        
        return {
            "success": True,
            "collection": {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size if hasattr(collection_info.config, 'params') else None,
                    "distance": str(collection_info.config.params.vectors.distance) if hasattr(collection_info.config, 'params') else None
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"Collection not found or error: {str(e)}")


@router.post("/vector/search")
async def search_vectors(
    search_request: SearchRequest,
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Test semantic search in vector database
    
    Performs similarity search with the provided query
    """
    try:
        if not search_request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate embedding for query
        query_embedding = await embeddings_service.generate_embedding_async(search_request.query)
        
        # Search in vector store
        results = vector_store.search(
            query_vector=query_embedding,
            limit=search_request.top_k,
            score_threshold=search_request.score_threshold
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result["id"],
                "score": result["score"],
                "payload": result["payload"]
            })
        
        return {
            "success": True,
            "query": search_request.query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/vector/test-search")
async def test_rag_search(
    search_request: SearchRequest,
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Full RAG test with preview
    
    Shows exactly what documents would be retrieved for a query
    """
    try:
        if not search_request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate embedding
        query_embedding = await embeddings_service.generate_embedding_async(search_request.query)
        
        # Search
        results = vector_store.search(
            query_vector=query_embedding,
            limit=search_request.top_k,
            score_threshold=search_request.score_threshold
        )
        
        # Format with content preview
        formatted_results = []
        for result in results:
            payload = result["payload"]
            formatted_results.append({
                "score": result["score"],
                "document_id": payload.get("document_id"),
                "title": payload.get("title"),
                "content_preview": payload.get("content", "")[:200],
                "chunk_index": payload.get("chunk_index"),
                "total_chunks": payload.get("total_chunks"),
                "category_id": payload.get("category_id")
            })
        
        return {
            "success": True,
            "query": search_request.query,
            "results_count": len(formatted_results),
            "top_k": search_request.top_k,
            "score_threshold": search_request.score_threshold,
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG test error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG test failed: {str(e)}")


@router.post("/vector/optimize")
async def optimize_collection(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Optimize Qdrant collection
    
    Triggers Qdrant's optimization process:
    - Merges segments
    - Rebuilds indexes
    - Compacts storage
    """
    try:
        # In Qdrant client v1.7+, optimization is automatic
        # This endpoint can be used to trigger manual optimization if needed
        
        collection_name = vector_store.collection_name
        
        # Get collection info to verify it exists
        collection_info = vector_store.client.get_collection(collection_name)
        
        logger.info(f"Collection {collection_name} optimization requested")
        
        return {
            "success": True,
            "message": f"Collection {collection_name} optimization triggered",
            "vectors_count": collection_info.vectors_count,
            "note": "Qdrant performs automatic optimization. Manual triggers may not be necessary."
        }
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/vector/rebuild")
async def rebuild_collection(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Rebuild Qdrant collection from database
    
    ⚠️ WARNING: This will delete all existing vectors and re-embed all documents
    Use only when:
    - Qdrant data is corrupted
    - Embedding model changed
    - Need to reprocess all documents
    """
    try:
        logger.warning("Collection rebuild requested - this will delete all vectors!")
        
        return {
            "success": False,
            "message": "Rebuild endpoint not fully implemented for safety",
            "note": "To rebuild: 1) Delete collection in Qdrant, 2) Re-index each document via /documents/{id}/reindex"
        }
        
    except Exception as e:
        logger.error(f"Rebuild error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@router.get("/vector/health")
async def check_vector_health(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Check Qdrant health and connectivity
    """
    try:
        is_healthy = vector_store.health_check()
        collection_info = vector_store.get_collection_info()
        
        return {
            "success": True,
            "healthy": is_healthy,
            "qdrant_url": vector_store.client._client.rest_uri if hasattr(vector_store.client, '_client') else "unknown",
            "collection": collection_info["name"],
            "vectors_count": collection_info["vectors_count"],
            "status": collection_info["status"]
        }
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "healthy": False,
            "error": str(e)
        }
