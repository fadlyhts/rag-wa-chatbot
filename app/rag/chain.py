"""Main RAG pipeline chain"""

from typing import List, Dict, Any, Optional
import time
import logging
from app.rag.retriever import retriever
from app.rag.generator import generator
from app.rag.prompt_templates import (
    format_conversation_history,
    format_context,
    build_messages,
    build_fallback_messages
)

logger = logging.getLogger(__name__)


class RAGChain:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self):
        self.retriever = retriever
        self.generator = generator
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess user query"""
        # Basic cleaning
        query = query.strip()
        
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        return query
    
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate RAG response (synchronous)
        
        Args:
            query: User query
            conversation_history: List of previous messages
            user_id: User ID (for logging)
            filters: Optional metadata filters for retrieval
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess query
            processed_query = self._preprocess_query(query)
            logger.info(f"[User {user_id}] Processing query: {processed_query[:100]}")
            
            # Step 2: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(
                query=processed_query,
                filters=filters
            )
            retrieval_time = time.time() - retrieval_start
            logger.info(f"[User {user_id}] Retrieved {len(retrieved_docs)} docs in {retrieval_time:.2f}s")
            
            # Step 3: Format context
            context = format_context(retrieved_docs)
            
            # Step 4: Format conversation history
            history_str = format_conversation_history(
                conversation_history or []
            )
            
            # Step 5: Build messages for LLM
            if retrieved_docs:
                messages = build_messages(
                    query=processed_query,
                    context=context,
                    conversation_history=history_str
                )
            else:
                # Fallback when no context found
                logger.warning(f"[User {user_id}] No context found, using fallback")
                messages = build_fallback_messages(
                    query=processed_query,
                    conversation_history=history_str
                )
            
            # Step 6: Generate response with LLM
            generation_start = time.time()
            llm_response = self.generator.generate(messages)
            generation_time = time.time() - generation_start
            logger.info(f"[User {user_id}] Generated response in {generation_time:.2f}s")
            
            # Step 7: Format for WhatsApp
            response_text = llm_response['text']
            formatted_messages = self.generator.format_for_whatsapp(response_text)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Extract source metadata
            sources = [
                {
                    'id': doc.get('id'),
                    'title': doc.get('payload', {}).get('title', 'Untitled'),
                    'score': doc.get('score', 0)
                }
                for doc in retrieved_docs
            ]
            
            result = {
                'text': formatted_messages[0] if len(formatted_messages) == 1 else formatted_messages,
                'sources': sources,
                'scores': [doc.get('score', 0) for doc in retrieved_docs],
                'tokens': llm_response['tokens'],
                'retrieval_time_ms': int(retrieval_time * 1000),
                'generation_time_ms': int(generation_time * 1000),
                'total_time_ms': int(total_time * 1000),
                'docs_retrieved': len(retrieved_docs)
            }
            
            logger.info(
                f"[User {user_id}] RAG complete: "
                f"{result['total_time_ms']}ms, "
                f"{result['tokens']} tokens, "
                f"{result['docs_retrieved']} docs"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[User {user_id}] Error in RAG pipeline: {e}", exc_info=True)
            
            # Return fallback response
            return {
                'text': "I apologize, but I'm having trouble processing your request right now. Please try again in a moment. 😊",
                'sources': [],
                'scores': [],
                'tokens': 0,
                'retrieval_time_ms': 0,
                'generation_time_ms': 0,
                'total_time_ms': int((time.time() - start_time) * 1000),
                'docs_retrieved': 0,
                'error': str(e)
            }
    
    async def generate_response_async(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate RAG response (asynchronous)
        
        Args:
            query: User query
            conversation_history: List of previous messages
            user_id: User ID (for logging)
            filters: Optional metadata filters for retrieval
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess query
            processed_query = self._preprocess_query(query)
            logger.info(f"[User {user_id}] Processing query (async): {processed_query[:100]}")
            
            # Step 2: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = await self.retriever.retrieve_async(
                query=processed_query,
                filters=filters
            )
            retrieval_time = time.time() - retrieval_start
            logger.info(f"[User {user_id}] Retrieved {len(retrieved_docs)} docs in {retrieval_time:.2f}s")
            
            # Step 3: Format context
            context = format_context(retrieved_docs)
            
            # Step 4: Format conversation history
            history_str = format_conversation_history(
                conversation_history or []
            )
            
            # Step 5: Build messages for LLM
            if retrieved_docs:
                messages = build_messages(
                    query=processed_query,
                    context=context,
                    conversation_history=history_str
                )
            else:
                # Fallback when no context found
                logger.warning(f"[User {user_id}] No context found, using fallback")
                messages = build_fallback_messages(
                    query=processed_query,
                    conversation_history=history_str
                )
            
            # Step 6: Generate response with LLM
            generation_start = time.time()
            llm_response = await self.generator.generate_async(messages)
            generation_time = time.time() - generation_start
            logger.info(f"[User {user_id}] Generated response in {generation_time:.2f}s")
            
            # Step 7: Format for WhatsApp
            response_text = llm_response['text']
            formatted_messages = self.generator.format_for_whatsapp(response_text)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Extract source metadata
            sources = [
                {
                    'id': doc.get('id'),
                    'title': doc.get('payload', {}).get('title', 'Untitled'),
                    'score': doc.get('score', 0)
                }
                for doc in retrieved_docs
            ]
            
            result = {
                'text': formatted_messages[0] if len(formatted_messages) == 1 else formatted_messages,
                'sources': sources,
                'scores': [doc.get('score', 0) for doc in retrieved_docs],
                'tokens': llm_response['tokens'],
                'retrieval_time_ms': int(retrieval_time * 1000),
                'generation_time_ms': int(generation_time * 1000),
                'total_time_ms': int(total_time * 1000),
                'docs_retrieved': len(retrieved_docs)
            }
            
            logger.info(
                f"[User {user_id}] RAG complete (async): "
                f"{result['total_time_ms']}ms, "
                f"{result['tokens']} tokens, "
                f"{result['docs_retrieved']} docs"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[User {user_id}] Error in RAG pipeline (async): {e}", exc_info=True)
            
            # Return fallback response
            return {
                'text': "I apologize, but I'm having trouble processing your request right now. Please try again in a moment. 😊",
                'sources': [],
                'scores': [],
                'tokens': 0,
                'retrieval_time_ms': 0,
                'generation_time_ms': 0,
                'total_time_ms': int((time.time() - start_time) * 1000),
                'docs_retrieved': 0,
                'error': str(e)
            }


# Global RAG chain instance
rag_chain = RAGChain()


# Convenience functions for easy import
def generate_rag_response(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Generate RAG response (sync) - convenience function"""
    return rag_chain.generate_response(query, conversation_history, user_id)


async def generate_rag_response_async(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Generate RAG response (async) - convenience function"""
    return await rag_chain.generate_response_async(query, conversation_history, user_id)
