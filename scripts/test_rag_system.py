"""
RAG System Test Script

Run this script to test the RAG system components
Usage: python scripts/test_rag_system.py
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_qdrant_connection():
    """Test 1: Qdrant Vector Store Connection"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Qdrant Vector Store Connection")
    logger.info("=" * 60)
    
    try:
        from app.rag.vector_store import vector_store
        
        # Health check
        if vector_store.health_check():
            logger.info("✓ Qdrant connection successful")
            
            # Get collection info
            info = vector_store.get_collection_info()
            logger.info(f"✓ Collection: {info['name']}")
            logger.info(f"  Vectors: {info.get('vectors_count', 0)}")
            logger.info(f"  Status: {info.get('status', 'unknown')}")
            return True
        else:
            logger.error("✗ Qdrant health check failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}")
        return False


def test_embeddings():
    """Test 2: Embeddings Generation"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Embeddings Generation")
    logger.info("=" * 60)
    
    try:
        from app.rag.embeddings import embeddings_service
        
        test_text = "What are your business hours?"
        
        # Generate embedding
        logger.info(f"Generating embedding for: '{test_text}'")
        embedding = embeddings_service.generate_embedding(test_text)
        
        logger.info(f"✓ Embedding generated successfully")
        logger.info(f"  Dimensions: {len(embedding)}")
        logger.info(f"  First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Embeddings generation failed: {e}")
        return False


def test_retrieval():
    """Test 3: Semantic Search Retrieval"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Semantic Search Retrieval")
    logger.info("=" * 60)
    
    try:
        from app.rag.retriever import retriever
        
        test_queries = [
            "What are your business hours?",
            "How do I return a product?",
            "What smartphones do you sell?"
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            
            results = retriever.retrieve(query, top_k=3)
            
            if results:
                logger.info(f"✓ Retrieved {len(results)} documents")
                for i, result in enumerate(results, 1):
                    title = result['payload'].get('title', 'Untitled')
                    score = result['score']
                    logger.info(f"  {i}. {title} (score: {score:.3f})")
            else:
                logger.warning(f"  No documents retrieved for this query")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Retrieval failed: {e}")
        return False


async def test_rag_generation():
    """Test 4: Full RAG Generation"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Full RAG Response Generation")
    logger.info("=" * 60)
    
    try:
        from app.rag import generate_rag_response_async
        
        test_queries = [
            {
                'query': "What are your business hours?",
                'history': []
            },
            {
                'query': "Can I return a product I bought last week?",
                'history': [
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hello! How can I help you?'}
                ]
            },
            {
                'query': "How much does the iPhone cost?",
                'history': []
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            logger.info(f"\n[Test {i}] Query: '{test['query']}'")
            logger.info("-" * 60)
            
            response = await generate_rag_response_async(
                query=test['query'],
                conversation_history=test['history'],
                user_id=999  # Test user ID
            )
            
            logger.info(f"✓ Response generated successfully")
            logger.info(f"  Response: {response['text'][:200]}...")
            logger.info(f"  Docs retrieved: {response['docs_retrieved']}")
            logger.info(f"  Tokens used: {response['tokens']}")
            logger.info(f"  Retrieval time: {response['retrieval_time_ms']}ms")
            logger.info(f"  Generation time: {response['generation_time_ms']}ms")
            logger.info(f"  Total time: {response['total_time_ms']}ms")
            
            if response.get('sources'):
                logger.info(f"  Sources:")
                for source in response['sources'][:3]:
                    logger.info(f"    - {source['title']} (score: {source['score']:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ RAG generation failed: {e}")
        logger.error(f"  Make sure OPENAI_API_KEY is set in .env")
        return False


def test_performance():
    """Test 5: Performance Metrics"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Performance Check")
    logger.info("=" * 60)
    
    try:
        from app.rag import generate_rag_response_async
        import time
        
        query = "What are your business hours?"
        
        # Run multiple times to get average
        times = []
        for i in range(3):
            start = time.time()
            response = asyncio.run(generate_rag_response_async(query, user_id=999))
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
        avg_time = sum(times) / len(times)
        
        logger.info(f"✓ Performance test complete")
        logger.info(f"  Average response time: {avg_time:.0f}ms")
        logger.info(f"  Min: {min(times):.0f}ms, Max: {max(times):.0f}ms")
        
        # Check against targets
        if avg_time < 5000:
            logger.info(f"  ✓ Performance target met (<5000ms)")
        else:
            logger.warning(f"  ⚠ Performance slower than target (5000ms)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "#" * 60)
    logger.info("# RAG SYSTEM TESTING")
    logger.info("#" * 60)
    
    tests = [
        ("Qdrant Connection", test_qdrant_connection),
        ("Embeddings Generation", test_embeddings),
        ("Semantic Retrieval", test_retrieval),
        ("RAG Generation", lambda: asyncio.run(test_rag_generation())),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"\n✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status} - {test_name}")
    
    passed = sum(1 for s in results.values() if s)
    total = len(results)
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Your RAG system is working correctly.")
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        logger.warning("Common issues:")
        logger.warning("  - OPENAI_API_KEY not set in .env")
        logger.warning("  - Qdrant not running (docker compose up qdrant)")
        logger.warning("  - No documents ingested (run scripts/ingest_sample_docs.py)")
    
    logger.info("\n" + "#" * 60)


if __name__ == "__main__":
    main()
