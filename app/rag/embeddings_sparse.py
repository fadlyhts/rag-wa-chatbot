import logging
from typing import List, Dict, Any, Union, Iterable
import qdrant_client.models as models

logger = logging.getLogger(__name__)

class SparseEmbeddings:
    """Sparse Embeddings using fastembed (BM25/SPLADE)"""
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
        
        # Lazy initialization
        try:
            self._init_model()
            self._initialized = True
        except Exception as e:
            logger.warning(f"FastEmbed not available at startup: {e}")
            logger.warning("Sparse embeddings will be initialized on first use")
            
    def _init_model(self):
        try:
            from fastembed import SparseTextEmbedding
            logger.info(f"Initializing FastEmbed sparse model: {self.model_name}")
            self.model = SparseTextEmbedding(model_name=self.model_name)
            logger.info(f"✅ FastEmbed sparse model loaded")
        except ImportError:
            logger.error("fastembed library not installed! Run: pip install fastembed")
            raise
        except Exception as e:
            logger.error(f"Failed to load fastembed sparse model: {e}")
            raise

    def _ensure_initialized(self):
        if not self._initialized or self.model is None:
            self._init_model()
            self._initialized = True

    def generate_sparse_embeddings_batch(self, texts: List[str]) -> List[models.SparseVector]:
        """
        Generate sparse embeddings for a batch of texts.
        
        Args:
            texts: List of strings
            
        Returns:
            List of qdrant_client.models.SparseVector objects
        """
        self._ensure_initialized()
        
        try:
            # model.embed returns a generator yielding SparseEmbedding objects
            embeddings = list(self.model.embed(texts))
            
            result = []
            for emb in embeddings:
                # SparseEmbedding has indices and values
                result.append(
                    models.SparseVector(
                        indices=emb.indices.tolist(),
                        values=emb.values.tolist()
                    )
                )
            return result
        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {e}")
            # Fallback to empty vectors if failed
            return [models.SparseVector(indices=[], values=[]) for _ in texts]
            
    def generate_sparse_embedding(self, text: str) -> models.SparseVector:
        """Generate sparse embedding for a single text"""
        return self.generate_sparse_embeddings_batch([text])[0]

